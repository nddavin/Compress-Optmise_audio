import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import hashlib
import time

class ConfigManager:
    """Unified configuration manager with progressive loading, caching, and validation"""

    def __init__(self, config_file: str = "compress_audio_config.json"):
        self.config_file = Path(config_file)
        self._cache: Optional[Dict[str, Any]] = None
        self._cache_hash: Optional[str] = None
        self._last_modified: float = 0
        self._validation_errors: List[str] = []
        self.default_config: Dict[str, Any] = {
            "model_paths": {
                "arnndn_model": "/usr/local/share/ffmpeg/arnndn-models/bd.cnr.mdl",
                "custom_models_dir": "./models"
            },
            "presets": {
                "speech": {
                    "compressor": {
                        "threshold": -20,
                        "ratio": 3,
                        "attack": 0.01,
                        "release": 0.1,
                        "makeup": 6
                    },
                    "multiband": {
                        "low_freq": 250,
                        "high_freq": 4000,
                        "low_threshold": -15,
                        "low_ratio": 2.5,
                        "low_attack": 0.01,
                        "low_release": 0.1,
                        "low_makeup": 3,
                        "mid_threshold": -18,
                        "mid_ratio": 3,
                        "mid_attack": 0.005,
                        "mid_release": 0.08,
                        "mid_makeup": 4,
                        "high_threshold": -20,
                        "high_ratio": 2,
                        "high_attack": 0.002,
                        "high_release": 0.05,
                        "high_makeup": 2
                    }
                },
                "music": {
                    "compressor": {
                        "threshold": -18,
                        "ratio": 4,
                        "attack": 0.005,
                        "release": 0.05,
                        "makeup": 4
                    },
                    "multiband": {
                        "low_freq": 200,
                        "high_freq": 5000,
                        "low_threshold": -12,
                        "low_ratio": 3,
                        "low_attack": 0.02,
                        "low_release": 0.15,
                        "low_makeup": 2,
                        "mid_threshold": -15,
                        "mid_ratio": 4,
                        "mid_attack": 0.01,
                        "mid_release": 0.1,
                        "mid_makeup": 3,
                        "high_threshold": -18,
                        "high_ratio": 2.5,
                        "high_attack": 0.005,
                        "high_release": 0.08,
                        "high_makeup": 1
                    }
                },
                "custom": {}
            },
            "output_formats": {
                "mp3": {
                    "codec": "libmp3lame",
                    "ext": ".mp3",
                    "speech": [64, 96, 128],
                    "music": [128, 192, 256]
                },
                "aac": {
                    "codec": "aac",
                    "ext": ".m4a",
                    "speech": [48, 64, 96],
                    "music": [96, 128, 192]
                },
                "ogg": {
                    "codec": "libvorbis",
                    "ext": ".ogg",
                    "speech": [48, 64, 96],
                    "music": [96, 128, 160]
                },
                "opus": {
                    "codec": "libopus",
                    "ext": ".opus",
                    "speech": [24, 32, 48],
                    "music": [64, 96, 128]
                },
                "flac": {
                    "codec": "flac",
                    "ext": ".flac",
                    "speech": [],
                    "music": []
                }
            },
            "default_settings": {
                "format": "mp3",
                "content_type": "speech",
                "channels": 1,
                "loudnorm_enabled": True,
                "compressor_enabled": False,
                "multiband_enabled": False,
                "ml_noise_reduction": False
            }
        }
        self.config = self._load_config_progressive()

    def _load_config_progressive(self) -> Dict[str, Any]:
        """Load configuration progressively from multiple sources with caching"""
        # Check if cached config is still valid
        if self._is_cache_valid():
            return self._cache.copy() if self._cache is not None else self.default_config.copy()

        # Load from multiple sources in priority order
        config = self.default_config.copy()

        # Load from environment variables
        env_config = self._load_from_env()
        config = self._merge_configs(config, env_config)

        # Load from file
        file_config = self._load_from_file()
        config = self._merge_configs(config, file_config)

        # Validate and apply fallbacks
        config = self._validate_config_with_fallbacks(config)
        self._update_cache(config)

        return config

    def _is_cache_valid(self) -> bool:
        """Check if cached configuration is still valid"""
        if self._cache is None:
            return False

        # Check file modification time
        if self.config_file.exists():
            current_mtime = self.config_file.stat().st_mtime
            if current_mtime > self._last_modified:
                return False

        # Check environment variables hash
        current_env_hash = self._get_env_hash()
        if current_env_hash != self._cache_hash:
            return False

        return True

    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_config: Dict[str, Any] = {}

        # Model paths
        if "COMPRESS_AUDIO_ARNNDN_MODEL" in os.environ:
            env_config.setdefault("model_paths", {})["arnndn_model"] = os.environ["COMPRESS_AUDIO_ARNNDN_MODEL"]
        if "COMPRESS_AUDIO_CUSTOM_MODELS_DIR" in os.environ:
            env_config.setdefault("model_paths", {})["custom_models_dir"] = os.environ["COMPRESS_AUDIO_CUSTOM_MODELS_DIR"]

        # Default settings
        if "COMPRESS_AUDIO_FORMAT" in os.environ:
            env_config.setdefault("default_settings", {})["format"] = os.environ["COMPRESS_AUDIO_FORMAT"]
        if "COMPRESS_AUDIO_CONTENT_TYPE" in os.environ:
            env_config.setdefault("default_settings", {})["content_type"] = os.environ["COMPRESS_AUDIO_CONTENT_TYPE"]
        if "COMPRESS_AUDIO_CHANNELS" in os.environ:
            try:
                env_config.setdefault("default_settings", {})["channels"] = int(os.environ["COMPRESS_AUDIO_CHANNELS"])
            except ValueError:
                pass

        return env_config

    def _load_from_file(self) -> Dict[str, Any]:
        """Load configuration from file with error recovery"""
        if not self.config_file.exists():
            logging.warning(f"Config file {self.config_file} does not exist, using defaults")
            return {}

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Basic validation of loaded config
            if not isinstance(config, dict):
                raise ValueError("Configuration root must be a dictionary")

            return config

        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in config file {self.config_file}: {e}"
            logging.error(error_msg)
            self._validation_errors.append(error_msg)

            # Try to create backup and reset to defaults
            try:
                backup_file = self.config_file.with_suffix('.backup')
                if self.config_file.exists():
                    self.config_file.replace(backup_file)
                    logging.info(f"Created backup of corrupted config: {backup_file}")
            except Exception as backup_e:
                logging.warning(f"Could not create backup: {backup_e}")

            return {}

        except IOError as e:
            error_msg = f"Could not read config file {self.config_file}: {e}"
            logging.error(error_msg)
            self._validation_errors.append(error_msg)
            return {}

        except Exception as e:
            error_msg = f"Unexpected error loading config file {self.config_file}: {e}"
            logging.error(error_msg)
            self._validation_errors.append(error_msg)
            return {}

    def _get_env_hash(self) -> str:
        """Generate hash of relevant environment variables"""
        env_vars = [
            os.environ.get("COMPRESS_AUDIO_ARNNDN_MODEL", ""),
            os.environ.get("COMPRESS_AUDIO_CUSTOM_MODELS_DIR", ""),
            os.environ.get("COMPRESS_AUDIO_FORMAT", ""),
            os.environ.get("COMPRESS_AUDIO_CONTENT_TYPE", ""),
            os.environ.get("COMPRESS_AUDIO_CHANNELS", "")
        ]
        return hashlib.md5("|".join(env_vars).encode()).hexdigest()

    def _update_cache(self, config: Dict[str, Any]) -> None:
        """Update the configuration cache"""
        self._cache = config.copy()
        self._cache_hash = self._get_env_hash()
        if self.config_file.exists():
            self._last_modified = self.config_file.stat().st_mtime
        else:
            self._last_modified = 0

    def save_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Save configuration to file with error recovery"""
        if config is None:
            config = self.config

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                # Ensure parent directory exists
                self.config_file.parent.mkdir(parents=True, exist_ok=True)

                # Write to temporary file first, then rename for atomic operation
                temp_file = self.config_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(config, f, indent=2)

                # Atomic rename
                temp_file.replace(self.config_file)

                # Update cache
                self._update_cache(config)
                return

            except IOError as e:
                last_error = e
                if attempt < max_retries - 1:
                    logging.warning(f"Config save attempt {attempt + 1} failed: {e}")
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                else:
                    logging.error(f"Failed to save config after {max_retries} attempts: {e}")

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    logging.warning(f"Unexpected error during config save attempt {attempt + 1}: {e}")
                    time.sleep(0.1 * (2 ** attempt))
                else:
                    logging.error(f"Unexpected error during config save: {e}")

        # If all attempts failed, raise the last error
        if last_error:
            raise IOError(f"Could not save config file {self.config_file}: {last_error}")

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge override config into base config"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    def get_model_path(self, model_name: str) -> Optional[str]:
        """Get path for a specific model"""
        return self.config.get("model_paths", {}).get(model_name)

    def set_model_path(self, model_name: str, path: str) -> None:
        """Set path for a specific model"""
        if "model_paths" not in self.config:
            self.config["model_paths"] = {}
        self.config["model_paths"][model_name] = path
        self.save_config()

    def get_preset(self, preset_name: str, preset_type: str = "compressor") -> Optional[Dict[str, Any]]:
        """Get a preset configuration"""
        presets = self.config.get("presets", {}).get(preset_name, {})
        return presets.get(preset_type)

    def set_preset(self, preset_name: str, preset_type: str, settings: Dict[str, Any]) -> None:
        """Set a preset configuration"""
        if "presets" not in self.config:
            self.config["presets"] = {}
        if preset_name not in self.config["presets"]:
            self.config["presets"][preset_name] = {}
        self.config["presets"][preset_name][preset_type] = settings
        self.save_config()

    def get_format_config(self, format_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for an output format"""
        return self.config.get("output_formats", {}).get(format_name)

    def get_default_setting(self, setting_name: str) -> Any:
        """Get a default setting value"""
        return self.config.get("default_settings", {}).get(setting_name)

    def set_default_setting(self, setting_name: str, value: Any) -> None:
        """Set a default setting value"""
        if "default_settings" not in self.config:
            self.config["default_settings"] = {}
        self.config["default_settings"][setting_name] = value
        self.save_config()

    def _validate_config_with_fallbacks(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration and apply automatic fallbacks for missing/invalid settings"""
        self._validation_errors = []

        # Ensure required sections exist with fallbacks
        required_sections = {
            "model_paths": {},
            "presets": self.default_config.get("presets", {}),
            "output_formats": self.default_config.get("output_formats", {}),
            "default_settings": self.default_config.get("default_settings", {})
        }

        for section, fallback in required_sections.items():
            if section not in config:
                config[section] = fallback.copy()
                self._validation_errors.append(f"Missing required section '{section}', using defaults")
            elif not isinstance(config[section], dict):
                config[section] = fallback.copy()
                self._validation_errors.append(f"Invalid section '{section}' type, using defaults")

        # Validate and fix model paths
        model_paths = config.get("model_paths", {})
        for model_name, path in list(model_paths.items()):
            if not isinstance(path, str):
                del model_paths[model_name]
                self._validation_errors.append(f"Invalid model path for '{model_name}', removing")
            elif not os.path.exists(path) and model_name != "custom_models_dir":
                self._validation_errors.append(f"Model path does not exist: {path} for '{model_name}'")

        # Validate presets structure
        presets = config.get("presets", {})
        for preset_name, preset_data in list(presets.items()):
            if not isinstance(preset_data, dict):
                del presets[preset_name]
                self._validation_errors.append(f"Invalid preset '{preset_name}' structure, removing")
                continue

            # Ensure preset has valid compressor/multiband settings
            for preset_type in ["compressor", "multiband"]:
                if preset_type not in preset_data:
                    if preset_name in self.default_config.get("presets", {}):
                        preset_data[preset_type] = self.default_config["presets"][preset_name].get(preset_type, {})
                        self._validation_errors.append(f"Missing '{preset_type}' in preset '{preset_name}', using defaults")

        # Validate output formats
        output_formats = config.get("output_formats", {})
        for format_name, format_data in list(output_formats.items()):
            if not isinstance(format_data, dict):
                del output_formats[format_name]
                self._validation_errors.append(f"Invalid output format '{format_name}' structure, removing")
                continue

            required_format_keys = ["codec", "ext", "speech", "music"]
            for key in required_format_keys:
                if key not in format_data:
                    if format_name in self.default_config.get("output_formats", {}):
                        format_data[key] = self.default_config["output_formats"][format_name].get(key)
                        self._validation_errors.append(f"Missing '{key}' in format '{format_name}', using defaults")

        # Validate default settings
        default_settings = config.get("default_settings", {})
        valid_formats = list(self.default_config.get("output_formats", {}).keys())
        if "format" in default_settings and default_settings["format"] not in valid_formats:
            default_settings["format"] = self.default_config["default_settings"]["format"]
            self._validation_errors.append(f"Invalid default format, using '{default_settings['format']}'")

        valid_content_types = ["speech", "music"]
        if "content_type" in default_settings and default_settings["content_type"] not in valid_content_types:
            default_settings["content_type"] = self.default_config["default_settings"]["content_type"]
            self._validation_errors.append(f"Invalid content type, using '{default_settings['content_type']}'")

        return config

    def validate_config(self) -> Tuple[bool, List[str]]:
        """Validate the current configuration and return detailed results"""
        # Force reload to ensure validation
        self.config = self._load_config_progressive()

        is_valid = len(self._validation_errors) == 0
        return is_valid, self._validation_errors.copy()

    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults"""
        self.config = self.default_config.copy()
        self._cache = None  # Invalidate cache
        self.save_config()

    def reload_config(self) -> None:
        """Force reload configuration from all sources"""
        self._cache = None  # Invalidate cache
        self.config = self._load_config_progressive()

    def get_validation_errors(self) -> List[str]:
        """Get list of current validation errors"""
        return self._validation_errors.copy()

# Global config instance
config_manager = ConfigManager()