import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    """Manages configuration files for audio compression settings and model paths"""

    def __init__(self, config_file: str = "compress_audio_config.json"):
        self.config_file = Path(config_file)
        self.default_config = {
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
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file, creating default if it doesn't exist"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                # Merge with defaults to ensure all keys exist
                return self._merge_configs(self.default_config, loaded_config)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load config file {self.config_file}: {e}")
                print("Using default configuration.")
                return self.default_config.copy()
        else:
            # Create default config file
            self.save_config(self.default_config)
            return self.default_config.copy()

    def save_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Save configuration to file"""
        if config is None:
            config = self.config

        try:
            # Ensure parent directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except IOError as e:
            raise IOError(f"Could not save config file {self.config_file}: {e}")

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

    def validate_config(self) -> bool:
        """Validate the current configuration"""
        required_keys = ["model_paths", "presets", "output_formats", "default_settings"]

        for key in required_keys:
            if key not in self.config:
                print(f"Warning: Missing required config section: {key}")
                return False

        # Validate model paths exist if specified
        model_paths = self.config.get("model_paths", {})
        for model_name, path in model_paths.items():
            if not os.path.exists(path) and model_name != "custom_models_dir":
                print(f"Warning: Model path does not exist: {path}")

        return True

    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults"""
        self.config = self.default_config.copy()
        self.save_config()

# Global config instance
config_manager = ConfigManager()