"""
High-level workflow presets for audio processing.
Provides one-click operations for common tasks like podcast optimization,
music mastering, and batch conversion.
"""

import json
import os
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import uuid

class WorkflowStep(Enum):
    """Enumeration of available workflow steps"""
    ANALYZE = "analyze"
    COMPRESS = "compress"
    NORMALIZE = "normalize"
    COMPRESS_AUDIO = "compress_audio"
    MULTIBAND_COMPRESS = "multiband_compress"
    NOISE_REDUCTION = "noise_reduction"
    SILENCE_TRIM = "silence_trim"
    NOISE_GATE = "noise_gate"
    METADATA_PRESERVE = "metadata_preserve"
    BATCH_PROCESS = "batch_process"
    UPLOAD_CLOUD = "upload_cloud"
    STORE_OFFLINE = "store_offline"

@dataclass
class WorkflowStepConfig:
    """Configuration for a single workflow step"""
    step_type: WorkflowStep
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # Step IDs this depends on
    order: int = 0

@dataclass
class CustomWorkflow:
    """Represents a custom workflow with multiple steps"""
    id: str
    name: str
    description: str
    icon: str
    category: str
    steps: List[WorkflowStepConfig] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: __import__('time').time())
    updated_at: float = field(default_factory=lambda: __import__('time').time())

    def add_step(self, step_config: WorkflowStepConfig) -> None:
        """Add a step to the workflow"""
        step_config.order = len(self.steps)
        self.steps.append(step_config)
        self.updated_at = __import__('time').time()

    def remove_step(self, step_id: str) -> bool:
        """Remove a step by ID"""
        for i, step in enumerate(self.steps):
            if f"{step.step_type.value}_{i}" == step_id:
                self.steps.pop(i)
                # Reorder remaining steps
                for j, remaining_step in enumerate(self.steps[i:], i):
                    remaining_step.order = j
                self.updated_at = __import__('time').time()
                return True
        return False

    def get_step_by_id(self, step_id: str) -> Optional[WorkflowStepConfig]:
        """Get a step by its ID"""
        for i, step in enumerate(self.steps):
            if f"{step.step_type.value}_{i}" == step_id:
                return step
        return None

    def validate_workflow(self) -> List[str]:
        """Validate the workflow configuration"""
        errors = []
        if not self.name.strip():
            errors.append("Workflow name cannot be empty")
        if not self.steps:
            errors.append("Workflow must have at least one step")

        # Check for circular dependencies (simplified)
        step_ids = {f"{step.step_type.value}_{i}" for i, step in enumerate(self.steps)}
        for step in self.steps:
            step_id = f"{step.step_type.value}_{step.order}"
            for dep in step.dependencies:
                if dep not in step_ids:
                    errors.append(f"Step {step_id} depends on unknown step {dep}")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "icon": self.icon,
            "category": self.category,
            "steps": [{"step_type": step.step_type.value, "enabled": step.enabled,
                      "parameters": step.parameters, "dependencies": step.dependencies,
                      "order": step.order} for step in self.steps],
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CustomWorkflow':
        steps = []
        for step_data in data.get("steps", []):
            step = WorkflowStepConfig(
                step_type=WorkflowStep(step_data["step_type"]),
                enabled=step_data.get("enabled", True),
                parameters=step_data.get("parameters", {}),
                dependencies=step_data.get("dependencies", []),
                order=step_data.get("order", 0)
            )
            steps.append(step)

        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            icon=data["icon"],
            category=data["category"],
            steps=steps,
            created_at=data.get("created_at", __import__('time').time()),
            updated_at=data.get("updated_at", __import__('time').time())
        )

@dataclass
class WorkflowPreset:
    """Represents a complete workflow preset"""
    name: str
    description: str
    icon: str  # Emoji or icon name
    category: str  # "podcast", "music", "batch", "custom"

    # Audio processing settings
    format: str
    bitrates: List[int]
    content_type: str
    channels: int

    # Processing options
    loudnorm_enabled: bool = True
    compressor_enabled: bool = False
    compressor_preset: str = "speech"
    multiband_enabled: bool = False
    multiband_preset: str = "speech"
    ml_noise_reduction: bool = False
    silence_trim_enabled: bool = False
    noise_gate_enabled: bool = False
    parallel_processing: bool = False

    # Advanced settings
    custom_filters: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "icon": self.icon,
            "category": self.category,
            "format": self.format,
            "bitrates": self.bitrates,
            "content_type": self.content_type,
            "channels": self.channels,
            "loudnorm_enabled": self.loudnorm_enabled,
            "compressor_enabled": self.compressor_enabled,
            "compressor_preset": self.compressor_preset,
            "multiband_enabled": self.multiband_enabled,
            "multiband_preset": self.multiband_preset,
            "ml_noise_reduction": self.ml_noise_reduction,
            "silence_trim_enabled": self.silence_trim_enabled,
            "noise_gate_enabled": self.noise_gate_enabled,
            "parallel_processing": self.parallel_processing,
            "custom_filters": self.custom_filters
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowPreset':
        return cls(
            name=data["name"],
            description=data["description"],
            icon=data["icon"],
            category=data["category"],
            format=data["format"],
            bitrates=data["bitrates"],
            content_type=data["content_type"],
            channels=data["channels"],
            loudnorm_enabled=data.get("loudnorm_enabled", True),
            compressor_enabled=data.get("compressor_enabled", False),
            compressor_preset=data.get("compressor_preset", "speech"),
            multiband_enabled=data.get("multiband_enabled", False),
            multiband_preset=data.get("multiband_preset", "speech"),
            ml_noise_reduction=data.get("ml_noise_reduction", False),
            silence_trim_enabled=data.get("silence_trim_enabled", False),
            noise_gate_enabled=data.get("noise_gate_enabled", False),
            parallel_processing=data.get("parallel_processing", False),
            custom_filters=data.get("custom_filters")
        )

class WorkflowEngine:
    """Executes custom workflows with step-by-step processing"""

    def __init__(self):
        self.workflows: Dict[str, CustomWorkflow] = {}
        self.step_handlers: Dict[WorkflowStep, Callable] = {}
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default step handlers"""
        self.step_handlers[WorkflowStep.ANALYZE] = self._handle_analyze
        self.step_handlers[WorkflowStep.COMPRESS] = self._handle_compress
        self.step_handlers[WorkflowStep.NORMALIZE] = self._handle_normalize
        self.step_handlers[WorkflowStep.COMPRESS_AUDIO] = self._handle_compress_audio
        self.step_handlers[WorkflowStep.MULTIBAND_COMPRESS] = self._handle_multiband_compress
        self.step_handlers[WorkflowStep.NOISE_REDUCTION] = self._handle_noise_reduction
        self.step_handlers[WorkflowStep.SILENCE_TRIM] = self._handle_silence_trim
        self.step_handlers[WorkflowStep.NOISE_GATE] = self._handle_noise_gate
        self.step_handlers[WorkflowStep.METADATA_PRESERVE] = self._handle_metadata_preserve
        self.step_handlers[WorkflowStep.BATCH_PROCESS] = self._handle_batch_process
        self.step_handlers[WorkflowStep.UPLOAD_CLOUD] = self._handle_upload_cloud
        self.step_handlers[WorkflowStep.STORE_OFFLINE] = self._handle_store_offline

    def execute_workflow(self, workflow: CustomWorkflow, input_files: List[str],
                        output_dir: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute a custom workflow"""
        results = {
            "success": True,
            "processed_files": [],
            "errors": [],
            "statistics": {}
        }

        try:
            # Sort steps by order and dependencies
            sorted_steps = self._topological_sort(workflow.steps)

            for step_config in sorted_steps:
                if not step_config.enabled:
                    continue

                if progress_callback:
                    progress_callback(f"Executing step: {step_config.step_type.value}")

                handler = self.step_handlers.get(step_config.step_type)
                if handler:
                    step_result = handler(step_config, input_files, output_dir, results)
                    if not step_result["success"]:
                        results["success"] = False
                        results["errors"].extend(step_result["errors"])
                        break
                    else:
                        results["processed_files"].extend(step_result.get("processed_files", []))
                else:
                    results["errors"].append(f"No handler for step: {step_config.step_type.value}")
                    results["success"] = False
                    break

        except Exception as e:
            results["success"] = False
            results["errors"].append(f"Workflow execution failed: {str(e)}")

        return results

    def _topological_sort(self, steps: List[WorkflowStepConfig]) -> List[WorkflowStepConfig]:
        """Sort steps by dependencies using topological sort"""
        # Simplified topological sort - in practice, you'd want a more robust implementation
        sorted_steps = []
        visited = set()
        temp_visited = set()

        def visit(step: WorkflowStepConfig):
            step_id = f"{step.step_type.value}_{step.order}"
            if step_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving step {step_id}")
            if step_id in visited:
                return

            temp_visited.add(step_id)

            # Visit dependencies first
            for dep_id in step.dependencies:
                for dep_step in steps:
                    if f"{dep_step.step_type.value}_{dep_step.order}" == dep_id:
                        visit(dep_step)
                        break

            temp_visited.remove(step_id)
            visited.add(step_id)
            sorted_steps.append(step)

        for step in steps:
            if f"{step.step_type.value}_{step.order}" not in visited:
                visit(step)

        return sorted_steps

    # Step handlers
    def _handle_analyze(self, step: WorkflowStepConfig, input_files: List[str],
                       output_dir: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle analysis step"""
        results = {"success": True, "errors": [], "analysis_results": []}
        
        try:
            from audio_analysis import audio_analyzer
        except ImportError:
            results["success"] = False
            results["errors"].append("audio_analysis module not available")
            return results

        for file_path in input_files:
            try:
                analysis = audio_analyzer.analyze_file(file_path)
                if analysis:
                    results["analysis_results"].append(analysis)
                    context["analysis_results"] = context.get("analysis_results", []) + [analysis]
                else:
                    results["errors"].append(f"Failed to analyze {file_path}")
            except Exception as e:
                results["errors"].append(f"Error analyzing {file_path}: {str(e)}")

        return results

    def _handle_compress(self, step: WorkflowStepConfig, input_files: List[str],
                        output_dir: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle compression step"""
        from compress_audio import compress_audio
        results = {"success": True, "errors": [], "processed_files": []}

        # Get parameters from step config
        bitrate = step.parameters.get("bitrate", 128)
        format_type = step.parameters.get("format", "mp3")
        filter_chain = step.parameters.get("filter_chain")

        for input_file in input_files:
            filename = Path(input_file).stem
            output_file = os.path.join(output_dir, f"{filename}_compressed.{format_type}")

            success, _, _ = compress_audio(
                input_file=input_file,
                output_file=output_file,
                bitrate=bitrate,
                filter_chain=filter_chain,
                output_format=format_type,
                channels=step.parameters.get("channels", 1),
                preserve_metadata=step.parameters.get("preserve_metadata", True)
            )

            if success:
                results["processed_files"].append(output_file)
            else:
                results["errors"].append(f"Failed to compress {input_file}")
                results["success"] = False

        return results

    def _handle_normalize(self, step: WorkflowStepConfig, input_files: List[str],
                         output_dir: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle normalization step"""
        from compress_audio import build_audio_filters, compress_audio
        results = {"success": True, "errors": [], "processed_files": []}

        filter_chain = build_audio_filters(loudnorm_enabled=True)

        for input_file in input_files:
            filename = Path(input_file).stem
            output_file = os.path.join(output_dir, f"{filename}_normalized.wav")

            success, _, _ = compress_audio(
                input_file=input_file,
                output_file=output_file,
                bitrate=128,
                filter_chain=filter_chain,
                output_format="wav",
                preserve_metadata=True
            )

            if success:
                results["processed_files"].append(output_file)
            else:
                results["errors"].append(f"Failed to normalize {input_file}")
                results["success"] = False

        return results

    def _handle_compress_audio(self, step: WorkflowStepConfig, input_files: List[str],
                              output_dir: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle audio compression step"""
        from compress_audio import build_audio_filters, compress_audio
        results = {"success": True, "errors": [], "processed_files": []}

        filter_chain = build_audio_filters(
            compressor_enabled=True,
            compressor_preset=step.parameters.get("preset", "speech")
        )

        for input_file in input_files:
            filename = Path(input_file).stem
            output_file = os.path.join(output_dir, f"{filename}_compressed_audio.wav")

            success, _, _ = compress_audio(
                input_file=input_file,
                output_file=output_file,
                bitrate=128,
                filter_chain=filter_chain,
                output_format="wav",
                preserve_metadata=True
            )

            if success:
                results["processed_files"].append(output_file)
            else:
                results["errors"].append(f"Failed to apply compression to {input_file}")
                results["success"] = False

        return results

    def _handle_multiband_compress(self, step: WorkflowStepConfig, input_files: List[str],
                                  output_dir: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle multiband compression step"""
        from compress_audio import build_audio_filters, compress_audio
        results = {"success": True, "errors": [], "processed_files": []}

        filter_chain = build_audio_filters(
            multiband_enabled=True,
            multiband_preset=step.parameters.get("preset", "speech")
        )

        for input_file in input_files:
            filename = Path(input_file).stem
            output_file = os.path.join(output_dir, f"{filename}_multiband.wav")

            success, _, _ = compress_audio(
                input_file=input_file,
                output_file=output_file,
                bitrate=128,
                filter_chain=filter_chain,
                output_format="wav",
                preserve_metadata=True
            )

            if success:
                results["processed_files"].append(output_file)
            else:
                results["errors"].append(f"Failed to apply multiband compression to {input_file}")
                results["success"] = False

        return results

    def _handle_noise_reduction(self, step: WorkflowStepConfig, input_files: List[str],
                               output_dir: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle noise reduction step"""
        from compress_audio import build_audio_filters, compress_audio
        results = {"success": True, "errors": [], "processed_files": []}

        filter_chain = build_audio_filters(ml_noise_reduction=True)

        for input_file in input_files:
            filename = Path(input_file).stem
            output_file = os.path.join(output_dir, f"{filename}_denoised.wav")

            success, _, _ = compress_audio(
                input_file=input_file,
                output_file=output_file,
                bitrate=128,
                filter_chain=filter_chain,
                output_format="wav",
                preserve_metadata=True
            )

            if success:
                results["processed_files"].append(output_file)
            else:
                results["errors"].append(f"Failed to apply noise reduction to {input_file}")
                results["success"] = False

        return results

    def _handle_silence_trim(self, step: WorkflowStepConfig, input_files: List[str],
                            output_dir: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle silence trimming step"""
        from compress_audio import build_audio_filters, compress_audio
        results = {"success": True, "errors": [], "processed_files": []}

        filter_chain = build_audio_filters(silence_trim_enabled=True)

        for input_file in input_files:
            filename = Path(input_file).stem
            output_file = os.path.join(output_dir, f"{filename}_trimmed.wav")

            success, _, _ = compress_audio(
                input_file=input_file,
                output_file=output_file,
                bitrate=128,
                filter_chain=filter_chain,
                output_format="wav",
                preserve_metadata=True
            )

            if success:
                results["processed_files"].append(output_file)
            else:
                results["errors"].append(f"Failed to trim silence from {input_file}")
                results["success"] = False

        return results

    def _handle_noise_gate(self, step: WorkflowStepConfig, input_files: List[str],
                          output_dir: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle noise gating step"""
        from compress_audio import build_audio_filters, compress_audio
        results = {"success": True, "errors": [], "processed_files": []}

        filter_chain = build_audio_filters(noise_gate_enabled=True)

        for input_file in input_files:
            filename = Path(input_file).stem
            output_file = os.path.join(output_dir, f"{filename}_gated.wav")

            success, _, _ = compress_audio(
                input_file=input_file,
                output_file=output_file,
                bitrate=128,
                filter_chain=filter_chain,
                output_format="wav",
                preserve_metadata=True
            )

            if success:
                results["processed_files"].append(output_file)
            else:
                results["errors"].append(f"Failed to apply noise gate to {input_file}")
                results["success"] = False

        return results

    def _handle_metadata_preserve(self, step: WorkflowStepConfig, input_files: List[str],
                                 output_dir: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle metadata preservation step"""
        # This is handled automatically in compression steps
        return {"success": True, "errors": [], "processed_files": []}

    def _handle_batch_process(self, step: WorkflowStepConfig, input_files: List[str],
                             output_dir: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle batch processing step"""
        from job_queue import job_queue
        results = {"success": True, "errors": [], "processed_files": []}

        # Queue jobs for batch processing
        for input_file in input_files:
            # This would need more sophisticated job creation based on step parameters
            # For now, just mark as processed
            results["processed_files"].append(input_file)

        return results

    def _handle_upload_cloud(self, step: WorkflowStepConfig, input_files: List[str],
                            output_dir: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cloud upload step"""
        from cloud_integration import cloud_manager
        results = {"success": True, "errors": [], "processed_files": []}

        for file_path in input_files:
            if cloud_manager.upload_file(file_path, step.parameters.get("bucket", "audio-processed")):
                results["processed_files"].append(file_path)
            else:
                results["errors"].append(f"Failed to upload {file_path} to cloud")
                results["success"] = False

        return results

    def _handle_store_offline(self, step: WorkflowStepConfig, input_files: List[str],
                             output_dir: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle offline storage step"""
        from offline_storage import storage_manager
        results = {"success": True, "errors": [], "processed_files": []}

        for file_path in input_files:
            metadata = step.parameters.get("metadata", {})
            if storage_manager.store_file(file_path, f"processed_audio/{Path(file_path).name}", metadata):
                results["processed_files"].append(file_path)
            else:
                results["errors"].append(f"Failed to store {file_path} offline")
                results["success"] = False

        return results

class PresetManager:
    """Manages workflow presets with persistence and validation"""

    def __init__(self, config_file: str = "workflow_presets.json", workflows_file: str = "custom_workflows.json"):
        self.config_file = Path(config_file)
        self.workflows_file = Path(workflows_file)
        self.presets: Dict[str, WorkflowPreset] = {}
        self.custom_workflows: Dict[str, CustomWorkflow] = {}
        self.workflow_engine = WorkflowEngine()
        self._load_presets()
        self._load_workflows()
        self._ensure_default_presets()

    def _load_presets(self):
        """Load presets from file"""
        if not self.config_file.exists():
            return

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for preset_id, preset_data in data.items():
                    self.presets[preset_id] = WorkflowPreset.from_dict(preset_data)
        except Exception as e:
            print(f"Error loading presets: {e}")

    def _load_workflows(self):
        """Load custom workflows from file"""
        if not self.workflows_file.exists():
            return

        try:
            with open(self.workflows_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for workflow_id, workflow_data in data.items():
                    self.custom_workflows[workflow_id] = CustomWorkflow.from_dict(workflow_data)
        except Exception as e:
            print(f"Error loading workflows: {e}")

    def _save_presets(self):
        """Save presets to file"""
        try:
            data = {pid: preset.to_dict() for pid, preset in self.presets.items()}
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving presets: {e}")

    def _save_workflows(self):
        """Save custom workflows to file"""
        try:
            data = {wid: workflow.to_dict() for wid, workflow in self.custom_workflows.items()}
            with open(self.workflows_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving workflows: {e}")

    def _ensure_default_presets(self):
        """Ensure default presets are available"""
        default_presets = self._get_default_presets()

        for preset_id, preset in default_presets.items():
            if preset_id not in self.presets:
                self.presets[preset_id] = preset

        self._save_presets()

    def _get_default_presets(self) -> Dict[str, WorkflowPreset]:
        """Get the default workflow presets"""
        return {
            "podcast_optimization": WorkflowPreset(
                name="Podcast Optimization",
                description="Optimize audio for podcast distribution with speech enhancement and loudness normalization",
                icon="🎙️",
                category="podcast",
                format="mp3",
                bitrates=[64, 96, 128],
                content_type="speech",
                channels=1,
                loudnorm_enabled=True,
                compressor_enabled=True,
                compressor_preset="speech",
                multiband_enabled=True,
                multiband_preset="speech",
                silence_trim_enabled=True,
                noise_gate_enabled=True,
                parallel_processing=True
            ),

            "music_mastering": WorkflowPreset(
                name="Music Mastering",
                description="Professional music mastering with dynamic range compression and multiband processing",
                icon="🎵",
                category="music",
                format="flac",
                bitrates=[],  # Lossless
                content_type="music",
                channels=2,
                loudnorm_enabled=True,
                compressor_enabled=True,
                compressor_preset="music",
                multiband_enabled=True,
                multiband_preset="music",
                parallel_processing=True
            ),

            "batch_conversion": WorkflowPreset(
                name="Batch Conversion",
                description="Fast batch conversion to multiple formats and bitrates for web distribution",
                icon="🔄",
                category="batch",
                format="mp3",
                bitrates=[96, 128, 192, 256],
                content_type="music",
                channels=2,
                loudnorm_enabled=True,
                parallel_processing=True
            ),

            "voice_enhancement": WorkflowPreset(
                name="Voice Enhancement",
                description="Enhance voice recordings with noise reduction and clarity improvements",
                icon="🗣️",
                category="podcast",
                format="aac",
                bitrates=[64, 96],
                content_type="speech",
                channels=1,
                loudnorm_enabled=True,
                compressor_enabled=True,
                compressor_preset="speech",
                ml_noise_reduction=True,
                silence_trim_enabled=True,
                noise_gate_enabled=True
            ),

            "streaming_optimization": WorkflowPreset(
                name="Streaming Optimization",
                description="Optimize for streaming platforms with efficient compression and metadata preservation",
                icon="📺",
                category="batch",
                format="aac",
                bitrates=[96, 128, 192],
                content_type="music",
                channels=2,
                loudnorm_enabled=True,
                compressor_enabled=True,
                compressor_preset="broadcast",
                parallel_processing=True
            ),

            "audiobook_production": WorkflowPreset(
                name="Audiobook Production",
                description="Professional audiobook production with speech optimization and chapter support",
                icon="📖",
                category="podcast",
                format="mp3",
                bitrates=[64, 128],
                content_type="speech",
                channels=1,
                loudnorm_enabled=True,
                compressor_enabled=True,
                compressor_preset="speech",
                silence_trim_enabled=True
            ),

            "live_recording_cleanup": WorkflowPreset(
                name="Live Recording Cleanup",
                description="Clean up live recordings with noise reduction and dynamic processing",
                icon="🎤",
                category="music",
                format="flac",
                bitrates=[],
                content_type="music",
                channels=2,
                loudnorm_enabled=True,
                compressor_enabled=True,
                compressor_preset="broadcast",
                multiband_enabled=True,
                multiband_preset="music",
                ml_noise_reduction=True,
                noise_gate_enabled=True
            ),

            "mobile_optimization": WorkflowPreset(
                name="Mobile Optimization",
                description="Optimize for mobile devices with efficient compression and battery-friendly processing",
                icon="📱",
                category="batch",
                format="aac",
                bitrates=[64, 96, 128],
                content_type="music",
                channels=2,
                loudnorm_enabled=True,
                parallel_processing=True
            )
        }

    def get_preset(self, preset_id: str) -> Optional[WorkflowPreset]:
        """Get a preset by ID"""
        return self.presets.get(preset_id)

    def get_presets_by_category(self, category: str) -> List[WorkflowPreset]:
        """Get all presets in a category"""
        return [p for p in self.presets.values() if p.category == category]

    def get_all_presets(self) -> List[WorkflowPreset]:
        """Get all presets"""
        return list(self.presets.values())

    def add_custom_preset(self, preset: WorkflowPreset) -> str:
        """Add a custom preset"""
        preset_id = f"custom_{len([p for p in self.presets.keys() if p.startswith('custom_')]) + 1}"
        preset.category = "custom"
        self.presets[preset_id] = preset
        self._save_presets()
        return preset_id

    def update_preset(self, preset_id: str, preset: WorkflowPreset):
        """Update an existing preset"""
        if preset_id in self.presets:
            self.presets[preset_id] = preset
            self._save_presets()

    def delete_preset(self, preset_id: str) -> bool:
        """Delete a preset (only custom presets can be deleted)"""
        if preset_id in self.presets and preset_id.startswith("custom_"):
            del self.presets[preset_id]
            self._save_presets()
            return True
        return False

    # Custom workflow management methods
    def create_custom_workflow(self, name: str, description: str, icon: str = "🔧",
                              category: str = "custom") -> str:
        """Create a new custom workflow"""
        workflow_id = str(uuid.uuid4())
        workflow = CustomWorkflow(
            id=workflow_id,
            name=name,
            description=description,
            icon=icon,
            category=category
        )
        self.custom_workflows[workflow_id] = workflow
        self._save_workflows()
        return workflow_id

    def get_custom_workflow(self, workflow_id: str) -> Optional[CustomWorkflow]:
        """Get a custom workflow by ID"""
        return self.custom_workflows.get(workflow_id)

    def get_all_custom_workflows(self) -> List[CustomWorkflow]:
        """Get all custom workflows"""
        return list(self.custom_workflows.values())

    def update_custom_workflow(self, workflow: CustomWorkflow):
        """Update an existing custom workflow"""
        if workflow.id in self.custom_workflows:
            self.custom_workflows[workflow.id] = workflow
            self._save_workflows()

    def delete_custom_workflow(self, workflow_id: str) -> bool:
        """Delete a custom workflow"""
        if workflow_id in self.custom_workflows:
            del self.custom_workflows[workflow_id]
            self._save_workflows()
            return True
        return False

    def execute_custom_workflow(self, workflow_id: str, input_files: List[str],
                               output_dir: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute a custom workflow"""
        workflow = self.get_custom_workflow(workflow_id)
        if not workflow:
            return {"success": False, "errors": [f"Workflow {workflow_id} not found"]}

        return self.workflow_engine.execute_workflow(workflow, input_files, output_dir, progress_callback)

    # Intelligent parameter suggestions based on content analysis
    def suggest_parameters(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate parameter suggestions based on audio analysis"""
        suggestions = {
            "format": "mp3",
            "bitrates": [128],
            "content_type": "speech",
            "enable_compression": False,
            "enable_multiband": False,
            "enable_loudnorm": True,
            "enable_noise_reduction": False,
            "confidence": 0.0,
            "reasoning": []
        }

        if not analysis_results or "content_analysis" not in analysis_results:
            suggestions["reasoning"].append("No analysis data available - using conservative defaults")
            return suggestions

        content = analysis_results["content_analysis"]

        # Content type detection
        speech_prob = content.get("speech_probability", 0)
        music_prob = content.get("music_probability", 0)

        if speech_prob > 0.7:
            suggestions["content_type"] = "speech"
            suggestions["bitrates"] = [64, 96, 128]
            suggestions["reasoning"].append("High speech probability detected")
        elif music_prob > 0.7:
            suggestions["content_type"] = "music"
            suggestions["bitrates"] = [128, 192, 256]
            suggestions["reasoning"].append("High music probability detected")
        else:
            suggestions["content_type"] = "speech"  # Default to speech for safety
            suggestions["bitrates"] = [96, 128, 192]
            suggestions["reasoning"].append("Mixed content detected - using moderate settings")

        # Dynamic range analysis
        dynamic_range = content.get("dynamic_range", "")
        if dynamic_range:
            try:
                dr_value = float(dynamic_range.split()[0])
                if dr_value > 20:
                    suggestions["enable_compression"] = True
                    suggestions["reasoning"].append(f"High dynamic range ({dr_value}dB) - compression recommended")
                elif dr_value < 10:
                    suggestions["enable_multiband"] = True
                    suggestions["reasoning"].append(f"Low dynamic range ({dr_value}dB) - multiband processing may help")
            except (ValueError, IndexError):
                pass

        # Quality assessment
        basic_info = analysis_results.get("basic_info", {})
        bitrate_kbps = basic_info.get("bitrate_kbps", 128)

        if bitrate_kbps > 256:
            suggestions["reasoning"].append("High bitrate source - can compress more aggressively")
        elif bitrate_kbps < 96:
            suggestions["enable_loudnorm"] = True
            suggestions["reasoning"].append("Low bitrate source - loudness normalization recommended")

        # Calculate confidence
        suggestions["confidence"] = min(speech_prob + music_prob, 1.0)

        return suggestions

    def apply_preset_to_gui(self, preset: WorkflowPreset) -> Dict[str, Any]:
        """Convert preset to GUI settings dictionary"""
        return {
            "format_var": preset.format,
            "bitrates": ','.join(map(str, preset.bitrates)),
            "content_type": preset.content_type,
            "channels_var": preset.channels,
            "normalize_var": preset.loudnorm_enabled,
            "compressor_var": preset.compressor_enabled,
            "comp_preset_var": preset.compressor_preset,
            "multiband_var": preset.multiband_enabled,
            "mb_preset_var": preset.multiband_preset,
            "ml_noise_var": preset.ml_noise_reduction,
            "silence_trim_var": preset.silence_trim_enabled,
            "noise_gate_var": preset.noise_gate_enabled,
            "parallel_var": preset.parallel_processing
        }

# Global preset manager instance
preset_manager = PresetManager()