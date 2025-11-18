"""
Advanced Audio Compression and Filter Processing Engine

This module provides enterprise-grade audio processing capabilities including:
- Multi-band compression and dynamic range processing
- ML-based noise reduction with FFmpeg arnndn models
- Real-time parameter adjustment and preview generation
- Advanced audio filters (de-essing, gating, normalization)
- Batch processing with error recovery
- Multi-stream output configuration
"""

import os
import subprocess
import tempfile
import json
import logging
import time
import threading
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid

from interfaces import IEventPublisher, IServiceProvider
from di_container import get_service
from security import security_manager, Permission
from audio_analysis_enhanced import AudioContentType, AudioAnalysisResult


class FilterType(Enum):
    """Types of audio filters"""
    COMPRESSOR = "compressor"
    MULTIBAND_COMPRESSOR = "multiband_compressor"
    NOISE_GATE = "noise_gate"
    DEESSER = "deesser"
    NORMALIZE = "normalize"
    LOUDNESS_NORMALIZE = "loudness_normalize"
    NOISE_REDUCTION = "noise_reduction"
    HIGH_PASS = "high_pass"
    LOW_PASS = "low_pass"
    EQ = "equalizer"
    REVERB = "reverb"
    DELAY = "delay"


class ProcessingQuality(Enum):
    """Processing quality levels"""
    REAL_TIME = "realtime"
    HIGH_QUALITY = "high_quality"
    MAXIMUM_QUALITY = "maximum_quality"


@dataclass
class FilterParameter:
    """Audio filter parameter definition"""
    name: str
    value: Union[float, int, str, bool]
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    unit: str = ""
    description: str = ""
    depends_on: Optional[str] = None  # Other parameter name


@dataclass
class AudioFilter:
    """Audio filter configuration"""
    filter_type: FilterType
    parameters: Dict[str, FilterParameter]
    enabled: bool = True
    order: int = 0  # Processing order
    bypass: bool = False
    
    def to_ffmpeg_filter(self) -> str:
        """Convert filter to FFmpeg filter string"""
        if not self.enabled or self.bypass:
            return ""
        
        filter_map = {
            FilterType.COMPRESSOR: self._compressor_to_ffmpeg,
            FilterType.MULTIBAND_COMPRESSOR: self._multiband_to_ffmpeg,
            FilterType.NOISE_GATE: self._gate_to_ffmpeg,
            FilterType.DEESSER: self._deesser_to_ffmpeg,
            FilterType.NORMALIZE: self._normalize_to_ffmpeg,
            FilterType.LOUDNESS_NORMALIZE: self._loudnorm_to_ffmpeg,
            FilterType.NOISE_REDUCTION: self._nr_to_ffmpeg,
            FilterType.HIGH_PASS: self._highpass_to_ffmpeg,
            FilterType.LOW_PASS: self._lowpass_to_ffmpeg,
            FilterType.EQ: self._eq_to_ffmpeg,
        }
        
        converter = filter_map.get(self.filter_type)
        if converter:
            return converter()
        else:
            return ""
    
    def _compressor_to_ffmpeg(self) -> str:
        """Convert compressor to FFmpeg filter"""
        params = self.parameters
        return (f"acompressor="
                f"threshold={params.get('threshold', -20).value}dB:"
                f"ratio={params.get('ratio', 3).value}:"
                f"attack={params.get('attack', 0.01).value}:"
                f"release={params.get('release', 0.1).value}:"
                f"makeup={params.get('makeup', 0).value}")
    
    def _multiband_to_ffmpeg(self) -> str:
        """Convert multiband compressor to FFmpeg filter"""
        params = self.parameters
        return (f"mbcompressor="
                f"f={params.get('low_freq', 250).value}:"
                f"f1={params.get('mid_freq', 2500).value}:"
                f"mode=4:"
                f"threshold={params.get('low_threshold', -12).value}:"
                f"ratio={params.get('low_ratio', 2).value}:"
                f"attack={params.get('low_attack', 0.01).value}:"
                f"release={params.get('low_release', 0.1).value}")
    
    def _gate_to_ffmpeg(self) -> str:
        """Convert noise gate to FFmpeg filter"""
        params = self.parameters
        return (f"agate="
                f"threshold={params.get('threshold', -40).value}dB:"
                f"ratio={params.get('ratio', 10).value}:"
                f"attack={params.get('attack', 0.1).value}:"
                f"release={params.get('release', 0.5).value}")
    
    def _deesser_to_ffmpeg(self) -> str:
        """Convert de-esser to FFmpeg filter"""
        params = self.parameters
        return (f"deesser="
                f"i={params.get('intensity', 0.5).value}:"
                f"m={params.get('mode', 'wideband').value}:"
                f"f={params.get('frequency', 6000).value}")
    
    def _normalize_to_ffmpeg(self) -> str:
        """Convert normalize to FFmpeg filter"""
        params = self.parameters
        return f"loudnorm=I={params.get('target_i', -16).value}:TP=-1.5:LRA=11"
    
    def _loudnorm_to_ffmpeg(self) -> str:
        """Convert loudness normalize to FFmpeg filter"""
        params = self.parameters
        return (f"loudnorm=I={params.get('target_i', -16).value}:"
                f"TP={params.get('target_tp', -1.5).value}:"
                f"LRA={params.get('target_lra', 11).value}")
    
    def _nr_to_ffmpeg(self) -> str:
        """Convert noise reduction to FFmpeg filter"""
        params = self.parameters
        model = params.get('model', '/usr/share/ffmpeg/arnndn-models/bd.cnr.mdl').value
        return f"arnndn=model={model}:micsize=257:winlen=512:winstep=256"
    
    def _highpass_to_ffmpeg(self) -> str:
        """Convert high-pass filter to FFmpeg filter"""
        params = self.parameters
        return f"highpass=f={params.get('frequency', 80).value}"
    
    def _lowpass_to_ffmpeg(self) -> str:
        """Convert low-pass filter to FFmpeg filter"""
        params = self.parameters
        return f"lowpass=f={params.get('frequency', 8000).value}"
    
    def _eq_to_ffmpeg(self) -> str:
        """Convert EQ to FFmpeg filter"""
        params = self.parameters
        # Simple 3-band EQ
        return (f"equalizer=f={params.get('low_freq', 100).value}:"
                f"width_type=h:width={params.get('low_width', 100).value}:"
                f"g={params.get('low_gain', 0).value}:"
                f"f={params.get('mid_freq', 1000).value}:"
                f"width_type=h:width={params.get('mid_width', 100).value}:"
                f"g={params.get('mid_gain', 0).value}:"
                f"f={params.get('high_freq', 10000).value}:"
                f"width_type=h:width={params.get('high_width', 1000).value}:"
                f"g={params.get('high_gain', 0).value}")


@dataclass
class ProcessingJob:
    """Audio processing job definition"""
    job_id: str
    input_file: str
    output_files: List[str]  # Multiple output files for multi-stream
    filters: List[AudioFilter]
    format_config: Dict[str, Any]
    quality: ProcessingQuality = ProcessingQuality.HIGH_QUALITY
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 2
    max_retries: int = 3
    timeout: int = 300  # 5 minutes
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if not self.job_id:
            self.job_id = str(uuid.uuid4())


class FilterPresetManager:
    """Manages audio filter presets for different content types"""

    def __init__(self):
        self.presets: Dict[str, List[AudioFilter]] = {
            "speech_clean": self._create_speech_clean_preset(),
            "speech_podcast": self._create_speech_podcast_preset(),
            "speech_broadcast": self._create_speech_broadcast_preset(),
            "music_general": self._create_music_general_preset(),
            "music_electronic": self._create_music_electronic_preset(),
            "music_vocal": self._create_music_vocal_preset(),
            "noise_reduction": self._create_noise_reduction_preset(),
            "broadcast": self._create_broadcast_preset(),
            "mastering": self._create_mastering_preset()
        }

    def _create_speech_clean_preset(self) -> List[AudioFilter]:
        """Clean speech preset - removes noise and normalizes"""
        filters = []
        
        # Noise gate
        gate = AudioFilter(
            FilterType.NOISE_GATE,
            {
                "threshold": FilterParameter("threshold", -45, -60, -20, 1, "dB", "Gate threshold"),
                "ratio": FilterParameter("ratio", 15, 2, 20, 1, ":1", "Gate ratio"),
                "attack": FilterParameter("attack", 0.05, 0.01, 0.5, 0.01, "s", "Attack time"),
                "release": FilterParameter("release", 0.2, 0.1, 2.0, 0.1, "s", "Release time")
            }
        )
        filters.append(gate)
        
        # Compressor
        compressor = AudioFilter(
            FilterType.COMPRESSOR,
            {
                "threshold": FilterParameter("threshold", -18, -30, -6, 1, "dB", "Compression threshold"),
                "ratio": FilterParameter("ratio", 4, 1.5, 8, 0.5, ":1", "Compression ratio"),
                "attack": FilterParameter("attack", 0.01, 0.005, 0.1, 0.005, "s", "Attack time"),
                "release": FilterParameter("release", 0.1, 0.05, 0.5, 0.05, "s", "Release time"),
                "makeup": FilterParameter("makeup", 4, 0, 12, 0.5, "dB", "Makeup gain")
            }
        )
        filters.append(compressor)
        
        # High-pass filter to remove rumble
        highpass = AudioFilter(
            FilterType.HIGH_PASS,
            {
                "frequency": FilterParameter("frequency", 80, 40, 200, 10, "Hz", "Cutoff frequency")
            }
        )
        filters.append(highpass)
        
        return filters

    def _create_speech_podcast_preset(self) -> List[AudioFilter]:
        """Podcast-specific speech preset"""
        filters = []
        
        # Gentle noise gate
        gate = AudioFilter(
            FilterType.NOISE_GATE,
            {
                "threshold": FilterParameter("threshold", -35, -50, -20, 1, "dB", "Gate threshold"),
                "ratio": FilterParameter("ratio", 8, 3, 15, 1, ":1", "Gate ratio"),
                "attack": FilterParameter("attack", 0.1, 0.01, 0.3, 0.01, "s", "Attack time"),
                "release": FilterParameter("release", 0.3, 0.1, 1.0, 0.1, "s", "Release time")
            }
        )
        filters.append(gate)
        
        # Broadcast compressor
        compressor = AudioFilter(
            FilterType.COMPRESSOR,
            {
                "threshold": FilterParameter("threshold", -20, -30, -10, 1, "dB", "Compression threshold"),
                "ratio": FilterParameter("ratio", 3, 2, 6, 0.5, ":1", "Compression ratio"),
                "attack": FilterParameter("attack", 0.02, 0.005, 0.05, 0.005, "s", "Attack time"),
                "release": FilterParameter("release", 0.1, 0.05, 0.3, 0.05, "s", "Release time"),
                "makeup": FilterParameter("makeup", 6, 2, 12, 0.5, "dB", "Makeup gain")
            }
        )
        filters.append(compressor)
        
        # Loudness normalization for consistent levels
        loudnorm = AudioFilter(
            FilterType.LOUDNESS_NORMALIZE,
            {
                "target_i": FilterParameter("target_i", -18, -24, -12, 1, "LUFS", "Integrated loudness target"),
                "target_tp": FilterParameter("target_tp", -1.5, -3, 0, 0.5, "LUFS", "True peak target"),
                "target_lra": FilterParameter("target_lra", 11, 7, 17, 1, "LUFS", "LRA target")
            }
        )
        filters.append(loudnorm)
        
        return filters

    def _create_speech_broadcast_preset(self) -> List[AudioFilter]:
        """Broadcast-quality speech preset"""
        filters = []
        
        # Multiband compressor for professional sound
        multiband = AudioFilter(
            FilterType.MULTIBAND_COMPRESSOR,
            {
                "low_freq": FilterParameter("low_freq", 200, 80, 500, 20, "Hz", "Low/mid crossover"),
                "mid_freq": FilterParameter("mid_freq", 3000, 2000, 5000, 100, "Hz", "Mid/high crossover"),
                "low_threshold": FilterParameter("low_threshold", -15, -30, -6, 1, "dB", "Low band threshold"),
                "low_ratio": FilterParameter("low_ratio", 2.5, 1.5, 4, 0.5, ":1", "Low band ratio"),
                "mid_threshold": FilterParameter("mid_threshold", -18, -30, -6, 1, "dB", "Mid band threshold"),
                "mid_ratio": FilterParameter("mid_ratio", 3, 2, 5, 0.5, ":1", "Mid band ratio"),
                "high_threshold": FilterParameter("high_threshold", -20, -30, -6, 1, "dB", "High band threshold"),
                "high_ratio": FilterParameter("high_ratio", 2, 1.5, 3, 0.5, ":1", "High band ratio")
            }
        )
        filters.append(multiband)
        
        # De-esser for sibilance control
        deesser = AudioFilter(
            FilterType.DEESSER,
            {
                "intensity": FilterParameter("intensity", 0.7, 0.3, 1.0, 0.1, "", "De-essing intensity"),
                "mode": FilterParameter("mode", "wideband", None, None, None, "", "Detection mode"),
                "frequency": FilterParameter("frequency", 6500, 4000, 8000, 100, "Hz", "Center frequency")
            }
        )
        filters.append(deesser)
        
        # Loudness normalization
        loudnorm = AudioFilter(
            FilterType.LOUDNESS_NORMALIZE,
            {
                "target_i": FilterParameter("target_i", -16, -23, -12, 1, "LUFS", "Integrated loudness target"),
                "target_tp": FilterParameter("target_tp", -1.0, -2, 0, 0.5, "LUFS", "True peak target"),
                "target_lra": FilterParameter("target_lra", 11, 7, 17, 1, "LUFS", "LRA target")
            }
        )
        filters.append(loudnorm)
        
        return filters

    def _create_music_general_preset(self) -> List[AudioFilter]:
        """General music preset preserving dynamics"""
        filters = []
        
        # Gentle compression
        compressor = AudioFilter(
            FilterType.COMPRESSOR,
            {
                "threshold": FilterParameter("threshold", -12, -24, -6, 1, "dB", "Compression threshold"),
                "ratio": FilterParameter("ratio", 2, 1.5, 3, 0.5, ":1", "Compression ratio"),
                "attack": FilterParameter("attack", 0.005, 0.001, 0.05, 0.001, "s", "Attack time"),
                "release": FilterParameter("release", 0.05, 0.01, 0.2, 0.01, "s", "Release time"),
                "makeup": FilterParameter("makeup", 1, 0, 3, 0.5, "dB", "Makeup gain")
            }
        )
        filters.append(compressor)
        
        return filters

    def _create_music_electronic_preset(self) -> List[AudioFilter]:
        """Electronic music preset with punch"""
        filters = []
        
        # Punchy compressor
        compressor = AudioFilter(
            FilterType.COMPRESSOR,
            {
                "threshold": FilterParameter("threshold", -18, -30, -12, 1, "dB", "Compression threshold"),
                "ratio": FilterParameter("ratio", 4, 3, 8, 0.5, ":1", "Compression ratio"),
                "attack": FilterParameter("attack", 0.003, 0.001, 0.02, 0.001, "s", "Attack time"),
                "release": FilterParameter("release", 0.1, 0.05, 0.3, 0.01, "s", "Release time"),
                "makeup": FilterParameter("makeup", 2, 0, 6, 0.5, "dB", "Makeup gain")
            }
        )
        filters.append(compressor)
        
        return filters

    def _create_music_vocal_preset(self) -> List[AudioFilter]:
        """Vocal-focused music preset"""
        filters = []
        
        # Vocal compressor
        compressor = AudioFilter(
            FilterType.COMPRESSOR,
            {
                "threshold": FilterParameter("threshold", -15, -25, -8, 1, "dB", "Compression threshold"),
                "ratio": FilterParameter("ratio", 3, 2, 6, 0.5, ":1", "Compression ratio"),
                "attack": FilterParameter("attack", 0.01, 0.005, 0.03, 0.005, "s", "Attack time"),
                "release": FilterParameter("release", 0.1, 0.05, 0.3, 0.05, "s", "Release time"),
                "makeup": FilterParameter("makeup", 3, 0, 8, 0.5, "dB", "Makeup gain")
            }
        )
        filters.append(compressor)
        
        # De-esser for vocal sibilance
        deesser = AudioFilter(
            FilterType.DEESSER,
            {
                "intensity": FilterParameter("intensity", 0.8, 0.5, 1.0, 0.1, "", "De-essing intensity"),
                "mode": FilterParameter("mode", "wideband", None, None, None, "", "Detection mode"),
                "frequency": FilterParameter("frequency", 7000, 5000, 9000, 100, "Hz", "Center frequency")
            }
        )
        filters.append(deesser)
        
        return filters

    def _create_noise_reduction_preset(self) -> List[AudioFilter]:
        """Noise reduction preset using ML models"""
        filters = []
        
        # ML-based noise reduction
        noise_reduction = AudioFilter(
            FilterType.NOISE_REDUCTION,
            {
                "model": FilterParameter("model", "/usr/share/ffmpeg/arnndn-models/bd.cnr.mdl", None, None, None, "", "ML model path")
            }
        )
        filters.append(noise_reduction)
        
        # Light compression after noise reduction
        compressor = AudioFilter(
            FilterType.COMPRESSOR,
            {
                "threshold": FilterParameter("threshold", -20, -30, -12, 1, "dB", "Compression threshold"),
                "ratio": FilterParameter("ratio", 2.5, 2, 4, 0.5, ":1", "Compression ratio"),
                "attack": FilterParameter("attack", 0.02, 0.01, 0.05, 0.005, "s", "Attack time"),
                "release": FilterParameter("release", 0.1, 0.05, 0.3, 0.05, "s", "Release time"),
                "makeup": FilterParameter("makeup", 2, 0, 6, 0.5, "dB", "Makeup gain")
            }
        )
        filters.append(compressor)
        
        return filters

    def _create_broadcast_preset(self) -> List[AudioFilter]:
        """Full broadcast processing chain"""
        filters = []
        
        # High-pass filter
        highpass = AudioFilter(
            FilterType.HIGH_PASS,
            {
                "frequency": FilterParameter("frequency", 60, 30, 120, 5, "Hz", "Cutoff frequency")
            }
        )
        filters.append(highpass)
        
        # Multiband compressor
        multiband = AudioFilter(
            FilterType.MULTIBAND_COMPRESSOR,
            {
                "low_freq": FilterParameter("low_freq", 250, 200, 400, 25, "Hz", "Low/mid crossover"),
                "mid_freq": FilterParameter("mid_freq", 4000, 3000, 6000, 200, "Hz", "Mid/high crossover"),
                "low_threshold": FilterParameter("low_threshold", -18, -30, -12, 1, "dB", "Low band threshold"),
                "low_ratio": FilterParameter("low_ratio", 3, 2, 5, 0.5, ":1", "Low band ratio"),
                "mid_threshold": FilterParameter("mid_threshold", -15, -30, -10, 1, "dB", "Mid band threshold"),
                "mid_ratio": FilterParameter("mid_ratio", 4, 3, 6, 0.5, ":1", "Mid band ratio"),
                "high_threshold": FilterParameter("high_threshold", -20, -30, -12, 1, "dB", "High band threshold"),
                "high_ratio": FilterParameter("high_ratio", 2.5, 2, 4, 0.5, ":1", "High band ratio")
            }
        )
        filters.append(multiband)
        
        # De-esser
        deesser = AudioFilter(
            FilterType.DEESSER,
            {
                "intensity": FilterParameter("intensity", 0.6, 0.3, 1.0, 0.1, "", "De-essing intensity"),
                "mode": FilterParameter("mode", "wideband", None, None, None, "", "Detection mode"),
                "frequency": FilterParameter("frequency", 6500, 5000, 8000, 100, "Hz", "Center frequency")
            }
        )
        filters.append(deesser)
        
        # Loudness normalization
        loudnorm = AudioFilter(
            FilterType.LOUDNESS_NORMALIZE,
            {
                "target_i": FilterParameter("target_i", -16, -23, -12, 1, "LUFS", "Integrated loudness target"),
                "target_tp": FilterParameter("target_tp", -1.0, -2, 0, 0.5, "LUFS", "True peak target"),
                "target_lra": FilterParameter("target_lra", 11, 7, 17, 1, "LUFS", "LRA target")
            }
        )
        filters.append(loudnorm)
        
        return filters

    def _create_mastering_preset(self) -> List[AudioFilter]:
        """Mastering-grade processing preset"""
        filters = []
        
        # Multi-band compressor for mastering
        multiband = AudioFilter(
            FilterType.MULTIBAND_COMPRESSOR,
            {
                "low_freq": FilterParameter("low_freq", 200, 150, 300, 10, "Hz", "Low/mid crossover"),
                "mid_freq": FilterParameter("mid_freq", 5000, 4000, 7000, 100, "Hz", "Mid/high crossover"),
                "low_threshold": FilterParameter("low_threshold", -12, -20, -6, 1, "dB", "Low band threshold"),
                "low_ratio": FilterParameter("low_ratio", 2, 1.5, 3, 0.5, ":1", "Low band ratio"),
                "mid_threshold": FilterParameter("mid_threshold", -15, -25, -8, 1, "dB", "Mid band threshold"),
                "mid_ratio": FilterParameter("mid_ratio", 2.5, 2, 4, 0.5, ":1", "Mid band ratio"),
                "high_threshold": FilterParameter("high_threshold", -18, -30, -10, 1, "dB", "High band threshold"),
                "high_ratio": FilterParameter("high_ratio", 1.5, 1.2, 2.5, 0.2, ":1", "High band ratio")
            }
        )
        filters.append(multiband)
        
        # Mastering EQ
        eq = AudioFilter(
            FilterType.EQ,
            {
                "low_freq": FilterParameter("low_freq", 100, 50, 200, 10, "Hz", "Low band center"),
                "low_width": FilterParameter("low_width", 100, 50, 200, 10, "Hz", "Low band width"),
                "low_gain": FilterParameter("low_gain", 0, -6, 6, 0.5, "dB", "Low band gain"),
                "mid_freq": FilterParameter("mid_freq", 1000, 800, 1500, 50, "Hz", "Mid band center"),
                "mid_width": FilterParameter("mid_width", 100, 50, 300, 10, "Hz", "Mid band width"),
                "mid_gain": FilterParameter("mid_gain", 0, -3, 3, 0.5, "dB", "Mid band gain"),
                "high_freq": FilterParameter("high_freq", 10000, 8000, 15000, 500, "Hz", "High band center"),
                "high_width": FilterParameter("high_width", 1000, 500, 2000, 100, "Hz", "High band width"),
                "high_gain": FilterParameter("high_gain", 0, -3, 3, 0.5, "dB", "High band gain")
            }
        )
        filters.append(eq)
        
        return filters

    def get_preset(self, preset_name: str) -> List[AudioFilter]:
        """Get a processing preset by name"""
        return self.presets.get(preset_name, self.presets["speech_clean"]).copy()

    def get_preset_names(self) -> List[str]:
        """Get list of available preset names"""
        return list(self.presets.keys())

    def create_custom_preset(self, name: str, filters: List[AudioFilter]) -> None:
        """Create a custom preset"""
        self.presets[name] = filters.copy()

    def suggest_preset(self, content_type: AudioContentType, analysis_result: Optional[AudioAnalysisResult] = None) -> str:
        """Suggest optimal preset based on content type and analysis"""
        if content_type == AudioContentType.SPEECH:
            if analysis_result:
                if analysis_result.noise_level > 20:
                    return "speech_broadcast"  # Better processing for noisy speech
                elif analysis_result.dynamic_range < 5:
                    return "speech_podcast"  # Consistent levels for podcast
                else:
                    return "speech_clean"  # Clean speech
            else:
                return "speech_clean"
        elif content_type == AudioContentType.MUSIC:
            if analysis_result:
                if analysis_result.features.tempo > 120:
                    return "music_electronic"  # Electronic/punchy
                elif max(analysis_result.features.mfcc_coefficients[:5]) > 0.8:
                    return "music_vocal"  # Vocal-heavy music
                else:
                    return "music_general"  # General music
            else:
                return "music_general"
        elif content_type == AudioContentType.MIXED:
            return "broadcast"  # Professional broadcast processing
        elif content_type == AudioContentType.NOISE or content_type == AudioContentType.SILENCE:
            return "noise_reduction"
        else:
            return "speech_clean"  # Default fallback


class AudioProcessingEngine:
    """Main audio processing engine"""

    def __init__(self):
        self.event_publisher = None
        try:
            self.event_publisher = get_service(IEventPublisher)
        except:
            pass
        
        self.ffmpeg_path = "ffmpeg"
        self.preset_manager = FilterPresetManager()
        self.temp_dir = Path(tempfile.gettempdir()) / "pure_sound_processing"
        self.temp_dir.mkdir(exist_ok=True)
        
        # Processing queues and status
        self.processing_queue: List[ProcessingJob] = []
        self.active_jobs: Dict[str, threading.Thread] = {}
        self.completed_jobs: Dict[str, Dict[str, Any]] = {}
        self.job_lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "jobs_processed": 0,
            "jobs_failed": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0
        }

    def create_processing_job(self, input_file: str, output_files: List[str],
                            preset_name: str = "speech_clean",
                            quality: ProcessingQuality = ProcessingQuality.HIGH_QUALITY,
                            custom_filters: Optional[List[AudioFilter]] = None) -> ProcessingJob:
        """Create a processing job"""
        # Get filters from preset
        if custom_filters:
            filters = custom_filters
        else:
            filters = self.preset_manager.get_preset(preset_name)
        
        # Set processing order
        for i, filter_obj in enumerate(filters):
            filter_obj.order = i
        
        job = ProcessingJob(
            job_id=str(uuid.uuid4()),
            input_file=input_file,
            output_files=output_files,
            filters=filters,
            format_config=self._get_format_config(quality),
            quality=quality
        )
        
        return job

    def submit_job(self, job: ProcessingJob, priority: int = 2) -> str:
        """Submit a processing job"""
        job.priority = priority
        
        with self.job_lock:
            self.processing_queue.append(job)
            # Sort by priority (higher number = higher priority)
            self.processing_queue.sort(key=lambda x: x.priority, reverse=True)
        
        # Start processing in background
        self._start_job_processing(job)
        
        return job.job_id

    def process_job_sync(self, job: ProcessingJob) -> Dict[str, Any]:
        """Process a job synchronously"""
        try:
            # Publish job start event
            if self.event_publisher:
                self.event_publisher.publish_event(
                    "processing.job_started",
                    {
                        "job_id": job.job_id,
                        "input_file": job.input_file,
                        "quality": job.quality.value
                    },
                    "AudioProcessingEngine"
                )
            
            start_time = time.time()
            
            # Create output directories
            for output_file in job.output_files:
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            
            # Build FFmpeg filter chain
            filter_chain = self._build_filter_chain(job.filters)
            
            # Process each output file
            results = []
            for output_file in job.output_files:
                result = self._process_single_output(job.input_file, output_file, filter_chain, job.format_config)
                results.append(result)
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self._update_stats(processing_time, True)
            
            # Publish completion event
            if self.event_publisher:
                self.event_publisher.publish_event(
                    "processing.job_completed",
                    {
                        "job_id": job.job_id,
                        "processing_time": processing_time,
                        "output_files": job.output_files
                    },
                    "AudioProcessingEngine"
                )
            
            return {
                "success": True,
                "job_id": job.job_id,
                "processing_time": processing_time,
                "output_files": results
            }
            
        except Exception as e:
            logging.error(f"Processing job {job.job_id} failed: {e}")
            
            # Update statistics
            self._update_stats(0, False)
            
            # Publish failure event
            if self.event_publisher:
                self.event_publisher.publish_event(
                    "processing.job_failed",
                    {
                        "job_id": job.job_id,
                        "error": str(e)
                    },
                    "AudioProcessingEngine"
                )
            
            return {
                "success": False,
                "job_id": job.job_id,
                "error": str(e)
            }

    def _start_job_processing(self, job: ProcessingJob) -> None:
        """Start job processing in background thread"""
        def process_wrapper():
            result = self.process_job_sync(job)
            
            # Store result
            with self.job_lock:
                self.completed_jobs[job.job_id] = result
                if job.job_id in self.active_jobs:
                    del self.active_jobs[job.job_id]
        
        thread = threading.Thread(target=process_wrapper, daemon=True)
        thread.start()
        
        with self.job_lock:
            self.active_jobs[job.job_id] = thread

    def _build_filter_chain(self, filters: List[AudioFilter]) -> str:
        """Build FFmpeg filter chain from filter list"""
        filter_parts = []
        
        # Sort filters by order
        sorted_filters = sorted([f for f in filters if f.enabled and not f.bypass], 
                               key=lambda x: x.order)
        
        for filter_obj in sorted_filters:
            filter_str = filter_obj.to_ffmpeg_filter()
            if filter_str:
                filter_parts.append(filter_str)
        
        return ",".join(filter_parts)

    def _process_single_output(self, input_file: str, output_file: str, 
                             filter_chain: str, format_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single output file"""
        # Build FFmpeg command
        cmd = [self.ffmpeg_path, "-y", "-i", input_file]
        
        # Add filters if present
        if filter_chain:
            cmd.extend(["-af", filter_chain])
        
        # Add format-specific options
        codec = format_config.get("codec", "libmp3lame")
        bitrate = format_config.get("bitrate", 128)
        
        cmd.extend(["-c:a", codec, "-b:a", f"{bitrate}k"])
        
        # Add metadata preservation if configured
        if format_config.get("preserve_metadata", True):
            cmd.append("-map_metadata")
            cmd.append("0")
        
        # Add output file
        cmd.append(output_file)
        
        # Run FFmpeg
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                return {
                    "success": True,
                    "output_file": output_file,
                    "command": " ".join(cmd),
                    "stderr": result.stderr
                }
            else:
                raise subprocess.CalledProcessError(
                    result.returncode, cmd, result.stdout, result.stderr
                )
                
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Processing timeout for {output_file}")
        except Exception as e:
            raise RuntimeError(f"FFmpeg processing failed: {e}")

    def _get_format_config(self, quality: ProcessingQuality) -> Dict[str, Any]:
        """Get format configuration based on quality level"""
        configs = {
            ProcessingQuality.REAL_TIME: {
                "codec": "libmp3lame",
                "bitrate": 128,
                "preserve_metadata": False,
                "threads": 2
            },
            ProcessingQuality.HIGH_QUALITY: {
                "codec": "libmp3lame", 
                "bitrate": 192,
                "preserve_metadata": True,
                "threads": 4
            },
            ProcessingQuality.MAXIMUM_QUALITY: {
                "codec": "libmp3lame",
                "bitrate": 256,
                "preserve_metadata": True,
                "threads": 8
            }
        }
        return configs.get(quality, configs[ProcessingQuality.HIGH_QUALITY])

    def _update_stats(self, processing_time: float, success: bool) -> None:
        """Update processing statistics"""
        self.stats["jobs_processed"] += 1
        if not success:
            self.stats["jobs_failed"] += 1
        
        if success:
            self.stats["total_processing_time"] += processing_time
            self.stats["average_processing_time"] = (
                self.stats["total_processing_time"] / 
                (self.stats["jobs_processed"] - self.stats["jobs_failed"])
            )

    def generate_preview_clip(self, input_file: str, duration: float = 10.0,
                            filters: Optional[List[AudioFilter]] = None) -> str:
        """Generate a preview clip with applied filters"""
        try:
            # Create temporary output file
            preview_file = self.temp_dir / f"preview_{uuid.uuid4()}.mp3"
            
            # Default to speech clean preset if no filters specified
            if filters is None:
                filters = self.preset_manager.get_preset("speech_clean")
            
            # Build filter chain
            filter_chain = self._build_filter_chain(filters)
            
            # Create preview command
            cmd = [
                self.ffmpeg_path, "-y",
                "-i", input_file,
                "-t", str(duration),  # Duration
                "-af", filter_chain,
                "-c:a", "libmp3lame",
                "-b:a", "128k",
                str(preview_file)
            ]
            
            # Run FFmpeg
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                return str(preview_file)
            else:
                raise RuntimeError(f"Preview generation failed: {result.stderr}")
                
        except Exception as e:
            logging.error(f"Preview clip generation failed: {e}")
            return ""

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job processing status"""
        with self.job_lock:
            # Check active jobs
            if job_id in self.active_jobs:
                thread = self.active_jobs[job_id]
                return {
                    "status": "running" if thread.is_alive() else "completed",
                    "job_id": job_id,
                    "progress": 0.5 if thread.is_alive() else 1.0
                }
            
            # Check completed jobs
            if job_id in self.completed_jobs:
                result = self.completed_jobs[job_id]
                return {
                    "status": "completed" if result["success"] else "failed",
                    "job_id": job_id,
                    "result": result
                }
            
            return None

    def get_queue_status(self) -> Dict[str, Any]:
        """Get processing queue status"""
        with self.job_lock:
            return {
                "queued_jobs": len(self.processing_queue),
                "active_jobs": len(self.active_jobs),
                "completed_jobs": len(self.completed_jobs),
                "stats": self.stats.copy()
            }

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a processing job"""
        with self.job_lock:
            # Remove from queue
            for i, job in enumerate(self.processing_queue):
                if job.job_id == job_id:
                    del self.processing_queue[i]
                    return True
            
            # Cancel active job
            if job_id in self.active_jobs:
                # Note: This is a simplified cancellation
                # In a real implementation, you'd need to send signals to FFmpeg
                del self.active_jobs[job_id]
                return True
            
            return False

    def get_available_presets(self) -> List[str]:
        """Get list of available filter presets"""
        return self.preset_manager.get_preset_names()

    def create_custom_preset(self, name: str, filters: List[AudioFilter]) -> None:
        """Create a custom filter preset"""
        self.preset_manager.create_custom_preset(name, filters)

    def suggest_processing_chain(self, analysis_result: AudioAnalysisResult) -> Tuple[str, List[AudioFilter]]:
        """Suggest optimal processing chain based on analysis"""
        preset_name = self.preset_manager.suggest_preset(
            analysis_result.content_type, analysis_result
        )
        filters = self.preset_manager.get_preset(preset_name)
        return preset_name, filters

    def get_filter_parameters(self, filter_type: FilterType) -> Dict[str, FilterParameter]:
        """Get parameters for a specific filter type"""
        # Create a sample filter to get parameter definitions
        sample_filter = AudioFilter(filter_type, {})
        
        # Add known parameter definitions for each filter type
        param_definitions = {
            FilterType.COMPRESSOR: {
                "threshold": FilterParameter("threshold", -20, -30, -6, 1, "dB", "Compression threshold"),
                "ratio": FilterParameter("ratio", 3, 1.5, 8, 0.5, ":1", "Compression ratio"),
                "attack": FilterParameter("attack", 0.01, 0.005, 0.1, 0.005, "s", "Attack time"),
                "release": FilterParameter("release", 0.1, 0.05, 0.5, 0.05, "s", "Release time"),
                "makeup": FilterParameter("makeup", 0, -6, 12, 0.5, "dB", "Makeup gain")
            },
            FilterType.NOISE_GATE: {
                "threshold": FilterParameter("threshold", -40, -60, -20, 1, "dB", "Gate threshold"),
                "ratio": FilterParameter("ratio", 10, 2, 20, 1, ":1", "Gate ratio"),
                "attack": FilterParameter("attack", 0.1, 0.01, 0.5, 0.01, "s", "Attack time"),
                "release": FilterParameter("release", 0.5, 0.1, 2.0, 0.1, "s", "Release time")
            },
            FilterType.LOUDNESS_NORMALIZE: {
                "target_i": FilterParameter("target_i", -16, -24, -12, 1, "LUFS", "Integrated loudness target"),
                "target_tp": FilterParameter("target_tp", -1.5, -3, 0, 0.5, "LUFS", "True peak target"),
                "target_lra": FilterParameter("target_lra", 11, 7, 17, 1, "LUFS", "LRA target")
            },
            FilterType.HIGH_PASS: {
                "frequency": FilterParameter("frequency", 80, 20, 200, 5, "Hz", "Cutoff frequency")
            },
            FilterType.LOW_PASS: {
                "frequency": FilterParameter("frequency", 8000, 1000, 20000, 500, "Hz", "Cutoff frequency")
            }
        }
        
        return param_definitions.get(filter_type, {})


# Global audio processing engine instance
audio_processing_engine = AudioProcessingEngine()