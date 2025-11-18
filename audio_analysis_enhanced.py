"""
Enhanced Audio Analysis and Content Detection Engine

This module provides advanced audio analysis capabilities including:
- Intelligent content detection (speech vs music vs other)
- Dynamic range analysis and compression recommendations
- Real-time audio feature extraction
- ML-based noise reduction recommendations
- Contextual preset suggestions
- Multi-format analysis support
"""

import os
import logging
import subprocess
import tempfile
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Audio processing libraries with fallbacks
try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("librosa not available, using basic FFmpeg analysis")

try:
    import scipy.signal
    import scipy.stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy not available, using NumPy for signal processing")

from interfaces import IEventPublisher, IServiceProvider
from di_container import get_service
from security import security_manager, Permission


class AudioContentType(Enum):
    """Audio content classification"""
    SPEECH = "speech"
    MUSIC = "music"
    MIXED = "mixed"
    SILENCE = "silence"
    NOISE = "noise"
    UNKNOWN = "unknown"


class AudioQuality(Enum):
    """Audio quality assessment"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


@dataclass
class AudioFeatures:
    """Extracted audio features"""
    spectral_centroid: float = 0.0
    spectral_rolloff: float = 0.0
    spectral_bandwidth: float = 0.0
    zero_crossing_rate: float = 0.0
    mfcc_coefficients: List[float] = field(default_factory=list)
    chroma_features: List[float] = field(default_factory=list)
    spectral_contrast: List[float] = field(default_factory=list)
    tempo: float = 0.0
    key: str = "unknown"
    loudness: float = -60.0
    dynamic_range: float = 0.0
    signal_to_noise: float = 0.0
    harmonic_to_noise: float = 0.0
    speech_probability: float = 0.0
    music_probability: float = 0.0


@dataclass
class AudioAnalysisResult:
    """Comprehensive audio analysis result"""
    file_path: str
    duration: float
    sample_rate: int
    channels: int
    format_name: str
    bitrate: Optional[int]
    file_size: int
    
    # Content analysis
    content_type: AudioContentType
    confidence: float
    
    # Audio features
    features: AudioFeatures
    
    # Quality metrics
    quality: AudioQuality
    noise_level: float
    clipping_detected: bool
    
    # Recommendations
    recommended_format: str
    recommended_bitrates: List[int]
    enable_compression: bool
    enable_noise_reduction: bool
    enable_loudness_normalization: bool
    recommended_preset: str
    
    # Processing suggestions
    processing_steps: List[str]
    warnings: List[str]
    reasoning: List[str]
    
    # Metadata
    analysis_time: float = field(default_factory=time.time)
    analysis_version: str = "1.0"


class BasicFFmpegAnalyzer:
    """Basic audio analyzer using FFmpeg (fallback when librosa not available)"""

    def __init__(self):
        self.ffmpeg_path = "ffmpeg"
        self.temp_dir = Path(tempfile.gettempdir()) / "pure_sound_analysis"
        self.temp_dir.mkdir(exist_ok=True)

    def analyze_audio(self, file_path: str) -> Dict[str, Any]:
        """Analyze audio using FFmpeg"""
        try:
            # Get basic file info
            probe_cmd = [
                self.ffmpeg_path, "-i", file_path,
                "-f", "null", "-"
            ]
            
            result = subprocess.run(
                probe_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Parse output for audio information
            output = result.stderr
            
            # Extract duration
            duration_match = None
            for line in output.split('\n'):
                if 'Duration:' in line:
                    duration_match = line
                    break
            
            duration = 0.0
            if duration_match:
                duration_str = duration_match.split('Duration:')[1].split(',')[0].strip()
                time_parts = duration_str.split(':')
                duration = float(time_parts[0]) * 3600 + float(time_parts[1]) * 60 + float(time_parts[2])
            
            # Extract audio stream info
            sample_rate = 0
            channels = 0
            bitrate = 0
            
            for line in output.split('\n'):
                if 'Audio:' in line:
                    # Parse audio stream info
                    audio_info = line.split('Audio:')[1].split(',')
                    for info in audio_info:
                        info = info.strip()
                        if 'Hz' in info:
                            sample_rate = int(info.split()[0])
                        elif 'channels' in info:
                            channels = int(info.split()[0])
                        elif 'kb/s' in info:
                            bitrate = int(float(info.replace('kb/s', '')) * 1000)
            
            return {
                "duration": duration,
                "sample_rate": sample_rate,
                "channels": channels,
                "bitrate": bitrate,
                "format": "unknown"
            }
            
        except Exception as e:
            logging.error(f"FFmpeg analysis failed for {file_path}: {e}")
            return {}

    def detect_content_type(self, file_path: str) -> AudioContentType:
        """Basic content type detection using silence analysis"""
        try:
            # Generate silence analysis
            silence_cmd = [
                self.ffmpeg_path, "-i", file_path,
                "-af", "silencedetect=noise=-50dB:d=0.5",
                "-f", "null", "-"
            ]
            
            result = subprocess.run(
                silence_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            silence_percentage = 0.0
            for line in result.stderr.split('\n'):
                if 'silence_' in line and 'percentage' in line:
                    # Extract silence percentage
                    parts = line.split(':')
                    if len(parts) > 1:
                        try:
                            silence_percentage = float(parts[1].strip().replace('%', ''))
                        except ValueError:
                            pass
            
            # Simple heuristic based on silence percentage
            if silence_percentage > 80:
                return AudioContentType.SILENCE
            elif silence_percentage < 20:
                return AudioContentType.MUSIC  # Default assumption for non-silent
            else:
                return AudioContentType.SPEECH  # Mixed content assumption
                
        except Exception as e:
            logging.error(f"Content detection failed for {file_path}: {e}")
            return AudioContentType.UNKNOWN


class AdvancedLibrosaAnalyzer:
    """Advanced audio analyzer using librosa and scipy"""

    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "pure_sound_analysis"
        self.temp_dir.mkdir(exist_ok=True)

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file with librosa"""
        try:
            y, sr = librosa.load(file_path, sr=None)
            return y, sr
        except Exception as e:
            logging.error(f"Failed to load audio file {file_path}: {e}")
            raise

    def extract_features(self, y: np.ndarray, sr: int) -> AudioFeatures:
        """Extract comprehensive audio features"""
        features = AudioFeatures()

        try:
            # Spectral features
            if len(y) > 0:
                # Spectral centroid
                features.spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
                
                # Spectral rolloff
                features.spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
                
                # Spectral bandwidth
                features.spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
                
                # Zero crossing rate
                features.zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y)))
                
                # MFCC coefficients
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                features.mfcc_coefficients = [float(x) for x in np.mean(mfcc, axis=1)]
                
                # Chroma features
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                features.chroma_features = [float(x) for x in np.mean(chroma, axis=1)]
                
                # Spectral contrast
                contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                features.spectral_contrast = [float(x) for x in np.mean(contrast, axis=1)]
                
                # Tempo and key
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                features.tempo = float(tempo)
                
                # Key estimation
                chroma_mean = np.mean(chroma, axis=1)
                key_idx = np.argmax(chroma_mean)
                keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                features.key = keys[key_idx] if key_idx < len(keys) else "unknown"
                
                # Loudness estimation (RMS energy)
                rms = librosa.feature.rms(y=y)[0]
                features.loudness = float(20 * np.log10(np.mean(rms) + 1e-10))
                
                # Dynamic range
                features.dynamic_range = float(np.max(rms) - np.min(rms + 1e-10))
                
                # Signal-to-noise estimation
                features.signal_to_noise = features.loudness - (-60.0)  # Relative to noise floor
                
                # Harmonic-to-noise ratio
                features.harmonic_to_noise = float(features.spectral_centroid / (sr / 2))

        except Exception as e:
            logging.error(f"Feature extraction failed: {e}")

        return features

    def classify_content(self, features: AudioFeatures) -> Tuple[AudioContentType, float]:
        """Classify audio content using extracted features"""
        try:
            # Heuristic-based classification
            speech_prob = 0.0
            music_prob = 0.0
            
            # Speech indicators
            if features.zero_crossing_rate > 0.1:  # High ZCR suggests speech
                speech_prob += 0.3
            if features.spectral_centroid < (features.spectral_bandwidth * 2):  # Narrow bandwidth
                speech_prob += 0.2
            if features.tempo > 60 and features.tempo < 180:  # Speech rhythm
                speech_prob += 0.1
            
            # Music indicators
            if features.spectral_contrast > 20:  # High spectral contrast
                music_prob += 0.4
            if features.tempo > 80 and features.tempo < 160:  # Musical tempo
                music_prob += 0.3
            if len(features.chroma_features) > 0 and max(features.chroma_features) > 0.8:  # Strong key
                music_prob += 0.2
            
            # Use MFCC for more sophisticated classification
            if len(features.mfcc_coefficients) > 0:
                # First MFCC coefficient relates to overall energy
                if features.mfcc_coefficients[0] > -10:
                    speech_prob += 0.2
                # Other coefficients relate to spectral shape
                spectral_variance = np.var(features.mfcc_coefficients[1:])
                if spectral_variance < 100:
                    speech_prob += 0.1
                else:
                    music_prob += 0.1
            
            # Normalize probabilities
            total = speech_prob + music_prob
            if total > 0:
                speech_prob /= total
                music_prob /= total
            
            # Set probabilities in features
            features.speech_probability = speech_prob
            features.music_probability = music_prob
            
            # Determine content type
            if speech_prob > 0.6:
                return AudioContentType.SPEECH, speech_prob
            elif music_prob > 0.6:
                return AudioContentType.MUSIC, music_prob
            elif abs(speech_prob - music_prob) < 0.2:
                return AudioContentType.MIXED, max(speech_prob, music_prob)
            else:
                return AudioContentType.UNKNOWN, max(speech_prob, music_prob)
                
        except Exception as e:
            logging.error(f"Content classification failed: {e}")
            return AudioContentType.UNKNOWN, 0.0

    def assess_quality(self, features: AudioFeatures) -> AudioQuality:
        """Assess audio quality based on features"""
        try:
            quality_score = 0.0
            
            # Good loudness range (-20 to -6 dB)
            if -20 <= features.loudness <= -6:
                quality_score += 0.3
            elif features.loudness < -30 or features.loudness > 0:
                quality_score -= 0.3
            
            # Good dynamic range (6-20 dB)
            if 6 <= features.dynamic_range <= 20:
                quality_score += 0.2
            elif features.dynamic_range < 3:
                quality_score -= 0.2
            
            # Signal-to-noise ratio
            if features.signal_to_noise > 20:
                quality_score += 0.2
            elif features.signal_to_noise < 10:
                quality_score -= 0.2
            
            # Harmonic content
            if features.harmonic_to_noise > 0.5:
                quality_score += 0.1
            elif features.harmonic_to_noise < 0.2:
                quality_score -= 0.1
            
            # Spectral balance
            if features.spectral_bandwidth > 1000 and features.spectral_bandwidth < 8000:
                quality_score += 0.1
            
            # Map score to quality
            if quality_score >= 0.7:
                return AudioQuality.EXCELLENT
            elif quality_score >= 0.4:
                return AudioQuality.GOOD
            elif quality_score >= 0.1:
                return AudioQuality.FAIR
            elif quality_score >= -0.2:
                return AudioQuality.POOR
            else:
                return AudioQuality.UNACCEPTABLE
                
        except Exception as e:
            logging.error(f"Quality assessment failed: {e}")
            return AudioQuality.FAIR

    def analyze_audio(self, file_path: str) -> Dict[str, Any]:
        """Comprehensive audio analysis"""
        try:
            # Load audio
            y, sr = self.load_audio(file_path)
            
            # Get file info
            file_info = {
                "duration": float(len(y) / sr),
                "sample_rate": sr,
                "channels": 1 if len(y.shape) == 1 else y.shape[1],
                "format": Path(file_path).suffix[1:].lower()
            }
            
            # Extract features
            features = self.extract_features(y, sr)
            
            # Classify content
            content_type, confidence = self.classify_content(features)
            
            # Assess quality
            quality = self.assess_quality(features)
            
            return {
                **file_info,
                "features": features,
                "content_type": content_type,
                "confidence": confidence,
                "quality": quality
            }
            
        except Exception as e:
            logging.error(f"Audio analysis failed for {file_path}: {e}")
            return {}


class AudioAnalysisEngine:
    """Main audio analysis engine coordinating multiple analyzers"""

    def __init__(self):
        self.event_publisher = None
        try:
            self.event_publisher = get_service(IEventPublisher)
        except:
            pass
        
        # Select analyzer based on available libraries
        if LIBROSA_AVAILABLE:
            self.analyzer = AdvancedLibrosaAnalyzer()
            self.analyzer_name = "librosa"
        else:
            self.analyzer = BasicFFmpegAnalyzer()
            self.analyzer_name = "ffmpeg"
        
        # Analysis caching
        self._analysis_cache: Dict[str, AudioAnalysisResult] = {}
        self._cache_lock = threading.RLock()
        
        # Configuration
        self.cache_enabled = True
        self.max_cache_size = 100
        self.feature_extraction_timeout = 300  # 5 minutes

    def analyze_file(self, file_path: str, use_cache: bool = True) -> Optional[AudioAnalysisResult]:
        """Analyze a single audio file"""
        file_path = str(Path(file_path).resolve())
        
        # Check cache first
        if use_cache and self.cache_enabled:
            with self._cache_lock:
                if file_path in self._analysis_cache:
                    cached_result = self._analysis_cache[file_path]
                    # Check if cache is still valid (24 hours)
                    if time.time() - cached_result.analysis_time < 86400:
                        return cached_result
        
        try:
            # Basic file validation
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            
            if not self._is_audio_file(file_path):
                raise ValueError(f"File is not a supported audio format: {file_path}")
            
            # Perform analysis
            start_time = time.time()
            
            if isinstance(self.analyzer, AdvancedLibrosaAnalyzer):
                analysis_data = self.analyzer.analyze_audio(file_path)
            else:
                # Basic FFmpeg analysis
                analysis_data = self.analyzer.analyze_audio(file_path)
                content_type = self.analyzer.detect_content_type(file_path)
                analysis_data["content_type"] = content_type
                analysis_data["confidence"] = 0.5
                analysis_data["quality"] = AudioQuality.FAIR
                analysis_data["features"] = AudioFeatures()
            
            if not analysis_data:
                return None
            
            # Create comprehensive result
            result = self._create_analysis_result(file_path, analysis_data)
            
            # Cache result
            if use_cache and self.cache_enabled:
                with self._cache_lock:
                    # Clean cache if too large
                    if len(self._analysis_cache) >= self.max_cache_size:
                        # Remove oldest entries
                        oldest_keys = sorted(
                            self._analysis_cache.keys(),
                            key=lambda k: self._analysis_cache[k].analysis_time
                        )[:len(self._analysis_cache) - self.max_cache_size + 10]
                        for key in oldest_keys:
                            del self._analysis_cache[key]
                    
                    self._analysis_cache[file_path] = result
            
            # Publish event
            if self.event_publisher:
                self.event_publisher.publish_event(
                    "audio.analyzed",
                    {
                        "file_path": file_path,
                        "content_type": result.content_type.value,
                        "quality": result.quality.value,
                        "duration": result.duration,
                        "analyzer": self.analyzer_name
                    },
                    "AudioAnalysisEngine"
                )
            
            return result
            
        except Exception as e:
            logging.error(f"Audio analysis failed for {file_path}: {e}")
            
            if self.event_publisher:
                self.event_publisher.publish_event(
                    "audio.analysis_failed",
                    {
                        "file_path": file_path,
                        "error": str(e)
                    },
                    "AudioAnalysisEngine"
                )
            
            return None

    def batch_analyze(self, file_paths: List[str], max_workers: int = 4) -> Dict[str, AudioAnalysisResult]:
        """Analyze multiple audio files in parallel"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all analysis tasks
            future_to_file = {
                executor.submit(self.analyze_file, file_path): file_path
                for file_path in file_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result(timeout=self.feature_extraction_timeout)
                    if result:
                        results[file_path] = result
                except Exception as e:
                    logging.error(f"Batch analysis failed for {file_path}: {e}")
        
        return results

    def get_quick_stats(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get quick statistics without full analysis"""
        try:
            if isinstance(self.analyzer, AdvancedLibrosaAnalyzer):
                # Quick analysis with librosa
                y, sr = self.analyzer.load_audio(file_path)
                duration = len(y) / sr
                
                # Basic statistics
                rms = np.sqrt(np.mean(y**2))
                peak = np.max(np.abs(y))
                zero_crossings = np.sum(np.diff(np.sign(y)) != 0)
                
                return {
                    "duration": duration,
                    "sample_rate": sr,
                    "channels": 1 if len(y.shape) == 1 else y.shape[1],
                    "rms_level": float(20 * np.log10(rms + 1e-10)),
                    "peak_level": float(20 * np.log10(peak + 1e-10)),
                    "zero_crossing_rate": float(zero_crossings / len(y)),
                    "dynamic_range": float(20 * np.log10(peak / (rms + 1e-10)))
                }
            else:
                # Quick stats with FFmpeg
                basic_info = self.analyzer.analyze_audio(file_path)
                if basic_info:
                    return {
                        "duration": basic_info.get("duration", 0),
                        "sample_rate": basic_info.get("sample_rate", 0),
                        "channels": basic_info.get("channels", 0),
                        "bitrate": basic_info.get("bitrate", 0)
                    }
            
            return None
            
        except Exception as e:
            logging.error(f"Quick stats failed for {file_path}: {e}")
            return None

    def get_recommendations(self, result: AudioAnalysisResult) -> Dict[str, Any]:
        """Generate processing recommendations based on analysis"""
        recommendations = {
            "format": "mp3",
            "bitrates": [128],
            "compression_settings": {},
            "processing_steps": [],
            "preset_suggestion": "default"
        }
        
        # Format recommendations based on content type
        if result.content_type == AudioContentType.SPEECH:
            recommendations["format"] = "opus"
            recommendations["bitrates"] = [24, 32, 48]
            recommendations["preset_suggestion"] = "speech_optimization"
            recommendations["processing_steps"] = [
                "Enable loudness normalization",
                "Apply noise gate if needed",
                "Consider noise reduction for poor quality"
            ]
        elif result.content_type == AudioContentType.MUSIC:
            recommendations["format"] = "aac"
            recommendations["bitrates"] = [128, 192, 256]
            recommendations["preset_suggestion"] = "music_optimization"
            recommendations["processing_steps"] = [
                "Preserve dynamic range",
                "Apply multi-band compression for professional sound"
            ]
        elif result.content_type == AudioContentType.MIXED:
            recommendations["format"] = "aac"
            recommendations["bitrates"] = [96, 128, 160]
            recommendations["preset_suggestion"] = "broadcast_optimization"
        
        # Quality-based adjustments
        if result.quality in [AudioQuality.POOR, AudioQuality.UNACCEPTABLE]:
            recommendations["compression_settings"]["noise_reduction"] = True
            recommendations["processing_steps"].append("Apply ML-based noise reduction")
        
        if result.clipping_detected:
            recommendations["processing_steps"].append("Apply de-essing to reduce clipping")
            recommendations["compression_settings"]["de_esser"] = True
        
        # Dynamic range considerations
        if result.features.dynamic_range < 6:
            recommendations["compression_settings"]["gentle_compression"] = True
        elif result.features.dynamic_range > 20:
            recommendations["compression_settings"]["preserve_loudness"] = True
        
        return recommendations

    def _is_audio_file(self, file_path: str) -> bool:
        """Check if file is a supported audio format"""
        audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg', '.opus'}
        return Path(file_path).suffix.lower() in audio_extensions

    def _create_analysis_result(self, file_path: str, analysis_data: Dict[str, Any]) -> AudioAnalysisResult:
        """Create AudioAnalysisResult from analysis data"""
        # Extract basic info
        duration = analysis_data.get("duration", 0.0)
        sample_rate = analysis_data.get("sample_rate", 44100)
        channels = analysis_data.get("channels", 1)
        format_name = analysis_data.get("format", "unknown")
        bitrate = analysis_data.get("bitrate")
        
        # Get file size
        try:
            file_size = os.path.getsize(file_path)
        except:
            file_size = 0
        
        # Extract content and quality
        content_type = analysis_data.get("content_type", AudioContentType.UNKNOWN)
        confidence = analysis_data.get("confidence", 0.0)
        quality = analysis_data.get("quality", AudioQuality.FAIR)
        
        # Extract features
        features = analysis_data.get("features", AudioFeatures())
        
        # Quality metrics
        noise_level = abs(features.loudness) if features.loudness < 0 else 0
        clipping_detected = features.loudness > -1  # Very loud suggests clipping
        
        # Generate recommendations
        temp_result = AudioAnalysisResult(
            file_path=file_path,
            duration=duration,
            sample_rate=sample_rate,
            channels=channels,
            format_name=format_name,
            bitrate=bitrate,
            file_size=file_size,
            content_type=content_type,
            confidence=confidence,
            features=features,
            quality=quality,
            noise_level=noise_level,
            clipping_detected=clipping_detected,
            recommended_format="mp3",
            recommended_bitrates=[128],
            enable_compression=False,
            enable_noise_reduction=False,
            enable_loudness_normalization=False,
            recommended_preset="default",
            processing_steps=[],
            warnings=[],
            reasoning=[]
        )
        
        # Get actual recommendations
        recommendations = self.get_recommendations(temp_result)
        temp_result.recommended_format = recommendations["format"]
        temp_result.recommended_bitrates = recommendations["bitrates"]
        temp_result.recommended_preset = recommendations["preset_suggestion"]
        temp_result.processing_steps = recommendations["processing_steps"]
        
        # Set processing flags based on content
        temp_result.enable_compression = content_type != AudioContentType.SILENCE
        temp_result.enable_noise_reduction = quality in [AudioQuality.POOR, AudioQuality.UNACCEPTABLE]
        temp_result.enable_loudness_normalization = content_type == AudioContentType.SPEECH
        
        # Generate warnings
        if quality == AudioQuality.UNACCEPTABLE:
            temp_result.warnings.append("Audio quality is unacceptable and may need significant processing")
        if clipping_detected:
            temp_result.warnings.append("Audio clipping detected - may need dynamic processing")
        if features.dynamic_range < 3:
            temp_result.warnings.append("Very low dynamic range - audio may be over-compressed")
        
        # Generate reasoning
        temp_result.reasoning.append(f"Content classified as {content_type.value} with {confidence:.2f} confidence")
        temp_result.reasoning.append(f"Audio quality assessed as {quality.value}")
        temp_result.reasoning.append(f"Recommended format: {temp_result.recommended_format}")
        
        return temp_result

    def clear_cache(self) -> None:
        """Clear the analysis cache"""
        with self._cache_lock:
            self._analysis_cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._cache_lock:
            return {
                "cached_files": len(self._analysis_cache),
                "max_size": self.max_cache_size,
                "cache_enabled": self.cache_enabled
            }

    def get_supported_analysis_types(self) -> List[str]:
        """Get list of supported analysis types"""
        analysis_types = [
            "content_detection",
            "quality_assessment", 
            "feature_extraction",
            "noise_analysis",
            "dynamic_range_analysis",
            "loudness_analysis",
            "spectral_analysis"
        ]
        
        if LIBROSA_AVAILABLE:
            analysis_types.extend([
                "mfcc_analysis",
                "chroma_analysis", 
                "tempo_detection",
                "key_detection"
            ])
        
        return analysis_types


# Global audio analysis engine instance
audio_analysis_engine = AudioAnalysisEngine()