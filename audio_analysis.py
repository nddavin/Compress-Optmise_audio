import os
import subprocess
import json
import logging
import tempfile
from typing import Dict, Any, Optional, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logging.warning("NumPy not available. Audio analysis features will be limited.")

class AudioAnalyzer:
    """Analyzes audio files to provide compression recommendations"""

    def __init__(self):
        self.ffmpeg_available = self._check_ffmpeg()

    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available"""
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def analyze_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Analyze an audio file and return comprehensive statistics with graceful degradation"""
        if not self.ffmpeg_available:
            logging.warning("FFmpeg not available for audio analysis. Returning basic fallback analysis.")
            return self._fallback_analysis(file_path)

        if not os.path.exists(file_path):
            logging.error(f"Audio file does not exist: {file_path}")
            return None

        try:
            # Get basic audio info using ffprobe
            probe_cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", file_path
            ]

            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True, timeout=30)
            probe_data = json.loads(result.stdout)

            # Extract audio stream info
            audio_stream = None
            for stream in probe_data.get("streams", []):
                if stream.get("codec_type") == "audio":
                    audio_stream = stream
                    break

            if not audio_stream:
                logging.error(f"No audio stream found in {file_path}")
                return self._fallback_analysis(file_path)

            # Get detailed audio statistics
            stats = self._extract_audio_stats(audio_stream, probe_data.get("format", {}))

            # Analyze content characteristics with fallback
            try:
                content_analysis = self._analyze_content_characteristics(file_path, stats)
            except Exception as e:
                logging.warning(f"Content analysis failed, using fallback: {e}")
                content_analysis = self._fallback_content_analysis(stats)

            return {
                "basic_info": stats,
                "content_analysis": content_analysis,
                "recommendations": self._generate_recommendations(stats, content_analysis)
            }

        except subprocess.TimeoutExpired:
            logging.warning(f"Analysis timed out for {file_path}, using fallback analysis")
            return self._fallback_analysis(file_path)
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            logging.error(f"Failed to analyze audio file {file_path}: {e}")
            return self._fallback_analysis(file_path)
        except Exception as e:
            logging.error(f"Unexpected error during analysis of {file_path}: {e}")
            return self._fallback_analysis(file_path)

    def _extract_audio_stats(self, stream: Dict[str, Any], format_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic audio statistics from ffprobe output"""
        stats = {
            "codec": stream.get("codec_name", "unknown"),
            "sample_rate": int(stream.get("sample_rate", 44100)),
            "channels": int(stream.get("channels", 1)),
            "channel_layout": stream.get("channel_layout", "unknown"),
            "bit_depth": stream.get("bits_per_sample", 16),
            "duration": float(format_info.get("duration", 0)),
            "bitrate": int(format_info.get("bit_rate", 128000)),
            "size_bytes": int(format_info.get("size", 0))
        }

        # Calculate derived statistics
        stats["size_mb"] = stats["size_bytes"] / (1024 * 1024)
        stats["bitrate_kbps"] = stats["bitrate"] / 1000

        return stats

    def _analyze_content_characteristics(self, file_path: str, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze audio content characteristics for compression recommendations"""
        analysis = {
            "content_type": "unknown",
            "dynamic_range": "unknown",
            "frequency_content": "unknown",
            "noise_level": "unknown",
            "speech_probability": 0,
            "music_probability": 0
        }

        try:
            # Use FFmpeg to extract audio samples for analysis
            # Create a temporary WAV file for analysis (first 30 seconds)
            temp_wav = self._extract_audio_segment(file_path, 30)

            if temp_wav and os.path.exists(temp_wav):
                try:
                    # Analyze the extracted segment
                    analysis.update(self._analyze_audio_segment(temp_wav, stats))
                finally:
                    # Clean up temp file
                    os.remove(temp_wav)

        except Exception as e:
            logging.warning(f"Content analysis failed: {e}")
        return analysis

    def _extract_audio_segment(self, input_file: str, duration: int = 30) -> Optional[str]:
        """Extract a short audio segment for analysis"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name

            cmd = [
                "ffmpeg", "-i", input_file, "-t", str(duration), "-acodec", "pcm_s16le",
                "-ar", "44100", "-ac", "1", "-y", temp_path
            ]

            subprocess.run(cmd, capture_output=True, check=True)
            return temp_path

        except subprocess.CalledProcessError:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return None

    def _analyze_audio_segment(self, wav_file: str, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a WAV audio segment"""
        analysis = {}

        if not HAS_NUMPY:
            # Basic analysis without numpy
            analysis["content_type"] = "unknown"
            analysis["dynamic_range"] = "unknown"
            analysis["speech_probability"] = 0
            analysis["music_probability"] = 0
            return analysis

        try:
            # Read WAV file and analyze
            import wave

            with wave.open(wav_file, 'rb') as wav:
                # Read audio data
                frames = wav.readframes(wav.getnframes())
                sample_width = wav.getsampwidth()
                frame_rate = wav.getframerate()

                # Convert to numpy array for analysis
                if sample_width == 2:  # 16-bit
                    audio_data = np.frombuffer(frames, dtype=np.int16)
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif sample_width == 4:  # 32-bit
                    audio_data = np.frombuffer(frames, dtype=np.int32)
                    audio_data = audio_data.astype(np.float32) / 2147483648.0
                else:
                    # 8-bit or other
                    audio_data = np.frombuffer(frames, dtype=np.uint8)
                    audio_data = audio_data.astype(np.float32) / 128.0 - 1.0

                # Basic analysis
                rms = np.sqrt(np.mean(audio_data**2))
                peak = np.max(np.abs(audio_data))

                # Dynamic range estimation
                if rms > 0:
                    dynamic_range_db = 20 * np.log10(peak / rms)
                    analysis["dynamic_range"] = f"{dynamic_range_db:.1f} dB"
                else:
                    analysis["dynamic_range"] = "unknown"

                # Content type detection (simple heuristics)
                # Speech typically has lower RMS and more consistent levels
                # Music typically has higher dynamic range and more variation
                speech_score = 0
                music_score = 0

                # RMS level (speech often quieter)
                if rms < 0.1:
                    speech_score += 1
                else:
                    music_score += 1

                # Dynamic range (music often has wider range)
                if 'dynamic_range' in analysis:
                    dr_value = float(analysis["dynamic_range"].split()[0])
                    if dr_value > 15:
                        music_score += 1
                    else:
                        speech_score += 1

                # Normalize scores
                total_score = speech_score + music_score
                if total_score > 0:
                    analysis["speech_probability"] = speech_score / total_score
                    analysis["music_probability"] = music_score / total_score

                    if analysis["speech_probability"] > 0.6:
                        analysis["content_type"] = "speech"
                    elif analysis["music_probability"] > 0.6:
                        analysis["content_type"] = "music"
                    else:
                        analysis["content_type"] = "mixed"

                # Frequency content estimation (simplified)
                # This is a very basic frequency analysis
                if len(audio_data) > 1024:
                    # Simple FFT for rough frequency content
                    fft = np.fft.fft(audio_data[:4096])
                    freqs = np.fft.fftfreq(len(fft), 1/frame_rate)

                    # Focus on audible range (20Hz - 20kHz)
                    audible_mask = (freqs >= 20) & (freqs <= 20000)
                    if np.any(audible_mask):
                        fft_audible = np.abs(fft[audible_mask])

                        # Check for high frequency content
                        high_freq_mask = freqs[audible_mask] > 8000
                        if np.any(high_freq_mask):
                            high_energy = np.sum(fft_audible[high_freq_mask])
                            total_energy = np.sum(fft_audible)

                            if total_energy > 0:
                                high_freq_ratio = high_energy / total_energy
                                if high_freq_ratio > 0.3:
                                    analysis["frequency_content"] = "wide"
                                elif high_freq_ratio > 0.1:
                                    analysis["frequency_content"] = "medium"
                                else:
                                    analysis["frequency_content"] = "narrow"

        except Exception as e:
            logging.warning(f"Audio segment analysis failed: {e}")

        return analysis

    def _generate_recommendations(self, stats: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate compression recommendations based on analysis"""
        recommendations = {
            "format": "mp3",
            "bitrates": [128],
            "content_type": analysis.get("content_type", "speech"),
            "enable_compression": False,
            "enable_multiband": False,
            "enable_loudnorm": True,
            "enable_noise_reduction": False,
            "reasoning": []
        }

        # Format recommendation
        if stats["codec"] in ["flac", "wav", "aiff"]:
            recommendations["format"] = "mp3"  # Lossy compression for lossless sources
            recommendations["reasoning"].append("Converting from lossless to lossy format")
        elif stats["codec"] in ["aac", "ogg", "opus"]:
            recommendations["format"] = stats["codec"]  # Keep same format
            recommendations["reasoning"].append("Keeping existing compressed format")

        # Bitrate recommendations based on content type
        content_type = analysis.get("content_type", "speech")

        if content_type == "speech":
            recommendations["bitrates"] = [64, 96, 128]
            recommendations["reasoning"].append("Speech content detected - using lower bitrates")
        elif content_type == "music":
            recommendations["bitrates"] = [128, 192, 256]
            recommendations["reasoning"].append("Music content detected - using higher bitrates")
        else:
            recommendations["bitrates"] = [96, 128, 192]
            recommendations["reasoning"].append("Mixed content - using moderate bitrates")

        # Compression recommendations
        dynamic_range = analysis.get("dynamic_range", "unknown")
        if dynamic_range != "unknown":
            try:
                dr_value = float(dynamic_range.split()[0])
                if dr_value > 20:
                    recommendations["enable_compression"] = True
                    recommendations["reasoning"].append("High dynamic range detected - compression recommended")
                elif dr_value < 10:
                    recommendations["enable_multiband"] = True
                    recommendations["reasoning"].append("Low dynamic range - multiband compression may help")
            except ValueError:
                pass

        # Quality-based recommendations
        if stats["bitrate_kbps"] > 256:
            recommendations["reasoning"].append("High bitrate source - can compress more aggressively")

        return recommendations

    def get_quick_stats(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get quick statistics without full analysis with graceful degradation"""
        if not self.ffmpeg_available:
            # Fallback to basic file stats
            return self._get_basic_file_stats(file_path)

        try:
            probe_cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", file_path
            ]

            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True, timeout=15)
            data = json.loads(result.stdout)

            audio_stream = None
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "audio":
                    audio_stream = stream
                    break

            if audio_stream:
                format_info = data.get("format", {})
                stats = self._extract_audio_stats(audio_stream, format_info)

                return {
                    "codec": stats["codec"],
                    "sample_rate": stats["sample_rate"],
                    "channels": stats["channels"],
                    "duration": stats["duration"],
                    "bitrate_kbps": stats["bitrate_kbps"],
                    "size_mb": stats["size_mb"]
                }

        except subprocess.TimeoutExpired:
            logging.warning(f"Quick stats timed out for {file_path}, using basic file stats")
            return self._get_basic_file_stats(file_path)
        except Exception as e:
            logging.error(f"Failed to get quick stats for {file_path}: {e}")
            return self._get_basic_file_stats(file_path)

        return None

    def _get_basic_file_stats(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get basic file statistics when FFmpeg is unavailable"""
        try:
            if not os.path.exists(file_path):
                return None

            file_size = os.path.getsize(file_path)
            return {
                "codec": "unknown",
                "sample_rate": 44100,  # Assume standard rate
                "channels": 2,  # Assume stereo
                "duration": 0,  # Unknown
                "bitrate_kbps": 0,  # Unknown
                "size_mb": file_size / (1024 * 1024)
            }
        except Exception as e:
            logging.error(f"Failed to get basic file stats for {file_path}: {e}")
            return None

    def _fallback_analysis(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Provide fallback analysis when full analysis fails"""
        try:
            basic_stats = self._get_basic_file_stats(file_path)
            if not basic_stats:
                return None

            # Create minimal analysis structure
            return {
                "basic_info": basic_stats,
                "content_analysis": {
                    "content_type": "unknown",
                    "dynamic_range": "unknown",
                    "frequency_content": "unknown",
                    "noise_level": "unknown",
                    "speech_probability": 0.0,
                    "music_probability": 0.0
                },
                "recommendations": {
                    "format": "mp3",
                    "bitrates": [128],
                    "content_type": "unknown",
                    "enable_compression": False,
                    "enable_multiband": False,
                    "enable_loudnorm": True,
                    "enable_noise_reduction": False,
                    "reasoning": ["Analysis unavailable - using conservative defaults"]
                }
            }
        except Exception as e:
            logging.error(f"Fallback analysis failed for {file_path}: {e}")
            return None

    def _fallback_content_analysis(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Provide fallback content analysis"""
        return {
            "content_type": "unknown",
            "dynamic_range": "unknown",
            "frequency_content": "unknown",
            "noise_level": "unknown",
            "speech_probability": 0.0,
            "music_probability": 0.0
        }

# Global analyzer instance
audio_analyzer = AudioAnalyzer()