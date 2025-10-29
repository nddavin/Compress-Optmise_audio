import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import subprocess
import tempfile

class MultiStreamProcessor:
    """Handles multiple output streams for audio processing"""

    def __init__(self):
        self.supported_formats = ["mp3", "aac", "ogg", "opus", "flac"]
        self.max_concurrent_streams = 4  # Limit concurrent processing

    def create_multiple_outputs(self,
                              input_file: str,
                              output_base: str,
                              bitrates: List[int],
                              formats: List[str],
                              filter_chain: Optional[str] = None,
                              channels: int = 1,
                              preserve_metadata: bool = True) -> List[Dict[str, Any]]:
        """
        Create multiple output streams with different bitrates and/or formats

        Args:
            input_file: Path to input audio file
            output_base: Base directory for outputs
            bitrates: List of bitrates to generate
            formats: List of formats to generate
            filter_chain: Audio filter chain to apply
            channels: Number of audio channels
            preserve_metadata: Whether to preserve metadata

        Returns:
            List of output file information dictionaries
        """
        outputs = []

        # Create output directory structure
        os.makedirs(output_base, exist_ok=True)

        # Generate all combinations of formats and bitrates
        for fmt in formats:
            if fmt not in self.supported_formats:
                logging.warning(f"Skipping unsupported format: {fmt}")
                continue

            format_dir = os.path.join(output_base, f"format_{fmt}")
            os.makedirs(format_dir, exist_ok=True)

            for bitrate in bitrates:
                # Skip bitrate for lossless formats
                if fmt == "flac" and bitrate != bitrates[0]:
                    continue

                output_info = self._create_single_output(
                    input_file=input_file,
                    output_dir=format_dir,
                    bitrate=bitrate,
                    format=fmt,
                    filter_chain=filter_chain,
                    channels=channels,
                    preserve_metadata=preserve_metadata
                )

                if output_info:
                    outputs.append(output_info)

        return outputs

    def _create_single_output(self,
                            input_file: str,
                            output_dir: str,
                            bitrate: int,
                            format: str,
                            filter_chain: Optional[str],
                            channels: int,
                            preserve_metadata: bool) -> Optional[Dict[str, Any]]:
        """Create a single output stream"""

        # Generate output filename
        input_name = Path(input_file).stem
        if format == "flac":
            output_file = os.path.join(output_dir, f"{input_name}.flac")
        else:
            output_file = os.path.join(output_dir, f"{input_name}_{bitrate}kbps.{format}")

        # Compress audio
        from compress_audio import compress_audio

        success, input_size, output_size = compress_audio(
            input_file=input_file,
            output_file=output_file,
            bitrate=bitrate,
            filter_chain=filter_chain,
            output_format=format,
            channels=channels,
            preserve_metadata=preserve_metadata,
            dry_run=False,
            preview_mode=False
        )

        if success:
            compression_ratio = (1 - output_size / input_size) * 100 if input_size > 0 else 0

            return {
                "input_file": input_file,
                "output_file": output_file,
                "format": format,
                "bitrate": bitrate,
                "channels": channels,
                "input_size": input_size,
                "output_size": output_size,
                "compression_ratio": compression_ratio,
                "success": True
            }
        else:
            logging.error(f"Failed to create output: {output_file}")
            return None

    def create_adaptive_streaming(self,
                                input_file: str,
                                output_base: str,
                                filter_chain: Optional[str] = None,
                                channels: int = 1) -> Dict[str, Any]:
        """
        Create adaptive bitrate streaming outputs (HLS/DASH compatible)

        Args:
            input_file: Path to input audio file
            output_base: Base directory for streaming outputs
            filter_chain: Audio filter chain to apply
            channels: Number of audio channels

        Returns:
            Dictionary with streaming information
        """
        streaming_info = {
            "master_playlist": None,
            "streams": [],
            "base_url": output_base
        }

        # Create streaming directory
        streaming_dir = os.path.join(output_base, "streaming")
        os.makedirs(streaming_dir, exist_ok=True)

        # Generate multiple bitrates for adaptive streaming
        bitrates = [64, 128, 256]  # Common adaptive streaming bitrates

        for bitrate in bitrates:
            segment_dir = os.path.join(streaming_dir, f"{bitrate}k")
            os.makedirs(segment_dir, exist_ok=True)

            # Create segmented output
            success = self._create_segmented_output(
                input_file=input_file,
                output_dir=segment_dir,
                bitrate=bitrate,
                filter_chain=filter_chain,
                channels=channels
            )

            if success:
                stream_info = {
                    "bitrate": bitrate,
                    "segment_dir": segment_dir,
                    "playlist_file": os.path.join(segment_dir, "playlist.m3u8")
                }
                streaming_info["streams"].append(stream_info)

        # Create master playlist
        if streaming_info["streams"]:
            master_playlist = self._create_master_playlist(streaming_info)
            streaming_info["master_playlist"] = master_playlist

        return streaming_info

    def _create_segmented_output(self,
                               input_file: str,
                               output_dir: str,
                               bitrate: int,
                               filter_chain: Optional[str],
                               channels: int) -> bool:
        """Create segmented output for streaming"""

        try:
            # Build FFmpeg command for HLS segmentation
            cmd = ["ffmpeg", "-i", input_file]

            # Add filter chain
            if filter_chain:
                cmd.extend(["-af", filter_chain])

            # Audio settings
            cmd.extend([
                "-ac", str(channels),
                "-ar", "44100",
                "-b:a", f"{bitrate}k",
                "-c:a", "aac"
            ])

            # HLS settings
            playlist_file = os.path.join(output_dir, "playlist.m3u8")
            segment_pattern = os.path.join(output_dir, "segment_%03d.ts")

            cmd.extend([
                "-f", "hls",
                "-hls_time", "10",  # 10 second segments
                "-hls_list_size", "0",  # Keep all segments
                "-hls_segment_filename", segment_pattern,
                playlist_file,
                "-y"  # Overwrite
            ])

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logging.info(f"Created HLS segments for {bitrate}kbps")
            return True

        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to create segmented output: {e.stderr}")
            return False

    def _create_master_playlist(self, streaming_info: Dict[str, Any]) -> str:
        """Create master playlist for adaptive streaming"""

        master_playlist_path = os.path.join(streaming_info["base_url"], "streaming", "master.m3u8")

        with open(master_playlist_path, 'w') as f:
            f.write("#EXTM3U\n")
            f.write("#EXT-X-VERSION:3\n")

            for stream in streaming_info["streams"]:
                bitrate = stream["bitrate"]
                bandwidth = bitrate * 1000  # Convert to bits per second
                playlist_path = os.path.relpath(stream["playlist_file"], os.path.dirname(master_playlist_path))

                f.write(f"#EXT-X-STREAM-INF:BANDWIDTH={bandwidth},CODECS=\"mp4a.40.2\"\n")
                f.write(f"{playlist_path}\n")

        return master_playlist_path

class SelectiveChannelProcessor:
    """Handles selective filter application per audio channel"""

    def __init__(self):
        pass

    def apply_channel_filters(self,
                            input_file: str,
                            output_file: str,
                            channel_filters: Dict[int, str],
                            base_filter: Optional[str] = None,
                            channels: int = 2) -> bool:
        """
        Apply different filters to different audio channels

        Args:
            input_file: Input audio file
            output_file: Output audio file
            channel_filters: Dictionary mapping channel index to filter string
            base_filter: Base filter applied to all channels
            channels: Total number of channels

        Returns:
            True if successful, False otherwise
        """

        if channels == 1:
            # Mono - apply base filter only
            filter_chain = base_filter or ""
        else:
            # Multi-channel - apply selective filters
            filter_parts = []

            if base_filter:
                filter_parts.append(base_filter)

            # Apply channel-specific filters
            for channel_idx, filter_str in channel_filters.items():
                if channel_idx < channels:
                    # Extract specific channel, apply filter, then remix
                    channel_filter = f"[0:a]pan=mono|c0=c{channel_idx}[ch{channel_idx}];[ch{channel_idx}]{filter_str}[filtered_ch{channel_idx}]"
                    filter_parts.append(channel_filter)

            # Mix filtered channels back together
            if len(channel_filters) == channels:
                # All channels have specific filters
                mix_inputs = "".join(f"[filtered_ch{i}]" for i in range(channels))
                mix_filter = f"{mix_inputs}amix=inputs={channels}:duration=longest[final]"
                filter_parts.append(mix_filter)
            elif channel_filters:
                # Some channels filtered, others pass through
                filtered_channels = list(channel_filters.keys())
                passthrough_channels = [i for i in range(channels) if i not in filtered_channels]

                # Mix filtered and passthrough channels
                mix_parts = []
                for i in filtered_channels:
                    mix_parts.append(f"[filtered_ch{i}]")
                for i in passthrough_channels:
                    mix_parts.append(f"[0:a]pan=mono|c0=c{i}")

                mix_inputs = "".join(mix_parts)
                mix_filter = f"{mix_inputs}amix=inputs={channels}:duration=longest[final]"
                filter_parts.append(mix_filter)
            else:
                # No channel-specific filters, just base filter
                filter_parts.append("acopy[final]")

            filter_chain = ",".join(filter_parts) + ";[final]acopy"

        # Apply the filter chain
        try:
            cmd = ["ffmpeg", "-i", input_file]

            if filter_chain:
                cmd.extend(["-af", filter_chain])

            cmd.extend([output_file, "-y"])

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logging.info(f"Applied selective channel filters to {output_file}")
            return True

        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to apply channel filters: {e.stderr}")
            return False

    def create_channel_split_outputs(self,
                                   input_file: str,
                                   output_base: str,
                                   channels: int = 2) -> List[str]:
        """
        Create separate output files for each audio channel

        Args:
            input_file: Input audio file
            output_base: Base directory for outputs
            channels: Number of channels to split

        Returns:
            List of output file paths
        """
        outputs = []
        input_name = Path(input_file).stem

        os.makedirs(output_base, exist_ok=True)

        for channel_idx in range(channels):
            output_file = os.path.join(output_base, f"{input_name}_channel_{channel_idx}.wav")

            try:
                # Extract single channel
                cmd = [
                    "ffmpeg", "-i", input_file,
                    "-af", f"pan=mono|c0=c{channel_idx}",
                    "-c:a", "pcm_s16le",
                    output_file, "-y"
                ]

                subprocess.run(cmd, capture_output=True, check=True)
                outputs.append(output_file)
                logging.info(f"Extracted channel {channel_idx} to {output_file}")

            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to extract channel {channel_idx}: {e.stderr}")

        return outputs

# Global instances
multi_stream_processor = MultiStreamProcessor()
selective_channel_processor = SelectiveChannelProcessor()