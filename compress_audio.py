import os
import subprocess
import argparse
import logging
import sys
import time
import multiprocessing
from pathlib import Path
from config import config_manager
from audio_analysis import audio_analyzer
from job_queue import job_queue
from cloud_integration import cloud_manager, distributed_processor, storage_manager
from multi_stream import multi_stream_processor, selective_channel_processor

def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        logging.info("FFmpeg is available.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("FFmpeg is not installed or not in PATH.")
        logging.error("Please install FFmpeg to use this audio compression tool:")
        logging.error("  - macOS: brew install ffmpeg")
        logging.error("  - Ubuntu/Debian: sudo apt install ffmpeg")
        logging.error("  - CentOS/RHEL: sudo yum install ffmpeg")
        logging.error("  - Windows: Download from https://ffmpeg.org/download.html")
        logging.error("")
        logging.error("After installation, ensure FFmpeg is in your system PATH.")
        logging.error("You can verify installation by running: ffmpeg -version")
        sys.exit(1)

def create_output_dirs(output_base, bitrates):
    output_dirs = {}
    for bitrate in bitrates:
        dir_name = f"optimised-{bitrate}kbps"
        full_path = os.path.join(output_base, dir_name)
        os.makedirs(full_path, exist_ok=True)
        output_dirs[bitrate] = full_path
    return output_dirs

def get_format_defaults(output_format, content_type="speech"):
    """Get codec, extension, and recommended bitrates for each format"""
    # Try to get from config first
    config_format = config_manager.get_format_config(output_format)
    if config_format:
        return config_format

    # Fallback to hardcoded defaults
    defaults = {
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
            "speech": [24, 32, 48],  # Opus excels at low bitrates for speech
            "music": [64, 96, 128]
        },
        "flac": {
            "codec": "flac",
            "ext": ".flac",
            "speech": [],  # Lossless doesn't use bitrates
            "music": []
        }
    }
    return defaults.get(output_format, {})

def get_compressor_preset(preset="speech"):
    """Get compressor presets for different content types"""
    # Try to get from config first
    config_preset = config_manager.get_preset(preset, "compressor")
    if config_preset:
        return config_preset

    # Fallback to hardcoded defaults
    presets = {
        "speech": {
            "threshold": -20,
            "ratio": 3,
            "attack": 0.01,
            "release": 0.1,
            "makeup": 6
        },
        "music": {
            "threshold": -18,
            "ratio": 4,
            "attack": 0.005,
            "release": 0.05,
            "makeup": 4
        },
        "broadcast": {
            "threshold": -15,
            "ratio": 6,
            "attack": 0.002,
            "release": 0.02,
            "makeup": 8
        },
        "gentle": {
            "threshold": -25,
            "ratio": 2,
            "attack": 0.02,
            "release": 0.2,
            "makeup": 3
        }
    }
    return presets.get(preset, presets["speech"])

def get_multiband_preset(preset="speech"):
    """Get multiband compression presets for different content types"""
    # Try to get from config first
    config_preset = config_manager.get_preset(preset, "multiband")
    if config_preset:
        return config_preset

    # Fallback to hardcoded defaults
    presets = {
        "speech": {
            "low_freq": 250,      # Hz - low/mid crossover
            "high_freq": 4000,    # Hz - mid/high crossover
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
        },
        "music": {
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
        },
        "vocal": {
            "low_freq": 300,
            "high_freq": 3000,
            "low_threshold": -18,
            "low_ratio": 2,
            "low_attack": 0.015,
            "low_release": 0.12,
            "low_makeup": 2,
            "mid_threshold": -16,
            "mid_ratio": 3.5,
            "mid_attack": 0.008,
            "mid_release": 0.09,
            "mid_makeup": 4,
            "high_threshold": -22,
            "high_ratio": 1.8,
            "high_attack": 0.003,
            "high_release": 0.06,
            "high_makeup": 1
        }
    }
    return presets.get(preset, presets["speech"])

def build_multiband_compressor(preset="speech", custom_freqs=None, custom_bands=None):
    """Build multiband compressor filter using FFmpeg's acrossor"""
    preset_data = get_multiband_preset(preset)

    # Use custom frequencies if provided
    low_freq = custom_freqs.get("low", preset_data["low_freq"]) if custom_freqs else preset_data["low_freq"]
    high_freq = custom_freqs.get("high", preset_data["high_freq"]) if custom_freqs else preset_data["high_freq"]

    # Build compressor settings for each band
    bands = []
    band_names = ["low", "mid", "high"]

    for band in band_names:
        if custom_bands and band in custom_bands:
            # Use custom band settings
            band_settings = custom_bands[band]
        else:
            # Use preset band settings
            band_settings = {
                "threshold": preset_data[f"{band}_threshold"],
                "ratio": preset_data[f"{band}_ratio"],
                "attack": preset_data[f"{band}_attack"],
                "release": preset_data[f"{band}_release"],
                "makeup": preset_data[f"{band}_makeup"]
            }

        # Create acompressor filter for this band
        comp_filter = f"acompressor=threshold={band_settings['threshold']}dB:ratio={band_settings['ratio']}:attack={band_settings['attack']}:release={band_settings['release']}:makeup={band_settings['makeup']}dB"
        bands.append(comp_filter)

    # Combine bands with frequency splitting using acrossor
    # acrossor splits into 3 bands: low, mid, high
    multiband_filter = f"acrossor=split={low_freq}:{high_freq},{bands[0]},{bands[1]},{bands[2]}"

    return multiband_filter

def get_channel_layout_info(layout_name):
    """Get channel count and layout string for different surround formats"""
    layouts = {
        "mono": {"channels": 1, "layout": "mono"},
        "stereo": {"channels": 2, "layout": "stereo"},
        "5.1": {"channels": 6, "layout": "5.1"},
        "7.1": {"channels": 8, "layout": "7.1"},
        "octagonal": {"channels": 8, "layout": "octagonal"},
        "hexadecagonal": {"channels": 16, "layout": "hexadecagonal"}
    }
    return layouts.get(layout_name, layouts["stereo"])

def build_channel_filters(channels, channel_layout=None, downmix=False, upmix=False):
    """Build channel mapping and mixing filters"""
    filters = []

    if channel_layout:
        layout_info = get_channel_layout_info(channel_layout)
        target_channels = layout_info["channels"]
        layout_string = layout_info["layout"]

        if downmix and channels > 2:
            # Downmix multichannel to stereo
            filters.append("pan=stereo|FL=FC+0.30*FL+0.30*BL|FR=FC+0.30*FR+0.30*BR")
        elif upmix and channels < target_channels:
            # Upmix to specified layout
            if target_channels == 6:  # 5.1
                filters.append("pan=5.1|FL=FL|FR=FR|FC=FC|LFE=0.5*FL+0.5*FR|BL=0.5*FL|BR=0.5*FR")
            elif target_channels == 8:  # 7.1
                filters.append("pan=7.1|FL=FL|FR=FR|FC=FC|LFE=0.3*FL+0.3*FR|BL=0.5*FL|BR=0.5*FR|SL=0.3*FL|SR=0.3*FR")
        else:
            # Set explicit channel layout
            filters.append(f"channelmap=channel_layout={layout_string}")

    return filters

def build_audio_filters(loudnorm_enabled=True, silence_trim_enabled=False, noise_gate_enabled=False,
                       silence_threshold=-50, silence_duration=0.5, gate_threshold=-35, gate_ratio=10, gate_attack=0.1,
                       compressor_enabled=False, compressor_preset="speech", comp_threshold=None, comp_ratio=None,
                       comp_attack=None, comp_release=None, comp_makeup=None,
                       multiband_enabled=False, multiband_preset="speech", custom_freqs=None, custom_bands=None,
                       ml_noise_reduction=False, channels=1, channel_layout=None, downmix=False, upmix=False):
    """Build chained audio filter string for FFmpeg"""
    filters = []

    # Channel mapping and mixing (first in chain)
    channel_filters = build_channel_filters(channels, channel_layout, downmix, upmix)
    filters.extend(channel_filters)

    # ML-based noise reduction (first in chain for best results)
    if ml_noise_reduction:
        # Use FFmpeg's arnndn filter with pre-trained model for noise reduction
        model_path = config_manager.get_model_path("arnndn_model")
        if model_path and os.path.exists(model_path):
            filters.append(f"arnndn=m='{model_path}'")
        else:
            logging.warning("ML noise reduction model not found. Skipping ML noise reduction.")
            logging.warning("To enable ML noise reduction:")
            logging.warning("  1. Download FFmpeg with arnndn support")
            logging.warning("  2. Obtain an arnndn model file (usually .mdl format)")
            logging.warning("  3. Set the model path: config_manager.set_model_path('arnndn_model', '/path/to/model.mdl')")
            logging.warning("  4. Or use --ml-noise-reduction flag again")

    # Multiband compression (first in chain for best results)
    if multiband_enabled:
        multiband_filter = build_multiband_compressor(multiband_preset, custom_freqs, custom_bands)
        filters.append(multiband_filter)

    # Dynamic range compression (skip if multiband is enabled)
    elif compressor_enabled:
        preset = get_compressor_preset(compressor_preset)
        threshold = comp_threshold if comp_threshold is not None else preset["threshold"]
        ratio = comp_ratio if comp_ratio is not None else preset["ratio"]
        attack = comp_attack if comp_attack is not None else preset["attack"]
        release = comp_release if comp_release is not None else preset["release"]
        makeup = comp_makeup if comp_makeup is not None else preset["makeup"]

        filters.append(f"acompressor=threshold={threshold}dB:ratio={ratio}:attack={attack}:release={release}:makeup={makeup}dB")

    # Loudness normalization (EBU R128)
    if loudnorm_enabled:
        filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")

    # Silence trimming (remove silence from start/end)
    if silence_trim_enabled:
        filters.append(f"silenceremove=start_threshold={silence_threshold}dB:start_duration={silence_duration}")

    # Noise gating (reduce background noise - skip if ML noise reduction is enabled)
    if noise_gate_enabled and not ml_noise_reduction:
        filters.append(f"agate=threshold={gate_threshold}dB:ratio={gate_ratio}:attack={gate_attack}")

    return ",".join(filters) if filters else None

def compress_audio(input_file, output_file, bitrate, filter_chain, output_format, channels=1, preserve_metadata=True, dry_run=False, preview_mode=False):
    format_info = get_format_defaults(output_format)
    if not format_info:
        logging.error(f"Unsupported output format: {output_format}")
        return False, 0, 0

    codec = format_info["codec"]
    ext = format_info["ext"]

    # Ensure output file has correct extension
    if not output_file.endswith(ext):
        output_file = os.path.splitext(output_file)[0] + ext

    # Build FFmpeg command
    cmd = ["ffmpeg", "-i", input_file]

    # Add metadata preservation if requested
    if preserve_metadata:
        cmd.extend(["-map_metadata", "0"])

    # Audio filter chain
    if filter_chain:
        cmd.extend(["-af", filter_chain])

    # Audio settings
    cmd.extend(["-ac", str(channels), "-ar", "44100"])

    # Preview mode: create short 10-second clip
    if preview_mode:
        cmd.extend(["-t", "10"])

    # Bitrate setting (skip for lossless)
    if output_format != "flac":
        cmd.extend(["-b:a", f"{bitrate}k"])

    cmd.extend(["-c:a", codec, output_file, "-y"])

    if dry_run:
        logging.info(f"Dry run - would execute: {' '.join(cmd)}")
        return True, 0, 0

    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        processing_time = time.time() - start_time

        # Get file sizes for statistics
        input_size = os.path.getsize(input_file)
        output_size = os.path.getsize(output_file)

        logging.info(f"Successfully compressed {input_file} to {output_file} in {processing_time:.2f}s")
        return True, input_size, output_size
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to compress {input_file}: {e.stderr}")
        logging.error("Common solutions:")
        logging.error("  - Check if input file exists and is readable")
        logging.error("  - Verify FFmpeg codec support: ffmpeg -codecs | grep <codec>")
        logging.error("  - Try a different output format or bitrate")
        logging.error("  - Ensure output directory is writable")
        return False, 0, 0

def process_files(input_dir, output_dirs, extensions, bitrates, filter_chain, output_format, channels=1, preserve_metadata=True, dry_run=False, parallel=False):
    files_to_process = []
    for file in os.listdir(input_dir):
        if file.lower().endswith(extensions):
            input_path = os.path.join(input_dir, file)
            filename, _ = os.path.splitext(file)
            for bitrate in bitrates:
                format_info = get_format_defaults(output_format)
                ext = format_info.get("ext", ".mp3")
                output_file = os.path.join(output_dirs[bitrate], f"{filename}{ext}")
                files_to_process.append((input_path, output_file, bitrate, filter_chain, output_format, channels, preserve_metadata, dry_run))

    if parallel and not dry_run:
        with multiprocessing.Pool() as pool:
            results = pool.starmap(compress_audio, files_to_process)
    else:
        results = [compress_audio(*args) for args in files_to_process]

    # Calculate statistics
    processed = 0
    failed = 0
    total_input_size = 0
    total_output_size = 0
    total_time = 0

    for success, input_size, output_size in results:
        if success:
            processed += 1
            total_input_size += input_size
            total_output_size += output_size
        else:
            failed += 1

    return processed, failed, total_input_size, total_output_size

def validate_inputs(args):
    """Validate input arguments and provide defaults"""
    # Validate input directory
    if not os.path.exists(args.input):
        logging.error(f"Input directory does not exist: {args.input}")
        logging.error("Please check the path and ensure the directory exists.")
        logging.error("Suggestions:")
        logging.error("  - Use an absolute path: /full/path/to/audio/files")
        logging.error("  - Check for typos in the directory name")
        logging.error("  - Ensure you have read permissions for the directory")
        logging.error("Example: python compress_audio.py -i /path/to/audio/files")
        sys.exit(1)

    # Validate output directory and create if needed
    try:
        os.makedirs(args.output, exist_ok=True)
    except OSError as e:
        logging.error(f"Cannot create output directory {args.output}: {e}")
        logging.error("Please check permissions or choose a different output directory.")
        logging.error("Suggestions:")
        logging.error("  - Check write permissions for the parent directory")
        logging.error("  - Try using sudo if on Linux/macOS")
        logging.error("  - Choose a different output path with write access")
        logging.error("Example: python compress_audio.py -o /path/to/output/directory")
        sys.exit(1)

    # Set default bitrates based on format and content type
    if not args.bitrates:
        format_defaults = get_format_defaults(args.format, args.content_type)
        args.bitrates = format_defaults.get(args.content_type, [128, 96])

    # Validate bitrates for lossless formats
    if args.format == "flac" and args.bitrates:
        logging.warning("Bitrates are ignored for lossless FLAC compression")
        logging.warning("FLAC is a lossless format - all files will be compressed without quality loss.")

    # Validate format
    supported_formats = ["mp3", "aac", "ogg", "opus", "flac"]
    if args.format not in supported_formats:
        logging.error(f"Unsupported format: {args.format}")
        logging.error(f"Supported formats: {', '.join(supported_formats)}")
        logging.error("Suggestions:")
        logging.error("  - Use 'opus' for best speech compression")
        logging.error("  - Use 'aac' for good compatibility")
        logging.error("  - Use 'mp3' for maximum compatibility")
        logging.error("  - Use 'flac' for lossless compression")
        logging.error("Example: python compress_audio.py -f mp3")
        sys.exit(1)

    # Validate content type
    supported_content_types = ["speech", "music"]
    if args.content_type not in supported_content_types:
        logging.error(f"Unsupported content type: {args.content_type}")
        logging.error(f"Supported content types: {', '.join(supported_content_types)}")
        logging.error("Suggestions:")
        logging.error("  - Use 'speech' for eLearning, podcasts, voice recordings")
        logging.error("  - Use 'music' for songs, soundtracks, complex audio")
        logging.error("Example: python compress_audio.py -t speech")
        sys.exit(1)

    # Validate compressor preset
    supported_presets = ["speech", "music", "broadcast", "gentle"]
    if hasattr(args, 'comp_preset') and args.comp_preset not in supported_presets:
        logging.error(f"Unsupported compressor preset: {args.comp_preset}")
        logging.error(f"Supported presets: {', '.join(supported_presets)}")
        logging.error("Suggestions:")
        logging.error("  - Use 'speech' for voice recordings and podcasts")
        logging.error("  - Use 'music' for songs and complex audio")
        logging.error("  - Use 'broadcast' for professional audio production")
        logging.error("  - Use 'gentle' for subtle compression")
        logging.error("Example: python compress_audio.py --comp-preset speech")
        sys.exit(1)

    # Validate multiband preset
    supported_mb_presets = ["speech", "music", "vocal"]
    if hasattr(args, 'mb_preset') and args.mb_preset not in supported_mb_presets:
        logging.error(f"Unsupported multiband preset: {args.mb_preset}")
        logging.error(f"Supported presets: {', '.join(supported_mb_presets)}")
        logging.error("Suggestions:")
        logging.error("  - Use 'speech' for voice recordings with frequency-specific processing")
        logging.error("  - Use 'music' for full-spectrum audio with complex dynamics")
        logging.error("  - Use 'vocal' for singing voice and vocal performances")
        logging.error("Example: python compress_audio.py --mb-preset speech")
        sys.exit(1)

    return args

def print_statistics(processed, failed, total_input_size, total_output_size, start_time):
    """Print compression statistics"""
    total_time = time.time() - start_time

    print(f"\n✅ Processing complete!")
    print(f"   Files processed: {processed}")
    if failed > 0:
        print(f"   Files failed: {failed}")
    print(f"   Total time: {total_time:.2f} seconds")

    if processed > 0 and total_input_size > 0:
        compression_ratio = (1 - total_output_size / total_input_size) * 100
        avg_time_per_file = total_time / processed
        print(f"   Average compression ratio: {compression_ratio:.1f}%")
        print(f"   Average time per file: {avg_time_per_file:.2f} seconds")
        print(f"   Total size reduction: {(total_input_size - total_output_size) / 1024 / 1024:.1f} MB")

def main():
    parser = argparse.ArgumentParser(description="Pure Sound - Professional Audio Processing Suite")
    parser.add_argument("-i", "--input", default=".", help="Input directory containing audio files (default: current directory)")
    parser.add_argument("-o", "--output", default=".", help="Output base directory (default: current directory)")
    parser.add_argument("-b", "--bitrates", nargs='+', type=int, help="Bitrates in kbps (uses format defaults if not specified)")
    parser.add_argument("-f", "--format", default="mp3", choices=["mp3", "aac", "ogg", "opus", "flac"], help="Output format (default: mp3)")
    parser.add_argument("-c", "--channels", type=int, default=1, help="Audio channels: 1=mono, 2=stereo, or surround layouts (default: 1)")
    parser.add_argument("--channel-layout", choices=["mono", "stereo", "5.1", "7.1", "octagonal", "hexadecagonal"], help="Channel layout for surround sound (auto-detected if not specified)")
    parser.add_argument("--downmix", action="store_true", help="Downmix multichannel audio to stereo")
    parser.add_argument("--upmix", action="store_true", help="Upmix mono/stereo to multichannel (requires channel-layout)")
    parser.add_argument("-t", "--content-type", default="speech", choices=["speech", "music"], help="Content type for bitrate defaults (default: speech)")

    # Audio processing options
    parser.add_argument("-n", "--no-normalize", action="store_true", help="Skip loudness normalization")
    parser.add_argument("--compressor", action="store_true", help="Enable dynamic range compression")
    parser.add_argument("--comp-preset", default="speech", choices=["speech", "music", "broadcast", "gentle"], help="Compressor preset (default: speech)")
    parser.add_argument("--comp-threshold", type=float, help="Compressor threshold in dB (uses preset default if not specified)")
    parser.add_argument("--comp-ratio", type=float, help="Compressor ratio (uses preset default if not specified)")
    parser.add_argument("--comp-attack", type=float, help="Compressor attack time in seconds (uses preset default if not specified)")
    parser.add_argument("--comp-release", type=float, help="Compressor release time in seconds (uses preset default if not specified)")
    parser.add_argument("--comp-makeup", type=float, help="Compressor makeup gain in dB (uses preset default if not specified)")
    parser.add_argument("--multiband", action="store_true", help="Enable multiband compression (overrides single-band compressor)")
    parser.add_argument("--mb-preset", default="speech", choices=["speech", "music", "vocal"], help="Multiband compressor preset (default: speech)")
    parser.add_argument("--mb-low-freq", type=int, help="Low/mid crossover frequency in Hz (uses preset default if not specified)")
    parser.add_argument("--mb-high-freq", type=int, help="Mid/high crossover frequency in Hz (uses preset default if not specified)")
    parser.add_argument("--ml-noise-reduction", action="store_true", help="Enable ML-based noise reduction (requires FFmpeg with arnndn models)")
    parser.add_argument("--silence-trim", action="store_true", help="Enable silence trimming from start/end")
    parser.add_argument("--silence-threshold", type=float, default=-50.0, help="Silence threshold in dB (default: -50)")
    parser.add_argument("--silence-duration", type=float, default=0.5, help="Minimum silence duration in seconds (default: 0.5)")
    parser.add_argument("--noise-gate", action="store_true", help="Enable noise gating")
    parser.add_argument("--gate-threshold", type=float, default=-35.0, help="Noise gate threshold in dB (default: -35)")
    parser.add_argument("--gate-ratio", type=float, default=10.0, help="Noise gate compression ratio (default: 10)")
    parser.add_argument("--gate-attack", type=float, default=0.1, help="Noise gate attack time in seconds (default: 0.1)")

    parser.add_argument("-m", "--no-metadata", action="store_true", help="Don't preserve metadata")
    parser.add_argument("-p", "--parallel", action="store_true", help="Enable parallel processing")
    parser.add_argument("-d", "--dry-run", action="store_true", help="Show commands without executing")
    parser.add_argument("--preview", action="store_true", help="Generate 10-second preview clips with filters applied")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--analyze", action="store_true", help="Analyze audio files and provide compression recommendations")
    parser.add_argument("--multi-stream", action="store_true", help="Create multiple output streams with different formats/bitrates")
    parser.add_argument("--streaming", action="store_true", help="Create adaptive bitrate streaming outputs")
    parser.add_argument("--job-queue", action="store_true", help="Use job queue for batch processing")
    parser.add_argument("--cloud-upload", action="store_true", help="Upload results to cloud storage")
    parser.add_argument("--offline-store", action="store_true", help="Store results in offline storage")
    parser.add_argument("--channel-split", action="store_true", help="Split audio into separate channel files")
    args = parser.parse_args()

    setup_logging(args.verbose)
    check_ffmpeg()

    # Analyze audio files if requested
    if args.analyze:
        print("🔍 Analyzing audio files...")
        extensions = (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus")
        analyzed_files = 0

        for file in os.listdir(args.input):
            if file.lower().endswith(extensions):
                file_path = os.path.join(args.input, file)
                print(f"\n📊 Analyzing: {file}")

                # Get quick stats
                quick_stats = audio_analyzer.get_quick_stats(file_path)
                if quick_stats:
                    print(f"   Codec: {quick_stats['codec']}")
                    print(f"   Sample Rate: {quick_stats['sample_rate']} Hz")
                    print(f"   Channels: {quick_stats['channels']}")
                    print(f"   Duration: {quick_stats['duration']:.1f} seconds")
                    print(f"   Bitrate: {quick_stats['bitrate_kbps']:.1f} kbps")
                    print(f"   Size: {quick_stats['size_mb']:.1f} MB")

                    # Get full analysis
                    analysis = audio_analyzer.analyze_file(file_path)
                    if analysis:
                        content = analysis["content_analysis"]
                        recommendations = analysis["recommendations"]

                        print(f"   Content Type: {content.get('content_type', 'unknown')}")
                        print(f"   Dynamic Range: {content.get('dynamic_range', 'unknown')}")
                        print(f"   Speech Probability: {content.get('speech_probability', 0):.2f}")
                        print(f"   Music Probability: {content.get('music_probability', 0):.2f}")

                        print("   💡 Recommendations:")
                        print(f"      Format: {recommendations.get('format', 'mp3')}")
                        print(f"      Bitrates: {', '.join(map(str, recommendations.get('bitrates', [128])))}")
                        print(f"      Enable Compression: {recommendations.get('enable_compression', False)}")
                        print(f"      Enable Loudness Norm: {recommendations.get('enable_loudnorm', True)}")

                        for reason in recommendations.get('reasoning', []):
                            print(f"      - {reason}")

                analyzed_files += 1
                if analyzed_files >= 5:  # Limit analysis to first 5 files
                    print("\n⚠️  Analysis limited to first 5 files. Use --input to analyze specific files.")
                    break

        if analyzed_files == 0:
            print("❌ No audio files found in the specified directory.")
        return

    # Handle special modes
    if args.multi_stream:
        # Create multiple output streams
        print("🎵 Creating multiple output streams...")
        outputs = multi_stream_processor.create_multiple_outputs(
            input_file=args.input,  # This would need to be a single file for multi-stream
            output_base=args.output,
            bitrates=args.bitrates,
            formats=[args.format],
            filter_chain=None,  # Would need to build filter chain
            channels=args.channels
        )
        print(f"Created {len(outputs)} output streams")
        return

    elif args.streaming:
        # Create adaptive streaming outputs
        print("📺 Creating adaptive streaming outputs...")
        streaming_info = multi_stream_processor.create_adaptive_streaming(
            input_file=args.input,  # Single file
            output_base=args.output,
            filter_chain=None,
            channels=args.channels
        )
        print(f"Created streaming outputs with master playlist: {streaming_info.get('master_playlist')}")
        return

    elif args.channel_split:
        # Split channels
        print("🎚️ Splitting audio channels...")
        channel_files = selective_channel_processor.create_channel_split_outputs(
            input_file=args.input,  # Single file
            output_base=args.output,
            channels=args.channels
        )
        print(f"Created {len(channel_files)} channel files")
        return

    elif args.offline_store:
        # Store results in offline storage
        print("💾 Storing results in offline storage...")
        stored_files = []

        for file in os.listdir(args.output):
            if file.lower().endswith(extensions):
                local_path = os.path.join(args.output, file)
                storage_key = f"compressed_audio/{file}"

                metadata = {
                    "compression_params": {
                        "format": args.format,
                        "bitrate": args.bitrates[0] if args.bitrates else None,
                        "channels": getattr(args, 'channels', 1),
                        "content_type": args.content_type
                    },
                    "original_size": os.path.getsize(local_path)
                }

                if storage_manager.store_file(local_path, storage_key, metadata):
                    stored_files.append(storage_key)
                    print(f"   ✅ Stored: {storage_key}")
                else:
                    print(f"   ❌ Failed to store: {file}")

        print(f"Successfully stored {len(stored_files)} files in offline storage")
        return

    elif args.job_queue:
        # Use job queue for batch processing
        print("📋 Using job queue for batch processing...")
        job_queue.start()

        # Submit jobs for all files
        extensions = (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus")
        submitted_jobs = []

        for file in os.listdir(args.input):
            if file.lower().endswith(extensions):
                input_path = os.path.join(args.input, file)
                filename, _ = os.path.splitext(file)

                for bitrate in args.bitrates:
                    output_file = os.path.join(args.output, f"optimised-{bitrate}kbps", f"{filename}_{bitrate}k.{args.format}")

                    from job_queue import CompressionJob

                    job = CompressionJob(
                        job_id=f"{filename}_{bitrate}",
                        input_file=input_path,
                        output_file=output_file,
                        bitrate=bitrate,
                        format=args.format,
                        channels=getattr(args, 'channels', 1),
                        preserve_metadata=not getattr(args, 'no_metadata', False)
                    )

                    job_id = job_queue.add_job(job)
                    submitted_jobs.append(job_id)

        print(f"Submitted {len(submitted_jobs)} jobs to queue")

        # Wait for completion
        while True:
            stats = job_queue.get_queue_stats()
            completed = stats['completed']
            total = stats['total']

            print(f"Progress: {completed}/{total} jobs completed")
            if completed == total:
                break
            time.sleep(2)

        job_queue.stop()
        return

    # Standard processing
    # Validate inputs and set defaults
    args = validate_inputs(args)

    # Allowed input extensions
    extensions = (".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".opus")

    # Build custom frequency settings for multiband
    custom_freqs = None
    if getattr(args, 'mb_low_freq', None) or getattr(args, 'mb_high_freq', None):
        custom_freqs = {}
        if getattr(args, 'mb_low_freq', None):
            custom_freqs["low"] = args.mb_low_freq
        if getattr(args, 'mb_high_freq', None):
            custom_freqs["high"] = args.mb_high_freq

    # Build audio filter chain
    filter_chain = build_audio_filters(
        loudnorm_enabled=not getattr(args, 'no_normalize', False),
        silence_trim_enabled=getattr(args, 'silence_trim', False),
        noise_gate_enabled=getattr(args, 'noise_gate', False),
        silence_threshold=getattr(args, 'silence_threshold', -50),
        silence_duration=getattr(args, 'silence_duration', 0.5),
        gate_threshold=getattr(args, 'gate_threshold', -35),
        gate_ratio=getattr(args, 'gate_ratio', 10),
        gate_attack=getattr(args, 'gate_attack', 0.1),
        compressor_enabled=getattr(args, 'compressor', False),
        compressor_preset=getattr(args, 'comp_preset', 'speech'),
        comp_threshold=getattr(args, 'comp_threshold', None),
        comp_ratio=getattr(args, 'comp_ratio', None),
        comp_attack=getattr(args, 'comp_attack', None),
        comp_release=getattr(args, 'comp_release', None),
        comp_makeup=getattr(args, 'comp_makeup', None),
        multiband_enabled=getattr(args, 'multiband', False),
        multiband_preset=getattr(args, 'mb_preset', 'speech'),
        custom_freqs=custom_freqs,
        custom_bands=None,  # Future enhancement for individual band control
        ml_noise_reduction=getattr(args, 'ml_noise_reduction', False),
        channels=getattr(args, 'channels', 1),
        channel_layout=getattr(args, 'channel_layout', None),
        downmix=getattr(args, 'downmix', False),
        upmix=getattr(args, 'upmix', False)
    )

    # Create output directories
    output_dirs = create_output_dirs(args.output, args.bitrates)

    start_time = time.time()
    processed, failed, total_input_size, total_output_size = process_files(
        args.input, output_dirs, extensions, args.bitrates, filter_chain,
        args.format, getattr(args, 'channels', 1), not getattr(args, 'no_metadata', False),
        getattr(args, 'dry_run', False), getattr(args, 'parallel', False)
    )

    # Generate preview clips if requested
    if args.preview and not args.dry_run and processed > 0:
        print(f"\n🎧 Generating preview clips...")
        preview_dir = os.path.join(args.output, "previews")
        os.makedirs(preview_dir, exist_ok=True)

        # Create preview for first file found
        for file in os.listdir(args.input):
            if file.lower().endswith(extensions):
                input_path = os.path.join(args.input, file)
                filename, _ = os.path.splitext(file)
                preview_file = os.path.join(preview_dir, f"{filename}_preview.{args.format}")

                # Use first bitrate for preview
                success, _, _ = compress_audio(
                    input_path, preview_file, args.bitrates[0], filter_chain,
                    args.format, args.channels, not args.no_metadata, False, True
                )
                if success:
                    print(f"   📼 Preview clip: {preview_file}")
                break

    if not args.dry_run:
        print_statistics(processed, failed, total_input_size, total_output_size, start_time)

        # Print output directories
        for bitrate in args.bitrates:
            dir_name = f"optimised-{bitrate}kbps"
            print(f"   - {bitrate} kbps in '{os.path.join(args.output, dir_name)}/'")
    else:
        print(f"\n📋 Dry run complete! Would process {processed} files.")

    # Show filter information
    if filter_chain:
        print(f"\n🎛️  Applied audio filters: {filter_chain}")
    else:
        print(f"\n🎛️  No audio filters applied")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--gui":
        try:
            from gui import main as gui_main
            gui_main()
        except (ImportError, ModuleNotFoundError) as e:
            print(f"GUI not available: {e}")
            print("Install tkinter to use the GUI:")
            print("  macOS: Tkinter comes with Python by default")
            print("  Ubuntu/Debian: sudo apt install python3-tk")
            print("  CentOS/RHEL: sudo yum install tkinter")
            print("  Windows: Tkinter comes with Python by default")
            print("  Or install from conda: conda install tk")
            print("\nFalling back to command-line interface...")
            main()
    else:
        main()
