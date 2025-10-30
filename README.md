# Pure Sound - Professional Audio Processing Suite

## Overview

Pure Sound is a comprehensive Python-based audio processing suite that leverages FFmpeg for professional-grade audio compression and optimization. It features advanced audio analysis, intelligent parameter suggestions, distributed processing capabilities, and extensive customization options, making it ideal for eLearning content creation, podcasting, broadcasting, and general audio optimization workflows.

## Features

### Core Audio Processing
- **Multiple Codecs**: MP3, AAC, OGG, Opus, and lossless FLAC compression
- **Smart Bitrate Defaults**: Content-aware defaults (speech vs music) with Opus optimized for low-bitrate speech
- **Parallel Processing**: Multiprocessing support for faster batch compression
- **Audio Channel Control**: Mono/stereo conversion with surround sound support
- **Loudness Normalization**: EBU R128 standard with optional disable
- **Metadata Preservation**: Optional metadata copying during compression
- **Dry-Run Mode**: Preview compression commands without execution
- **Comprehensive Statistics**: File sizes, compression ratios, processing times
- **Robust Error Handling**: Input validation, FFmpeg error checking, filesystem safety with actionable suggestions

### User Interfaces
- **Flexible CLI**: Extensive command-line options for all features
- **Graphical User Interface**: Visual parameter adjustment with real-time preview

### Advanced Features
- **Audio Analysis Engine**: Automatic content detection and compression recommendations
- **Job Queue System**: Background batch processing with priority management
- **Cloud Integration**: Distributed processing across multiple nodes with AWS S3 support
- **Multiple Output Streams**: Create various formats and bitrates simultaneously
- **Selective Channel Processing**: Apply different filters per audio channel
- **Offline Storage**: Local file storage with metadata indexing
- **Configuration Management**: JSON-based presets and model path management

## Requirements

### System Requirements
- Python 3.8 or higher (tested with Python 3.13)
- **Primary Multimedia Backend**: Choose one of the following:
  - **FFmpeg** (Recommended): Comprehensive audio/video processing framework
    - macOS: `brew install ffmpeg`
    - Ubuntu/Debian: `sudo apt install ffmpeg`
    - Windows: Download from https://ffmpeg.org/download.html
    - Opus codec: Usually included with FFmpeg, but verify with `ffmpeg -codecs | grep opus`
  - **GStreamer**: Alternative multimedia framework with Python bindings
    - Ubuntu/Debian: `sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good python3-gst-1.0`
    - macOS: `brew install gstreamer gst-plugins-base gst-plugins-good gst-python`
    - Windows: Download from https://gstreamer.freedesktop.org/download/

### Python Dependencies

#### Core Dependencies (Required)
```bash
pip install -r requirements.txt
```

#### Optional Dependencies
- **NumPy**: `pip install numpy` (for advanced audio analysis and signal processing)
- **SciPy**: `pip install scipy` (alternative to NumPy with additional scientific algorithms)
- **GUI Frameworks** (choose one):
  - **Tkinter**: Usually included with Python (simple GUI interface)
  - **PyQt6/PySide6**: `pip install PyQt6` or `pip install PySide6` (feature-rich cross-platform GUI)
- **Cloud Integration**: `pip install boto3` (for AWS S3 cloud storage)

## Usage

### Command Line Interface
```bash
python compress_audio.py [OPTIONS]
```

### Graphical User Interface
```bash
python compress_audio.py --gui
```

### Command Line Options

#### Basic Options
- `-i, --input DIR`: Input directory containing audio files (default: current directory)
- `-o, --output DIR`: Output base directory (default: current directory)
- `-b, --bitrates BITRATE [BITRATE ...]`: Bitrates in kbps (uses format/content defaults if not specified)
- `-f, --format {mp3,aac,ogg,opus,flac}`: Output format (default: mp3)
- `-c, --channels CHANNELS`: Audio channels: 1=mono, 2=stereo, or higher numbers for surround (default: 1)
- `--channel-layout {mono,stereo,5.1,7.1,octagonal,hexadecagonal}`: Explicit channel layout specification
- `--downmix`: Downmix multichannel audio to stereo
- `--upmix`: Upmix mono/stereo to multichannel (requires --channel-layout)
- `-t, --content-type {speech,music}`: Content type for bitrate defaults (default: speech)

#### Audio Processing Filters
- `-n, --no-normalize`: Skip loudness normalization
- `--compressor`: Enable dynamic range compression
- `--comp-preset {speech,music,broadcast,gentle}`: Compressor preset (default: speech)
- `--comp-threshold FLOAT`: Compressor threshold in dB (uses preset default if not specified)
- `--comp-ratio FLOAT`: Compressor ratio (uses preset default if not specified)
- `--comp-attack FLOAT`: Compressor attack time in seconds (uses preset default if not specified)
- `--comp-release FLOAT`: Compressor release time in seconds (uses preset default if not specified)
- `--comp-makeup FLOAT`: Compressor makeup gain in dB (uses preset default if not specified)
- `--multiband`: Enable multiband compression (professional mastering-grade)
- `--mb-preset {speech,music,vocal}`: Multiband compressor preset (default: speech)
- `--mb-low-freq INT`: Low/mid crossover frequency in Hz (uses preset default if not specified)
- `--mb-high-freq INT`: Mid/high crossover frequency in Hz (uses preset default if not specified)
- `--ml-noise-reduction`: Enable ML-based noise reduction (requires FFmpeg with arnndn models)
- `--silence-trim`: Enable silence trimming from start/end
- `--silence-threshold FLOAT`: Silence threshold in dB (default: -50)
- `--silence-duration FLOAT`: Minimum silence duration in seconds (default: 0.5)
- `--noise-gate`: Enable noise gating
- `--gate-threshold FLOAT`: Noise gate threshold in dB (default: -35)
- `--gate-ratio FLOAT`: Noise gate compression ratio (default: 10)
- `--gate-attack FLOAT`: Noise gate attack time in seconds (default: 0.1)

#### Advanced Options
- `--analyze`: Analyze audio files and provide compression recommendations
- `--multi-stream`: Create multiple output streams with different formats/bitrates
- `--streaming`: Create adaptive bitrate streaming outputs (HLS/DASH)
- `--job-queue`: Use job queue for batch processing with priority management
- `--cloud-upload`: Upload results to cloud storage (AWS S3)
- `--offline-store`: Store results in offline storage with metadata indexing
- `--channel-split`: Split audio into separate channel files

#### Other Options
- `-m, --no-metadata`: Don't preserve metadata
- `-p, --parallel`: Enable parallel processing
- `-d, --dry-run`: Show commands without executing
- `--preview`: Generate 10-second preview clips with filters applied
- `-v, --verbose`: Enable verbose logging
- `--gui`: Launch graphical user interface

## Examples

### Basic Usage

#### Speech Content (Opus Recommended)
```bash
python compress_audio.py -f opus
# Uses Opus with speech-optimized bitrates: 24, 32, 48 kbps
```

#### Music Content with AAC
```bash
python compress_audio.py -f aac -t music
# Uses AAC with music-optimized bitrates: 96, 128, 192 kbps
```

#### Custom Bitrates and Parallel Processing
```bash
python compress_audio.py -i /path/to/audio -b 64 128 -p -v
```

#### Lossless Compression
```bash
python compress_audio.py -f flac -c 2
# FLAC compression ignores bitrates, preserves stereo
```

### Advanced Usage

#### Dry Run to Preview
```bash
python compress_audio.py -d -f opus -b 32
```

#### Audio Analysis and Recommendations
```bash
python compress_audio.py --analyze
# Analyzes audio files and provides intelligent compression suggestions
```

#### Audio Processing with Filters
```bash
# Clean up speech recordings with silence trimming and noise gating
python compress_audio.py \
  -f opus \
  -t speech \
  --silence-trim \
  --silence-threshold -45 \
  --noise-gate \
  --gate-threshold -40

# Preview filter effects before processing
python compress_audio.py -f opus --silence-trim --noise-gate --preview

# Full custom configuration with all filters
python compress_audio.py \
  -i ./input_audio \
  -o ./compressed_output \
  -f opus \
  -b 24 32 48 \
  -c 1 \
  -t speech \
  --silence-trim \
  --silence-threshold -45 \
  --silence-duration 0.3 \
  --noise-gate \
  --gate-threshold -40 \
  --gate-ratio 8 \
  --gate-attack 0.05 \
  -p \
  -v
```

### Advanced Features

#### Batch Processing with Job Queue
```bash
python compress_audio.py --job-queue -i /path/to/audio -f opus -p
# Processes files in background with queue management
```

#### Multiple Output Streams
```bash
python compress_audio.py --multi-stream -i input.wav -o output/ -b 64 128 256
# Creates multiple bitrate versions simultaneously
```

#### Cloud Integration
```bash
# Set AWS credentials and bucket
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_BUCKET_NAME=your_bucket

python compress_audio.py -f opus --cloud-upload
# Processes and uploads results to cloud storage
```

#### Offline Storage
```bash
python compress_audio.py -f opus --offline-store
# Stores compressed files with metadata indexing
```

#### Channel-Specific Processing
```bash
python compress_audio.py --channel-split -i stereo.wav -o channels/
# Splits stereo file into separate left/right channel files
```

#### Graphical Interface
```bash
python compress_audio.py --gui
# Launches visual parameter adjustment interface
```

## Architecture

Pure Sound consists of several modular components with modern design patterns:

### Core Architecture Patterns

#### Event-Driven Communication
- **Centralized Event Bus** (`events.py`): All modules communicate through a pub-sub event system
- **Loose Coupling**: Modules don't need direct dependencies on each other
- **Asynchronous Processing**: Events are processed asynchronously for better performance
- **Event History**: Full event history for debugging and monitoring

#### Dependency Injection
- **Service Container** (`di_container.py`): Centralized dependency management
- **Automatic Injection**: Dependencies are resolved automatically based on type hints
- **Service Locator**: Global service registry for easy access
- **Lifecycle Management**: Services can be initialized and shut down properly

#### Standardized Interfaces
- **Abstract Base Classes** (`interfaces.py`): Common interfaces for audio processors, storage providers, etc.
- **Protocol-Based Design**: Type-safe interfaces using Python protocols
- **Data Transfer Objects**: Standardized data structures for module communication

### Module Components

#### Core Modules
- **Core Engine** (`compress_audio.py`): Main compression logic and CLI interface
- **Configuration Manager** (`config.py`): JSON-based settings and preset management
- **Audio Analyzer** (`audio_analysis.py`): Content detection and recommendation engine

#### Processing Modules
- **Job Queue** (`job_queue.py`): Background processing with priority management
- **Cloud Integration** (`cloud_integration.py`): Distributed processing and storage
- **Multi-Stream Processor** (`multi_stream.py`): Multiple output format generation
- **Resource Pool** (`resource_pool.py`): Connection and resource management

#### User Interface
- **GUI Interface** (`gui.py`): Visual parameter adjustment and monitoring
- **Presets Manager** (`presets.py`): Workflow presets and custom workflows

#### Testing & Infrastructure
- **Test Suite** (`test_new_system.py`, `test_compress_audio.py`): Comprehensive unit testing
- **Dependency Injection** (`di_container.py`): Service management and injection
- **Event System** (`events.py`): Cross-module communication

### Module Interactions

The system is designed with clear separation of concerns and standardized communication:

1. **compress_audio.py**: Main CLI interface and FFmpeg orchestration
2. **gui.py**: Tkinter-based graphical user interface
3. **job_queue.py**: Background job processing and queue management
4. **audio_analysis.py**: Audio file analysis and recommendations
5. **config.py**: Configuration management and persistence
6. **cloud_integration.py**: Cloud storage and distributed processing
7. **multi_stream.py**: Multiple output stream generation
8. **presets.py**: Workflow management and presets

All modules communicate through the event system and use dependency injection for service resolution.

## Code Examples

### Events System
```python
from events import event_bus, publish_event

# Publish an event
publish_event("file.processed", {"file": "audio.wav", "size": 1024}, "processor")

# Subscribe to events
subscription_id = event_bus.subscribe("file.*", callback_function)
```

### Dependency Injection
```python
from di_container import di_container, get_service

# Register services
di_container.register_singleton(AudioProcessor)
di_container.register_transient(TempFileManager)

# Resolve services
processor = get_service(AudioProcessor)
```

### Standardized Interfaces
```python
from interfaces import AudioProcessor, AudioFileInfo, CompressionJob

class MyProcessor(AudioProcessor):
    def process_file(self, input_path: str, output_path: str, **kwargs) -> bool:
        # Implementation here
        pass
```

### Workflow Presets
```python
from presets import preset_manager

# Get available presets
presets = preset_manager.get_all_presets()
print(f"Available presets: {[p.name for p in presets]}")

# Apply a preset
podcast_preset = preset_manager.get_preset("podcast_optimization")
print(f"Podcast preset format: {podcast_preset.format}")
print(f"Podcast preset bitrates: {podcast_preset.bitrates}")
```

## Input/Output

### Supported Formats
- **Input Formats**: WAV, MP3, M4A, FLAC, AAC, OGG, Opus
- **Output Formats**: MP3, AAC, OGG, Opus, FLAC (lossless)
- **Output Structure**: Creates `optimised-{bitrate}kbps/` subdirectories
- **Naming**: Maintains original filenames with appropriate extensions
- **Safety**: Never modifies original files
- **Metadata**: Preserves audio metadata and adds compression parameters

### Configuration
- **Config File**: `compress_audio_config.json` (auto-created)
- **Model Paths**: Configurable paths for ML models and presets
- **Presets**: Customizable compression presets for different content types
- **Defaults**: User-configurable default settings
- **Workflows**: Custom workflow definitions with step-by-step processing

Codec Recommendations

| Content Type | Recommended Codec | Default Bitrates | Notes |
|-------------|-------------------|------------------|--------|
| Speech (eLearning) | Opus | 24, 32, 48 kbps | Excellent compression efficiency |
| Speech (General) | AAC | 48, 64, 96 kbps | Good compatibility |
| Music | AAC/MP3 | 96, 128, 192 kbps | Better quality for complex audio |
| Archival | FLAC | N/A (lossless) | Perfect quality preservation |
| Streaming | AAC | 64, 128, 256 kbps | Adaptive bitrate support |

## Advanced Features

### Audio Processing
- **Audio Analysis**: Automatic content type detection (speech/music)
- **Smart Recommendations**: Bitrate and format suggestions based on analysis
- **Dynamic Range Compression**: Single-band and multiband compression presets
- **Noise Reduction**: ML-based noise reduction with FFmpeg arnndn support
- **Silence Processing**: Automatic silence trimming and noise gating
- **Loudness Normalization**: EBU R128 standard compliance

### Workflow Management
- **Preset Workflows**: 8 built-in presets for common audio processing tasks
- **Custom Workflows**: User-defined multi-step processing pipelines
- **Workflow Engine**: Step-by-step execution with dependency management
- **Parameter Suggestions**: AI-powered parameter recommendations

### Processing Infrastructure
- **Job Queue**: Background processing with progress tracking and persistence
- **Cloud Processing**: Distributed compression across multiple nodes
- **Multi-Stream**: Simultaneous output in multiple formats/bitrate
- **Channel Processing**: Selective filter application per audio channel
- **Resource Pooling**: Connection and resource management for performance

### Storage & Distribution
- **Offline Storage**: Local storage with metadata indexing and search
- **Cloud Integration**: AWS S3 support with automatic failover
- **Streaming Support**: HLS/DASH adaptive bitrate streaming
- **Metadata Management**: Comprehensive metadata preservation and indexing

### User Interface
- **GUI Interface**: Visual parameter adjustment with real-time preview
- **Command Line**: Extensive CLI options for automation and scripting
- **Progress Monitoring**: Real-time progress tracking and statistics

## Notes

### Audio Processing Tips
- **Opus** provides superior quality at low bitrates, especially for speech content
- **Mono conversion** significantly reduces file sizes for voice content (podcasts, eLearning)
- **Parallel processing** speeds up batch operations but uses more CPU resources
- **Loudness normalization** ensures consistent perceived volume across files
- **Dry-run mode** helps verify settings before large batch operations

### System Requirements
- Audio analysis requires NumPy for advanced features
- Cloud features require AWS credentials and boto3 library
- Job queue persists across sessions for reliability
- ML noise reduction requires FFmpeg with arnndn model support

### Performance Considerations
- Use parallel processing for large batch operations
- Enable job queue for long-running background processing
- Cloud processing distributes workload across multiple nodes
- Resource pooling optimizes connection reuse and memory usage

## Getting Started

### Installation

1. **Install Multimedia Backend:**
    ```bash
    # Option A: FFmpeg (Recommended)
    brew install ffmpeg                    # macOS
    sudo apt install ffmpeg               # Ubuntu/Debian

    # Option B: GStreamer (Alternative)
    brew install gstreamer gst-plugins-base gst-plugins-good gst-python  # macOS
    sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-base \
                     gstreamer1.0-plugins-good python3-gst-1.0            # Ubuntu
    ```

2. **Install Python Dependencies:**
    ```bash
    # Basic functionality
    pip install -r requirements.txt

    # Full feature set (recommended)
    pip install numpy scipy boto3 PyQt6
    ```

### Quick Start

1. **Launch GUI (requires GUI framework):**
    ```bash
    python compress_audio.py --gui
    ```

2. **Command line analysis:**
    ```bash
    python compress_audio.py --analyze
    ```

3. **Basic compression:**
    ```bash
    python compress_audio.py -f opus
    ```

### First Steps

1. **Test Installation:**
    ```bash
    python compress_audio.py --help
    ```

2. **Analyze Your Audio:**
    ```bash
    python compress_audio.py --analyze -i /path/to/audio/files
    ```

3. **Try Different Presets:**
    ```bash
    # Speech content (podcasts, eLearning)
    python compress_audio.py -f opus -t speech

    # Music content
    python compress_audio.py -f aac -t music

    # Lossless archival
    python compress_audio.py -f flac
    ```

## Key Features

### 🎵 Professional Audio Processing
- Industry-standard compression with FFmpeg/GStreamer backends
- Multiple codec support: MP3, AAC, OGG, Opus, FLAC
- Advanced audio filters: compression, normalization, noise reduction
- Content-aware bitrate optimization

### 🧠 AI-Powered Analysis
- Automatic content detection (speech vs music)
- Intelligent parameter suggestions based on analysis
- Dynamic range analysis and compression recommendations
- ML-based noise reduction support

### ☁️ Cloud Integration
- Distributed processing with AWS S3 support
- Automatic failover between cloud and local storage
- Background upload/download with progress tracking
- Connection pooling for optimal performance

### 📋 Job Queue Management
- Background processing with priority control
- Persistent job queue across sessions
- Rate limiting and resource management
- Real-time progress monitoring

### 🎛️ User Interfaces
- Intuitive GUI for parameter adjustment (Tkinter/PyQt)
- Extensive command-line options for automation
- Real-time preview and analysis
- Workflow preset management

### 🔄 Advanced Processing
- Multi-stream output generation
- Selective channel processing per audio channel
- Adaptive bitrate streaming (HLS/DASH)
- Custom workflow pipelines

### 💾 Storage & Metadata
- Offline storage with metadata indexing
- Comprehensive metadata preservation
- Search and organization capabilities
- Automatic cleanup and maintenance

### ⚙️ Configuration Management
- Customizable presets and settings
- JSON-based configuration with validation
- Model path management for ML features
- User-defined workflow templates

### 🔧 Flexible Architecture
- Choose between FFmpeg or GStreamer engines
- Modular design with dependency injection
- Event-driven communication between modules
- Extensible plugin architecture

License
Pure Sound is released under the MIT License. See the LICENSE file for more details.

Contact & Support
For questions, issues, or contributions:
- **GitHub Issues**: [Create an issue](https://github.com/Dark-stream/pure-sound/issues) for bug reports and feature requests
- **GitHub Discussions**: [Start a discussion](https://github.com/Dark-stream/pure-sound/discussions) for questions and general support
- **Documentation**: Check the [Wiki](https://github.com/Dark-stream/pure-sound/wiki) for detailed guides and troubleshooting
- **Security Issues**: Email security@puresound.dev for security-related concerns

For inline help: `python compress_audio.py --help`
