# Pure Sound - Docker Setup Guide

This guide provides instructions for running the Pure Sound Audio Processing Suite using Docker containers.

## Quick Start

1. **Clone and Setup**
```bash
git clone <repository-url>
cd pure-sound
cp docker/.env.example docker/.env
```

2. **Start Services**
```bash
./scripts/start.sh start
```

3. **Check Status**
```bash
./scripts/start.sh status
```

## Prerequisites

- Docker (version 20.10 or higher)
- Docker Compose (version 1.29 or higher)
- At least 2GB of RAM available
- 4GB+ of disk space

## Directory Structure

```
pure-sound/
├── docker/                    # Docker configuration files
│   ├── Dockerfile           # Production Dockerfile
│   ├── Dockerfile.dev       # Development Dockerfile
│   ├── .env.example        # Environment variables template
│   └── monitoring/         # Monitoring and logging
├── scripts/                 # Management scripts
│   └── start.sh            # Main startup script
├── input/                  # Input audio files (mount point)
├── output/                 # Processed audio files (mount point)
├── models/                 # ML models (mount point)
├── config/                 # Configuration files (mount point)
├── logs/                   # Application logs (mount point)
├── .dockerignore           # Docker build ignore file
└── README-Docker.md       # This file
```

## Configuration

### Environment Variables

Copy the environment template and customize:

```bash
cp docker/.env.example docker/.env
```

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `COMPRESS_AUDIO_FORMAT` | Output audio format | `opus` |
| `COMPRESS_AUDIO_CONTENT_TYPE` | Audio content type | `speech` |
| `COMPRESS_AUDIO_CHANNELS` | Number of audio channels | `1` |
| `FFMPEG_THREADS` | FFmpeg processing threads | `4` |
| `LOG_LEVEL` | Application log level | `INFO` |
| `AWS_ACCESS_KEY_ID` | AWS S3 access key | *Optional* |
| `AWS_SECRET_ACCESS_KEY` | AWS S3 secret key | *Optional* |

### FFmpeg Configuration

The Docker image includes FFmpeg with support for:
- MP3 (libmp3lame)
- AAC (libfdk-aac)
- Opus (libopus)
- OGG (libvorbis)
- FLAC (lossless)

## Services Overview

### Core Services

1. **pure-sound**: Main audio processing application
   - Built on Python 3.13
   - Includes all dependencies
   - Health checks enabled
   - Non-root user for security

2. **redis**: Job queue backend
   - Redis 7 for job management
   - Persistent storage
   - Fast in-memory operations

3. **prometheus**: Metrics collection
   - Scrapes service metrics
   - Stores time-series data
   - Configured retention policies

4. **grafana**: Visualization and dashboards
   - Pre-configured dashboards
   - Real-time monitoring
   - Alert management

### Service Ports

| Service | Port | Purpose |
|---------|------|---------|
| pure-sound | 8080 | API/Interface |
| redis | 6379 | Database |
| prometheus | 9090 | Metrics |
| grafana | 3000 | Dashboard |

## Usage

### Command Line Interface (CLI)

```bash
# Access the application container
docker-compose exec pure-sound python compress_audio.py --help

# Process audio files
docker-compose exec pure-sound python compress_audio.py -i /app/input -f opus

# Analyze audio content
docker-compose exec pure-sound python compress_audio.py --analyze
```

### Graphical User Interface (GUI)

```bash
# Start GUI mode
docker-compose exec pure-sound python compress_audio.py --gui
```

### Development Environment

```bash
# Build development image
docker-compose -f docker-compose.dev.yml build

# Start development services
docker-compose -f docker-compose.dev.yml up -d

# Enter development shell
docker-compose -f docker-compose.dev.yml exec pure-sound bash

# Run tests
docker-compose -f docker-compose.dev.yml exec pure-sound pytest
```

## Management Scripts

The `scripts/start.sh` script provides comprehensive management:

```bash
# Start all services
./scripts/start.sh start

# Stop all services
./scripts/start.sh stop

# Restart services
./scripts/start.sh restart

# Show status
./scripts/start.sh status

# View logs
./scripts/start.sh logs

# Enter development shell
./scripts/start.sh dev

# Run tests
./scripts/start.sh test

# Build documentation
./scripts/start.sh docs

# Cleanup containers and images
./scripts/start.sh cleanup

# Show help
./scripts/start.sh help
```

## File Organization

### Input Files
Place audio files in the `input/` directory:
```
input/
├── podcast_episode1.wav
├── music_track1.mp3
├── interview_session1.flac
└── elearning_lesson1.m4a
```

### Output Structure
Processed files are organized by bitrate:
```
output/
├── optimised-24kbps/
│   ├── podcast_episode1.opus
│   └── interview_session1.opus
├── optimised-32kbps/
│   ├── podcast_episode1.opus
│   └── interview_session1.opus
└── optimised-48kbps/
    ├── podcast_episode1.opus
    └── interview_session1.opus
```

### Configuration Files

#### compress_audio_config.json
```json
{
  "model_paths": {
    "arnndn_model": "/app/models/bd.cnr.mdl",
    "custom_models_dir": "/app/models"
  },
  "presets": {
    "speech": {
      "compressor": {
        "threshold": -20,
        "ratio": 3,
        "attack": 0.01,
        "release": 0.1,
        "makeup": 6
      }
    }
  }
}
```

## Monitoring and Observability

### Prometheus Metrics

The application exposes metrics at:
- `http://localhost:9090` (Prometheus)
- `http://localhost:3000` (Grafana)

Key metrics tracked:
- Processing job counts
- Audio processing duration
- Memory usage
- Error rates
- Throughput metrics

### Grafana Dashboards

Pre-configured dashboards:
- **System Overview**: Service health and status
- **Processing Jobs**: Job creation and completion rates
- **Performance**: Processing time and throughput
- **Storage**: Memory and disk usage

## Production Deployment

### Security Considerations

1. **Non-root User**: Application runs as non-root user
2. **Environment Variables**: Sensitive data stored in environment
3. **Network Isolation**: Services in dedicated network
4. **Resource Limits**: Configure memory and CPU limits

### Resource Allocation

```yaml
# Add to docker-compose.yml
services:
  pure-sound:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
```

### Scaling

```bash
# Scale services
docker-compose up -d --scale pure-sound=3

# Horizontal scaling for load balancing
docker-compose up -d --scale pure-sound=3 redis
```

## Troubleshooting

### Common Issues

1. **FFmpeg Not Found**
   ```bash
   # Check FFmpeg installation
   docker-compose exec pure-sound ffmpeg -version
   ```

2. **Permission Errors**
   ```bash
   # Fix permissions
   sudo chown -R $USER:$USER input/ output/ models/
   ```

3. **Memory Issues**
   ```bash
   # Increase Docker memory limit
   # Docker Desktop > Preferences > Resources > Memory
   ```

4. **Port Conflicts**
   ```bash
   # Change ports in docker-compose.yml
   ports:
     - "8081:8080"  # Change host port
   ```

### Debug Mode

```bash
# Start with debug logging
docker-compose up -d
docker-compose logs -f pure-sound

# Interactive debugging
docker-compose exec pure-sound /bin/bash
```

## Backup and Recovery

### Configuration Backup

```bash
# Backup configuration
tar -czf config-backup-$(date +%Y%m%d).tar.gz config/ docker/

# Backup environment variables
docker.env docker/.env > backup.env
```

### Data Recovery

```bash
# Recover processed files
cp -r output/ /path/to/recovery/

# Recover configuration
cp compress_audio_config.json /path/to/recovery/
```

## Performance Optimization

### CPU Optimization

```bash
# Adjust CPU limits
docker-compose up -d --pure-sound cpus=4
```

### Memory Optimization

```bash
# Add memory limits
services:
  pure-sound:
    environment:
      - FFMPEG_THREADS=2  # Reduce FFmpeg threads
      - PYTHONOPTIMIZE=1  # Python optimization
```

### Network Optimization

```bash
# Use host network for better performance
services:
  pure-sound:
    network_mode: host
```

## Support

For issues and questions:
- Check the troubleshooting section above
- Review the logs: `./scripts/start.sh logs`
- Visit the [Pure Sound Documentation](docs/)

## License

This Docker setup is part of the Pure Sound Audio Processing Suite and is subject to the same license terms.