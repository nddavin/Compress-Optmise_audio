# Audio Compression Script

Overview
`compress_audio.py` is a simple script designed to compress audio files to a specified bitrate. This can be useful for reducing file size while maintaining a reasonable level of audio quality.

Features

- Compresses audio files to a specified bitrate.
- Supports multiple audio formats (e.g., WAV, MP3).
- Can handle single files or directories containing audio files.
Requirements

- Python 3.6 or higher
- `pydub` library: Install using `pip install pydub`.
- `ffmpeg` or `libav` (required by `pydub` for audio processing): Ensure it is installed and added to your system's PATH.

Usage
Compress a Single File

```bash
python compress_audio.py -i <input_file> -b <bitrate>

* <input_file>: Path to the audio file you want to compress.
* <bitrate>: Desired bitrate in kbps (e.g., 64, 128, 256).
Compress All Files in a Directory
python compress_audio.py -d <directory> -b <bitrate>

* <directory>: Path to the directory containing audio files you want to compress.
* <bitrate>: Desired bitrate in kbps (e.g., 64, 128, 256).
Example
python compress_audio.py -i "path/to/your/audio/file.mp3" -b 64

This command will compress the specified MP3 file to a 64 kbps bitrate.
Output
The compressed audio files will be saved in the same directory as the original files, with a _compressed suffix added to the filename.
Notes
* The script uses pydub to handle audio processing. Ensure that ffmpeg or libav is installed on your system and properly configured.
* The script does not overwrite the original files; it creates new compressed files.
License
This script is released under the MIT License. See the LICENSE file for more details.
Contact
For any questions or issues, feel free to contact the maintainer:
* Email: ndavindouglas@gmail.com
* GitHub: Dark-stream
