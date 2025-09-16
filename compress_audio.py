import os
import subprocess

# Target folders
output_dirs = {
    "128k": "optimised-128kbps",
    "96k": "optimised-96kbps"
}

# Create output folders if not exist
for d in output_dirs.values():
    os.makedirs(d, exist_ok=True)

# Allowed input extensions
extensions = (".wav", ".mp3", ".m4a")

# Loudnorm filter (EBU R128 standard, good for eLearning)
loudnorm_filter = "loudnorm=I=-16:TP=-1.5:LRA=11"

# Iterate over files
for file in os.listdir("."):
    if file.lower().endswith(extensions):
        filename, _ = os.path.splitext(file)
        print(f"🎧 Processing: {file}")

        # 128 kbps version
        out_128 = os.path.join(output_dirs["128k"], f"{filename}.mp3")
        subprocess.run([
            "ffmpeg", "-i", file,
            "-af", loudnorm_filter,
            "-ac", "1", "-ar", "44100", "-b:a", "128k",
            out_128, "-y"
        ])

        # 96 kbps version
        out_96 = os.path.join(output_dirs["96k"], f"{filename}.mp3")
        subprocess.run([
            "ffmpeg", "-i", file,
            "-af", loudnorm_filter,
            "-ac", "1", "-ar", "44100", "-b:a", "96k",
            out_96, "-y"
        ])

print("✅ Done! All files compressed and normalised.")
print("   - 128 kbps in 'optimised-128kbps/'")
print("   - 96 kbps in 'optimised-96kbps/'")
