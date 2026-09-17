import subprocess
from pathlib import Path

input_file = Path("leg.mov")
output_file = input_file.with_suffix(".mp4")

subprocess.run(["ffmpeg", "-i", input_file, output_file], check=True)
input_file.unlink()
