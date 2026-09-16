import subprocess
from pathlib import Path

input_video = Path("xxxtestleg.mov")
output_video = input_video.with_suffix(".mp4")

subprocess.run([
    "ffmpeg", "-y",
    "-i", input_video,
    "-vf", "fps=30",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "-an",
    output_video,
], check=True)

if not output_video.is_file() or output_video.stat().st_size == 0:
    raise RuntimeError(f"转换完成但输出文件无效：{output_video}")

input_video.unlink()
print(f"输出文件：{output_video}")
print(f"已删除原文件：{input_video}")
