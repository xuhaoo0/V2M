'''
将输入视频帧率统一为 30 FPS

方法：
使用 FFmpeg 对原视频进行编解码，将帧率统一为 30 FPS。
先输出到原视频同目录的临时文件，成功后再覆盖原视频。

args:
- input_video，例如{input_dir}/a/b/c.mp4
- device，例如0
'''

import argparse
import subprocess
import tempfile
import time
from fractions import Fraction
from pathlib import Path


TARGET_FPS = 30


def get_video_fps(input_video: Path) -> float:
    """读取原视频的 FPS。"""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(input_video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    try:
        fps = float(Fraction(result.stdout.strip()))
    except (ValueError, ZeroDivisionError) as error:
        raise RuntimeError(f"无法读取视频 FPS：{input_video}") from error

    if fps <= 0:
        raise RuntimeError(f"无法读取视频 FPS：{input_video}")

    return fps


def normalize_video_fps(
    input_video: Path,
    device: int,
) -> None:
    """使用 NVDEC + NVENC 将视频帧率统一为 30 FPS，并覆盖原视频。"""
    temporary_dir = tempfile.TemporaryDirectory(
        prefix=f".{input_video.stem}-30fps-",
        dir=input_video.parent,
    )
    temporary_video = Path(temporary_dir.name) / input_video.name

    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-hwaccel", "cuda", "-hwaccel_device", str(device),
        "-i", str(input_video),
        "-map", "0:v:0", "-map", "0:a?",
        "-vf", f"fps={TARGET_FPS}",
        "-c:v", "h264_nvenc", "-gpu", str(device), "-preset", "p1",
        "-rc", "vbr", "-cq", "23", "-b:v", "0",
        "-c:a", "copy", "-movflags", "+faststart",
        str(temporary_video),
    ]

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        temporary_video.replace(input_video)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"FFmpeg 统一视频帧率失败：\n{error.stderr.strip()}"
        ) from error
    finally:
        temporary_dir.cleanup()


def parse_cli_paths(
    input_video: Path,
    device: int,
) -> tuple[Path, int]:
    """从命令行读取参数，未传入的参数沿用测试值。"""
    parser = argparse.ArgumentParser(description="将输入视频帧率统一为 30 FPS")

    parser.add_argument(
        "--input_video",
        "--input-video",
        dest="input_video",
        type=Path,
        default=input_video,
        help="需要统一帧率的视频路径",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=device,
        help="GPU 编号，例如 0 或 1",
    )

    args = parser.parse_args()

    return args.input_video, args.device


if __name__ == "__main__":
    start_time = time.perf_counter()

    # 【用于测试】
    input_video = Path("origin_data/test_fix/xk.mp4")
    device = 0

    # 从外部读取
    input_video, device = parse_cli_paths(
        input_video,
        device,
    )

    if not input_video.is_file():
        raise FileNotFoundError(f"输入视频不存在：{input_video}")

    source_fps = get_video_fps(input_video)

    print(f"开始处理：{input_video}")
    print(f"原视频 FPS：{source_fps:.2f}")
    print(f"目标 FPS：{TARGET_FPS}")
    print(f"将覆盖原视频：{input_video}")
    print(f"使用 GPU：{device}")

    if abs(source_fps - TARGET_FPS) < 1e-6:
        print("原视频已经是 30 FPS，无需重新编码")
    else:
        normalize_video_fps(
            input_video,
            device,
        )

    print()
    print(f"处理完成：{input_video}")

    elapsed_seconds = round(time.perf_counter() - start_time)
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"norm_30fps运行时间：{hours}时{minutes}分{seconds}秒")
