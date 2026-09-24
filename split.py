'''
按画面突变切分一条视频
【需要安装包：pip install git+https://github.com/UVA-Computer-Vision-Lab/OmniShotCut.git】

方法：
原视频、OmniShotCut检测得到切片的帧号，例如：[[0, 301], [301, 361], ..., [1273, 1404]]
对原视频编解码得到各个切片视频（这一步要求尽量使用gpu、速度要快）、过滤短切片

args:
- input_video，例如{input_dir}/a/b/c.mp4
- input_dir，用于替换路径
- output_dir，把所有输出放在这下面，例如{output_dir}/a/b/c/c-001/c-001.mp4
- device，例如0
'''

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import omnishotcut
import torch


MODEL_PATH = Path("checkpoints/omnishotcut/OmniShotCut_ckpt.pth")


def get_video_output_dir(
    input_video: Path, input_dir: Path, output_dir: Path
) -> Path:
    """根据输入视频的相对路径生成输出目录。"""
    relative_path = input_video.relative_to(input_dir)
    return output_dir / relative_path.parent / relative_path.stem


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


def get_video_frame_count(input_video: Path) -> int:
    """读取视频帧数。"""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(input_video),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def copy_source_json(
    input_video: Path,
    video_output_dir: Path,
) -> Path:
    """把原视频目录下的 JSON 改为视频同名后复制到输出目录。"""
    json_files = sorted(input_video.parent.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(
            f"未找到原视频目录下的 JSON：{input_video.parent}"
        )
    if len(json_files) > 1:
        raise RuntimeError(
            f"原视频目录下存在多个 JSON，无法确定要复制的文件："
            f"{', '.join(map(str, json_files))}"
        )

    video_output_dir.mkdir(parents=True, exist_ok=True)
    output_json = video_output_dir / f"{input_video.stem}.json"
    shutil.copy2(json_files[0], output_json)
    return output_json


def normalize_ranges(ranges) -> list[tuple[int, int]]:
    """
    把 OmniShotCut 的区间转换为内部使用的左闭右开区间。

    OmniShotCut 输出例如：
    [[0, 301], [301, 361], ..., [1273, 1404]]

    所有区间统一按左闭右开处理，包括最后一个区间。
    """
    raw_ranges = [tuple(map(int, frame_range)) for frame_range in ranges]
    normalized_ranges = []

    for frame_range in raw_ranges:
        if len(frame_range) != 2:
            raise ValueError(f"无效的切片区间：{frame_range}")

        start_frame, end_frame = frame_range

        if start_frame < 0 or end_frame <= start_frame:
            raise ValueError(f"无效的切片区间：{frame_range}")

        normalized_ranges.append((start_frame, end_frame))

    return normalized_ranges


def detect_ranges(
    input_video: Path,
    device: int,
) -> tuple[list[tuple[int, int]], float, int]:
    """使用 OmniShotCut 检测原视频切片。"""
    fps = get_video_fps(input_video)
    frame_count = get_video_frame_count(input_video)

    torch.cuda.set_device(device)
    model = omnishotcut.load(str(MODEL_PATH))
    ranges = model.inference(
        str(input_video),
        mode="clean_shot",
    )

    normalized_ranges = normalize_ranges(ranges)

    if not normalized_ranges:
        raise RuntimeError("OmniShotCut 未返回任何切片区间")

    return normalized_ranges, fps, frame_count


def split_video(
    input_video: Path,
    video_output_dir: Path,
    ranges: list[tuple[int, int]],
    min_clip_frames: int,
    device: int,
) -> list[Path]:
    """使用 NVDEC + NVENC 按帧范围切分视频。"""
    video_output_dir.mkdir(parents=True, exist_ok=True)

    if not ranges:
        return []

    frame_count = get_video_frame_count(input_video)
    desired_ranges = set(ranges)
    # 同时使用目标段的起点和终点切分，缺失区间只作为临时补充段。
    segment_boundaries = sorted({
        frame
        for start_frame, end_frame in ranges
        for frame in (start_frame, end_frame)
        if 0 < frame < frame_count
    })
    expected_ranges = list(zip(
        [0, *segment_boundaries],
        [*segment_boundaries, frame_count],
    ))
    force_keyframes = "+".join(f"eq(n,{frame})" for frame in segment_boundaries)

    with tempfile.TemporaryDirectory(prefix=".split_segments_", dir=video_output_dir) as temp_dir:
        temp_dir = Path(temp_dir)
        temp_output_pattern = temp_dir / "segment-%03d.mp4"
        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-hwaccel", "cuda", "-hwaccel_device", str(device), "-hwaccel_output_format", "cuda",
            "-i", str(input_video), "-map", "0:v:0", "-an", "-vsync", "0",
            "-c:v", "h264_nvenc", "-gpu", str(device), "-preset", "p1",
            "-rc", "vbr", "-cq", "23", "-b:v", "0", "-forced-idr", "1",
        ]
        if segment_boundaries:
            command.extend([
                "-force_key_frames", f"expr:{force_keyframes}",
                "-f", "segment", "-segment_format", "mp4",
                "-segment_frames", ",".join(map(str, segment_boundaries)),
                "-reset_timestamps", "1", str(temp_output_pattern),
            ])
        else:
            command.append(str(temp_dir / "segment-000.mp4"))

        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"FFmpeg 切分视频失败：\n{error.stderr.strip()}") from error

        temp_outputs = sorted(temp_dir.glob("segment-*.mp4"))
        output_paths = []
        output_index = 1

        for temp_output, (start_frame, end_frame) in zip(temp_outputs, expected_ranges):
            clip_frames = get_video_frame_count(temp_output)

            if (start_frame, end_frame) not in desired_ranges:
                print(f"丢弃补充片段：frame {start_frame} ~ {end_frame - 1}，共 {clip_frames} 帧")
                continue

            if clip_frames < min_clip_frames:
                print(f"丢弃短片段：frame {start_frame} ~ {end_frame - 1}，共 {clip_frames} 帧")
                continue

            clip_name = f"{input_video.stem}-{output_index:03d}"
            clip_dir = video_output_dir / clip_name
            clip_dir.mkdir(parents=True, exist_ok=True)
            output_path = clip_dir / f"{clip_name}.mp4"
            output_json = clip_dir / f"{clip_name}.json"
            shutil.move(str(temp_output), str(output_path))

            with output_json.open("w", encoding="utf-8") as json_file:
                json.dump({"split_info": f"[{start_frame}, {end_frame - 1}]"}, json_file, ensure_ascii=False, indent=4)
                json_file.write("\n")

            output_paths.append(output_path)
            output_index += 1

    return output_paths

def parse_cli_paths(
    input_dir: Path,
    output_dir: Path,
    input_video: Path,
    device: int,
) -> tuple[Path, Path, Path, int]:
    """从命令行读取路径，未传入的参数沿用测试值。"""
    parser = argparse.ArgumentParser(description="使用 OmniShotCut 切分视频")

    parser.add_argument(
        "--input_dir",
        "--input-dir",
        dest="input_dir",
        type=Path,
        default=input_dir,
        help="输入视频根目录",
    )

    parser.add_argument(
        "--output_dir",
        "--output-dir",
        dest="output_dir",
        type=Path,
        default=output_dir,
        help="切分结果根目录",
    )

    parser.add_argument(
        "--input_video",
        "--input-video",
        dest="input_video",
        type=Path,
        default=input_video,
        help="需要切分的视频路径",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=device,
        help="GPU 编号，例如 0 或 1",
    )

    args = parser.parse_args()

    return args.input_dir, args.output_dir, args.input_video, args.device


if __name__ == "__main__":
    start_time = time.perf_counter()

    # 【用于测试】
    input_dir = Path("origin_data/batch1")  # 必须是input_video的前缀
    output_dir = Path("origin_data/test_out")
    input_video = Path("origin_data/batch1/武当张资恍/2026-08-28-7679082047969286810/武当张资恍-2026-08-28-7679082047969286810.mp4")
    device = 0

    # 从外部读取
    input_dir, output_dir, input_video, device = parse_cli_paths(
        input_dir,
        output_dir,
        input_video,
        device,
    )

    # 重要参数【可能需要加大】
    min_clip_frames = 20     # 少于该帧数的切片直接丢弃

    video_output_dir = get_video_output_dir(
        input_video,
        input_dir,
        output_dir,
    )

    source_json_output = copy_source_json(
        input_video,
        video_output_dir,
    )

    print(f"开始检测：{input_video}")
    print(f"输出目录：{video_output_dir}")
    print(f"已复制原 JSON：{source_json_output}")
    print(f"使用 GPU：{device}")

    ranges, fps, frame_count = detect_ranges(input_video, device)

    print()
    print(f"原视频共 {frame_count} 帧")
    print(f"FPS：{fps:.2f}")
    print(f"检测到 {len(ranges)} 个切片")

    for start_frame, end_frame in ranges:
        print(
            f"frame={start_frame} ~ {end_frame - 1}, "
            f"time={start_frame / fps:.2f}s ~ "
            f"{(end_frame - 1) / fps:.2f}s"
        )

    output_paths = split_video(
        input_video,
        video_output_dir,
        ranges,
        min_clip_frames,
        device,
    )

    print()
    print(f"切分完成，共输出 {len(output_paths)} 个片段")

    elapsed_seconds = round(time.perf_counter() - start_time)
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"split运行时间：{hours}时{minutes}分{seconds}秒")
