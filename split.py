'''
按画面突变切分一条视频
【需要安装包：pip install git+https://github.com/UVA-Computer-Vision-Lab/OmniShotCut.git】

args:
- input_video，例如{input_dir}/a/b/c.mp4
- input_dir，用于替换路径
- output_dir，把所有输出放在这下面，例如{output_dir}/a/b/c/c-001/c-001.mp4
'''

import argparse
import json
import shutil
import subprocess
import time
from fractions import Fraction
from pathlib import Path

import omnishotcut


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

    其中前面的右端点是下一段的起点，最后一段的右端点
    是最后一帧的编号。
    """
    raw_ranges = [tuple(map(int, frame_range)) for frame_range in ranges]
    normalized_ranges = []

    for index, frame_range in enumerate(raw_ranges):
        if len(frame_range) != 2:
            raise ValueError(f"无效的切片区间：{frame_range}")

        start_frame, raw_end_frame = frame_range
        end_frame = raw_end_frame + (index == len(raw_ranges) - 1)

        if start_frame < 0 or end_frame <= start_frame:
            raise ValueError(f"无效的切片区间：{frame_range}")

        normalized_ranges.append((start_frame, end_frame))

    return normalized_ranges


def detect_ranges(
    input_video: Path,
) -> tuple[list[tuple[int, int]], float, int]:
    """使用 OmniShotCut 检测原视频切片。"""
    fps = get_video_fps(input_video)

    model = omnishotcut.load(str(MODEL_PATH))
    ranges = model.inference(
        str(input_video),
        mode="clean_shot",
    )

    normalized_ranges = normalize_ranges(ranges)

    if not normalized_ranges:
        raise RuntimeError("OmniShotCut 未返回任何切片区间")

    frame_count = normalized_ranges[-1][1]
    return normalized_ranges, fps, frame_count


def split_video(
    input_video: Path,
    video_output_dir: Path,
    ranges: list[tuple[int, int]],
    min_clip_frames: int,
) -> list[Path]:
    """
    根据 OmniShotCut 返回的区间切分视频。

    使用一次 FFmpeg 解码生成所有保留的切片。
    少于 min_clip_frames 帧的切片直接丢弃。
    """
    video_output_dir.mkdir(parents=True, exist_ok=True)

    clips = []
    output_index = 1

    for start_frame, end_frame in ranges:
        clip_frames = end_frame - start_frame

        # 少于指定帧数的切片直接丢弃
        if clip_frames < min_clip_frames:
            print(
                f"跳过短片段：frame {start_frame} ~ {end_frame - 1}，"
                f"共 {clip_frames} 帧"
            )
            continue

        clip_name = f"{input_video.stem}-{output_index:03d}"

        clip_dir = video_output_dir / clip_name
        clip_dir.mkdir(parents=True, exist_ok=True)

        output_path = clip_dir / f"{clip_name}.mp4"
        output_json = clip_dir / f"{clip_name}.json"

        clips.append(
            (start_frame, end_frame, output_path, output_json)
        )
        output_index += 1

    if not clips:
        return []

    filter_parts = []

    if len(clips) == 1:
        trim_inputs = ["[0:v]"]
    else:
        split_outputs = "".join(
            f"[split{index}]" for index in range(len(clips))
        )
        filter_parts.append(
            f"[0:v]split={len(clips)}{split_outputs}"
        )
        trim_inputs = [
            f"[split{index}]" for index in range(len(clips))
        ]

    for index, (start_frame, end_frame, _, _) in enumerate(clips):
        filter_parts.append(
            f"{trim_inputs[index]}"
            f"trim=start_frame={start_frame}:end_frame={end_frame},"
            f"setpts=PTS-STARTPTS[clip{index}]"
        )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-filter_complex",
        ";".join(filter_parts),
    ]

    for index, (_, _, output_path, _) in enumerate(clips):
        command.extend(
            [
                "-map",
                f"[clip{index}]",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ]
        )

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for start_frame, end_frame, _, output_json in clips:
        split_info = {
            "split_info": f"[{start_frame}, {end_frame - 1}]"
        }
        with output_json.open("w", encoding="utf-8") as json_file:
            json.dump(split_info, json_file, ensure_ascii=False, indent=4)
            json_file.write("\n")

    return [output_path for _, _, output_path, _ in clips]


def parse_cli_paths(
    input_dir: Path,
    output_dir: Path,
    input_video: Path,
) -> tuple[Path, Path, Path]:
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

    args = parser.parse_args()

    return args.input_dir, args.output_dir, args.input_video


if __name__ == "__main__":
    start_time = time.perf_counter()

    # 【用于测试】
    input_dir = Path("origin_data/batch1")  # 必须是input_video的前缀
    output_dir = Path("origin_data/test_out")
    input_video = Path("origin_data/batch1/武当张资恍/2026-08-28-7679082047969286810/武当张资恍-2026-08-28-7679082047969286810.mp4")

    # 从外部读取
    input_dir, output_dir, input_video = parse_cli_paths(
        input_dir,
        output_dir,
        input_video,
    )

    # 重要参数
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

    ranges, fps, frame_count = detect_ranges(input_video)

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
    )

    print()
    print(f"切分完成，共输出 {len(output_paths)} 个片段")

    elapsed_seconds = round(time.perf_counter() - start_time)
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"split运行时间：{hours}时{minutes}分{seconds}秒")
