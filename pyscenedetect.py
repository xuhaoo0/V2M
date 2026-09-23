'''
切分一条视频【需要安装pyscenedetect对应的包，效果很差】

args:
- input_video，例如{input_dir}/a/b/c.mp4
- input_dir，用于替换路径
- output_dir，把所有输出放在这下面，例如{output_dir}/a/b/c/c-001/c-001.mp4
'''

import argparse
import subprocess
from pathlib import Path


def get_video_output_dir(
    input_video: Path, input_dir: Path, output_dir: Path
) -> Path:
    """根据输入视频的相对路径生成输出目录。"""
    relative_path = input_video.relative_to(input_dir)
    return output_dir / relative_path.parent / relative_path.stem


def split_video(
    input_video: Path,
    video_output_dir: Path,
    adaptive_threshold: float,
    min_content_val: float,
    min_scene_len: str,
) -> None:
    """检测镜头并切分视频。"""
    video_output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "scenedetect",
        "-i",
        str(input_video),
        "-o",
        str(video_output_dir),
        "detect-adaptive",
        "--threshold",
        str(adaptive_threshold),
        "--min-content-val",
        str(min_content_val),
        "--min-scene-len",
        min_scene_len,
        "list-scenes",
        "--filename",
        "scene_list.csv",
        "split-video",
        "--filename",
        "$VIDEO_NAME-$SCENE_NUMBER",
    ]
    subprocess.run(command, check=True)


def organize_clips(video_output_dir: Path, video_name: str) -> list[Path]:
    """把每个视频片段放进同名文件夹。"""
    clip_paths = sorted(video_output_dir.glob(f"{video_name}-*.mp4"))
    output_paths = []

    for clip_path in clip_paths:
        clip_dir = clip_path.with_suffix("")
        clip_dir.mkdir(exist_ok=True)
        output_path = clip_dir / clip_path.name
        clip_path.replace(output_path)
        output_paths.append(output_path)

    return output_paths


def parse_cli_paths(
    input_dir: Path,
    output_dir: Path,
    input_video: Path,
) -> tuple[Path, Path, Path]:
    """从命令行读取路径，未传入的参数沿用测试值。"""
    parser = argparse.ArgumentParser(description="检测镜头并切分视频")
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
    # 【用于测试】
    input_dir = Path("origin_data/batch1")  # 必须是input_video的前缀
    output_dir = Path("martial_data")
    input_video = Path("origin_data/batch1/武当黄少侠/2026-04-27-7633317253353848761/武当黄少侠-2026-04-27-7633317253353848761.mp4")

    # 从外部读取
    input_dir, output_dir, input_video = parse_cli_paths(input_dir, output_dir, input_video)

    # 切分参数，数值越小越敏感【很难调整】
    adaptive_threshold = 2.0  # 当前画面变化相对于前后帧平均变化的突兀程度
    min_content_val = 8.0  # 画面变化量
    min_scene_len = "2s"  # 最短镜头时长

    video_output_dir = get_video_output_dir(input_video, input_dir, output_dir)
    print(f"开始切分：{input_video}")
    print(f"输出目录：{video_output_dir}")

    split_video(
        input_video,
        video_output_dir,
        adaptive_threshold,
        min_content_val,
        min_scene_len,
    )
    output_paths = organize_clips(video_output_dir, input_video.stem)

    print(f"切分完成，共输出 {len(output_paths)} 个片段")
