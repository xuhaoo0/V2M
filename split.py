'''
切分一条视频

args:
- input_video，例如{input_dir}/a/b/c.mp4
- input_dir，用于替换路径
- output_dir，把所有输出放在这下面，例如{output_dir}/a/b/c/c-001/c-001.mp4
'''

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


if __name__ == "__main__":
    # 路径参数
    input_dir = Path("origin_data/batch1")
    output_dir = Path("martial_data")
    input_video = Path("origin_data/batch1/武当陈师睿/2025-08-03-7534355708314733866/武当陈师睿-2025-08-03-7534355708314733866.mp4")

    # 切分参数，数值越小越敏感【需要调整】
    adaptive_threshold = 3.5
    min_content_val = 10.0

    video_output_dir = get_video_output_dir(input_video, input_dir, output_dir)
    print(f"开始切分：{input_video}")
    print(f"输出目录：{video_output_dir}")

    split_video(
        input_video,
        video_output_dir,
        adaptive_threshold,
        min_content_val,
    )
    output_paths = organize_clips(video_output_dir, input_video.stem)

    print(f"切分完成，共输出 {len(output_paths)} 个片段")
