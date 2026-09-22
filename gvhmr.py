'''
重建一条数据（要求视频里面尽量只有一个人，或者每一帧都能保证重建的人物面积最大）

args:
- input_video，例如a/b/c/c-001/c-001.mp4

输出：
output_root取input_video的a/b/c/c-001
a/b/c/c-001/c-001文件夹
最后把上面的文件夹里面的pt复制一份，存为a/b/c/c-001/c-001.pt
'''

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def get_output_paths(input_video: Path) -> tuple[Path, Path, Path]:
    """根据输入视频路径生成 GVHMR 输出路径。"""
    output_root = input_video.parent
    output_dir = output_root / input_video.stem
    output_pt = output_root / f"{input_video.stem}.pt"
    return output_root, output_dir, output_pt


def run_gvhmr(
    input_video: Path, output_root: Path, gvhmr_root: Path, device: int
) -> None:
    """调用 GVHMR 重建视频。"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)

    command = [
        sys.executable,
        "tools/demo/demo.py",
        "--video",
        str(input_video.resolve()),
        "--output_root",
        str(output_root.resolve()),
    ]
    subprocess.run(command, cwd=gvhmr_root, env=env, check=True)


def copy_result(output_dir: Path, output_pt: Path) -> None:
    """复制重建结果，供后续检测和修复使用。"""
    shutil.copy2(output_dir / "hmr4d_results.pt", output_pt)


def parse_cli_args(input_video: Path, device: int) -> tuple[Path, int]:
    """从命令行读取参数，未传入的参数沿用测试值。"""
    parser = argparse.ArgumentParser(description="使用 GVHMR 重建视频")
    parser.add_argument(
        "--input_video",
        "--input-video",
        dest="input_video",
        type=Path,
        default=input_video,
        help="需要重建的视频路径",
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
    gvhmr_root = Path(__file__).resolve().parent / "GVHMR"

    # 【用于测试】
    input_video = Path("gvhmr_out/long/long.mp4")
    device = 0

    # 从外部获取参数
    input_video, device = parse_cli_args(input_video, device)

    output_root, output_dir, output_pt = get_output_paths(input_video)
    print(f"开始重建：{input_video}")
    print(f"输出目录：{output_dir}")

    run_gvhmr(input_video, output_root, gvhmr_root, device)
    copy_result(output_dir, output_pt)

    print(f"重建完成：{output_pt}")
