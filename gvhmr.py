'''
重建一条数据

args:
- input_video，例如a/b/c/c-001/c-001.mp4

输出：
output_root取input_video的a/b/c/c-001
a/b/c/c-001/c-001文件夹
最后把上面的文件夹里面的pt复制一份，存为a/b/c/c-001/c-001.pt
'''

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
    input_video: Path, output_root: Path, gvhmr_root: Path, gpu: int
) -> None:
    """调用 GVHMR 重建视频。"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

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


if __name__ == "__main__":
    # 路径参数
    gvhmr_root = Path(__file__).resolve().parent / "GVHMR"
    input_video = Path("martial_data/武当陈师睿/2025-08-03-7534355708314733866/武当陈师睿-2025-08-03-7534355708314733866/武当陈师睿-2025-08-03-7534355708314733866-001/武当陈师睿-2025-08-03-7534355708314733866-001.mp4")

    # GPU 参数
    gpu = 1

    output_root, output_dir, output_pt = get_output_paths(input_video)
    print(f"开始重建：{input_video}")
    print(f"输出目录：{output_dir}")

    run_gvhmr(input_video, output_root, gvhmr_root, gpu)
    copy_result(output_dir, output_pt)

    print(f"重建完成：{output_pt}")
