'''
重建一条数据

args:
- video，例如a/b/c.mp4
- output_root，例如a/b/c

输出：
a/b/c/c文件夹，最重要的是里面的c.pt
'''

import os
import shutil
import subprocess
import sys
from pathlib import Path

v2m_root = Path(__file__).resolve().parent
gvhmr_root = v2m_root / "GVHMR"

# 超参数
video = Path("/home/xuhao/data/V2M/xk.mp4")  # 要使用绝对路径
output_root = v2m_root / "gvhmr_out"

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "1"  # gpu

subprocess.run(
    [
        sys.executable,
        "tools/demo/demo.py",
        "--video",
        str(video),
        "--output_root",
        str(output_root),
    ],
    cwd=gvhmr_root,
    env=env,
    check=True,
)

# 复制一份pt文件，供后续检测、修复，作为最终结果。
output_dir = output_root / video.stem
shutil.copy2(output_dir / "hmr4d_results.pt", output_dir / f"{video.stem}.pt")
