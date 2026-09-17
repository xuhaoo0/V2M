'''
重建一条数据

args:
- video，例如a/b/c.mp4
- output_root，例如gvhmr_out/，结果输出在gvhmr_out/c
'''

import os
import subprocess
import sys
from pathlib import Path

v2m_root = Path(__file__).resolve().parent
gvhmr_root = v2m_root / "GVHMR"

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "1"  # gpu

subprocess.run(
    [
        sys.executable,
        "tools/demo/demo.py",
        "--video",
        "/home/xuhao/data/sam3-main/dance/single/KG_kg_iconx/person1.mp4",  # 输入视频
        "--output_root",
        str(v2m_root / "gvhmr_out"),  # 输出目录（最好不要修改）
    ],
    cwd=gvhmr_root,
    env=env,
    check=True,
)
