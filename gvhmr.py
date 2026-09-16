'''
跑一条数据
args:
- video，例如a/b/c.mp4
- output_root，例如gvhmr_out/，结果输出在gvhmr_out/c
'''

import os
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # gpu

subprocess.run([
    "python",
    "GVHMR/tools/demo/demo.py",
    "--video",
    "/home/xuhao/data/sam3-main/dance/single/KG_kg_iconx/person1.mp4",  # 输入视频
    "--output_root",
    "gvhmr_out",  # 输出目录
], check=True)
