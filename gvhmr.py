'''
跑一条数据
输入：a/b/c.mp4
输出：gvhmr_out/c
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
