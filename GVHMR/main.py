'''
跑一条数据
'''

import os
import subprocess

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

subprocess.run([
    "python",
    "tools/demo/demo.py",
    "--video",
    "/home/xuhao/data/sam3-main/dance/single_white/KG_kg_iconx/person1.mp4"
], check=True)