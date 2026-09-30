# V2M

## 使用前必读

务必仔细阅读项目目录下每个脚本顶部的注释和底部的超参数。

## 使用方式

- **功能测试**：直接修改脚本内部的超参数，运行单个脚本
- **批量运行**：修改 `config.yml` 的配置，运行 `main.py` 里给出的命令

## 环境配置

### 1. 创建并激活 Conda 环境

```bash
conda create -y -n v2m python=3.10
conda activate v2m
```

### 2. 安装 GVHMR

```bash
cd GVHMR
pip install -r requirements.txt
pip install -e .
cd ..
```

### 3. 下载模型权重

- 将服务器上 `/media/data/xuhao_data/GVHMR_data/inputs/` 复制到你的 `V2M/GVHMR/` 下
  - 自行下载参考：[GVHMR 安装文档](https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md)
- 将服务器上 `/media/data/xuhao_data/V2M/checkpoints` 复制到你的 `V2M/` 下
  - 自行下载链接：[OmniShotCut_v1.5 权重](https://huggingface.co/uva-cv-lab/OmniShotCut_v1.5/tree/main)

### 4. 其他依赖

部分包需要额外安装，请查看对应功能脚本顶部的注释中的安装说明。
