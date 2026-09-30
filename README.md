务必仔细阅读项目目录下每个脚本顶部的注释、底部的超参数

功能测试：直接修改内部的超参数，运行单个脚本

批量运行：修改config.yml的配置，运行main.py里给的命令

环境配置命令如下：

conda create -y -n v2m python=3.10

conda activate v2m

cd GVHMR

pip install -r requirements.txt

pip install -e .

cd ..

下载模型权重：

将 /media/data/xuhao_data/GVHMR_data/inputs/ 复制到你的 V2M/GVHMR/ 下面

【注意：如果没用中关村八卡服务器，则需要自行下载：https://github.com/zju3dv/GVHMR/blob/main/docs/INSTALL.md】

将 /media/data/xuhao_data/V2M/checkpoints 复制到你的 V2M/ 下面

【链接：https://huggingface.co/uva-cv-lab/OmniShotCut_v1.5/tree/main】

某些包需要看对应的功能脚本顶部的注释，会有安装说明