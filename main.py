'''
在这里将所有流程串起来，只留统一的参数，并能根据参数判断
内容应该是一行一行命令（保证可以注释掉几条命令），从config.yml中读取参数【直接写在这里更好？】

args:
- input_video，例如a/b/c.mp4
- input_dir，例如a，会把a文件夹下的所有视频都跑一遍
- output_dir，例如out，把所有输出放在这下面，并保持原有的目录结构
- gpu，例如0，使用哪块显卡

输出示例：
假设处理的视频是{input_dir}/a/b/c.mp4
{output_dir}
- a/b/c
  - c-001
    - c-001.mp4  # from split
    - c-001  # from gvhmr
      - xxx.pt
      - xxx.mp4
    - c-001.json  # from detect_xxx
    - c-001.pt  # from fix
  - c-002
    ...

'''



'''
流程（顺序很重要）：
split：切一条视频，参考宗宇的代码？【未完成】
gvhmr：重建一条视频（内含2D的左右腿修复）【完成】
fix_leg：判断是否存在左右腿互换【没批量测试】
fix_inpenet：用VolumetricSMPL判断是否存在穿模以及修正【没批量测试】
detect_float：判断是否存在悬空或穿地【没批量测试】
'''