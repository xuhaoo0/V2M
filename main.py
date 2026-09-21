'''
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
    - c-001.pt  # from gvhmr
    - c-001-fix-xxx # from fix_xxx
  - c-002
    ...

'''

'''
基本功能：
split：切一条视频
gvhmr：重建一条视频（内含2D的左右腿修复）
fix_leg：判断是否存在左右腿互换
fix_inpenet：用VolumetricSMPL判断是否存在穿模以及修正
detect_float：判断是否存在悬空或穿地
'''

'''
流程：
从config.yml读取所有参数
遍历input_dir下面的所有mp4
如果split：
  用命令行调用split
如果不split：
  直接用该mp4作为“-001”构建类似“切视频”的结果
对切出来的每个视频（结合config.yml里面的字段决定是否调用该py、传入的参数值是什么）：
  用命令行调用gvhmr、fix_leg、fix_inpenet、detect_float
'''

import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parent


def run_script(script, *args):
    command = [sys.executable, str(PROJECT_DIR / script), *map(str, args)]
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def get_clip_videos(input_video, input_dir, output_dir, split):
    relative_path = input_video.relative_to(input_dir)
    video_output_dir = output_dir / relative_path.parent / input_video.stem
    # 切视频
    if split:
        run_script(
            "split.py",
            "--input_dir", input_dir,
            "--output_dir", output_dir,
            "--input_video", input_video,
        )
        return sorted(video_output_dir.glob("*/*.mp4"))
    # 如果不切视频，还是用原视频构造一个001切片，方便后续处理
    clip_name = f"{input_video.stem}-001"
    clip_video = video_output_dir / clip_name / f"{clip_name}.mp4"
    clip_video.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_video, clip_video)
    return [clip_video]


def main():
    with (PROJECT_DIR / "config.yml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    input_dir = PROJECT_DIR / config["input_dir"]
    output_dir = PROJECT_DIR / config["output_dir"]
    device = config["device"][0]

    input_videos = sorted(input_dir.rglob("*.mp4"))
    for input_video in input_videos:  # 遍历所有mp4
        start_time = time.perf_counter()
        clip_videos = get_clip_videos(
            input_video,
            input_dir,
            output_dir,
            config["split"],
        )

        for clip_video in clip_videos:  # 遍历所有切片
            smpl_file = clip_video.with_suffix(".pt")

            if config["gvhmr"]:
                run_script(
                    "gvhmr.py",
                    "--input_video", clip_video,
                    "--device", device,
                )

            if config["detect_leg"]:
                run_script(
                    "fix_leg.py",
                    "--smpl_file", smpl_file,
                    "--fix_leg", str(config["fix_leg"]).lower(),
                    "--device", device,
                )

            if config["detect_inpenet"]:
                run_script(
                    "fix_inpenet.py",
                    "--smpl_file", smpl_file,
                    "--fix_inpenet", str(config["fix_inpenet"]).lower(),
                    "--device", device,
                )

            if config["detect_float"]:
                run_script(
                    "detect_float.py",
                    "--smpl_file", smpl_file,
                    "--device", device,
                )

        elapsed_seconds = round(time.perf_counter() - start_time)
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"{input_video} 运行时间：{hours}时{minutes}分{seconds}秒")


if __name__ == "__main__":
    main()
