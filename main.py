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

import multiprocessing as mp
import os
import queue
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parent


def run_script(script, *args, log_file=None):
    command = [sys.executable, str(PROJECT_DIR / script), *map(str, args)]
    if log_file is not None:
        log_file.flush()
    subprocess.run(
        command,
        cwd=PROJECT_DIR,
        check=True,
        stdout=log_file,
        stderr=subprocess.STDOUT if log_file is not None else None,
    )


def get_clip_videos(input_video, input_dir, output_dir, split, log_file=None):
    relative_path = input_video.relative_to(input_dir)
    video_output_dir = output_dir / relative_path.parent / input_video.stem
    # 切视频
    if split:
        run_script(
            "split.py",
            "--input_dir", input_dir,
            "--output_dir", output_dir,
            "--input_video", input_video,
            log_file=log_file,
        )
        return sorted(video_output_dir.glob("*/*.mp4"))
    # 如果不切视频，还是用原视频构造一个001切片，方便后续处理
    clip_name = f"{input_video.stem}-001"
    clip_video = video_output_dir / clip_name / f"{clip_name}.mp4"
    clip_video.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_video, clip_video)
    return [clip_video]


def process_video(input_video, input_dir, output_dir, config, device, log_file):
    """在固定GPU上处理一条原始视频及其所有切片。"""
    start_time = time.perf_counter()
    clip_videos = get_clip_videos(
        input_video,
        input_dir,
        output_dir,
        config["split"],
        log_file,
    )

    for clip_video in clip_videos:
        smpl_file = clip_video.with_suffix(".pt")

        if config["gvhmr"]:
            run_script(
                "gvhmr.py",
                "--input_video", clip_video,
                "--device", device,
                log_file=log_file,
            )

        if config["detect_leg"]:
            run_script(
                "fix_leg.py",
                "--smpl_file", smpl_file,
                "--fix_leg", str(config["fix_leg"]).lower(),
                "--device", device,
                log_file=log_file,
            )

        if config["detect_inpenet"]:
            run_script(
                "fix_inpenet.py",
                "--smpl_file", smpl_file,
                "--fix_inpenet", str(config["fix_inpenet"]).lower(),
                "--device", device,
                log_file=log_file,
            )

        if config["detect_float"]:
            run_script(
                "detect_float.py",
                "--smpl_file", smpl_file,
                "--device", device,
                log_file=log_file,
            )

    elapsed_seconds = round(time.perf_counter() - start_time)
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"{input_video} 运行时间：{hours}时{minutes}分{seconds}秒")


def collect_tasks(input_videos):
    """把所有原始视频放入多进程共享任务队列。"""
    task_queue = mp.JoinableQueue()
    for input_video in input_videos:
        task_queue.put(input_video)
    return task_queue


def worker(
    task_queue,
    gpu_id,
    input_dir,
    output_dir,
    config,
    log_dir,
    completed_count,
    total_tasks,
):
    """从共享队列领取视频，并始终使用指定GPU。"""
    pid = os.getpid()
    log_path = log_dir / f"main_{gpu_id}_{pid}.log"

    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = log_file
        sys.stderr = log_file

        try:
            print(f"***** process: {pid}, GPU: {gpu_id}, start *****")
            while True:
                status = "fail"
                try:
                    input_video = task_queue.get(timeout=2)
                except queue.Empty:
                    print(f"***** process: {pid}, GPU: {gpu_id}, quit *****")
                    break

                try:
                    print(
                        f"***** process: {pid}, GPU: {gpu_id}, "
                        f"video: {input_video}, start *****"
                    )
                    process_video(
                        input_video,
                        input_dir,
                        output_dir,
                        config,
                        gpu_id,
                        log_file,
                    )
                    print(
                        f"***** process: {pid}, GPU: {gpu_id}, "
                        f"video: {input_video}, finished *****"
                    )
                    status = "success"
                except Exception:
                    print(
                        f"***** process: {pid}, GPU: {gpu_id}, "
                        f"video: {input_video}, exception *****"
                    )
                    traceback.print_exc()
                finally:
                    with completed_count.get_lock():
                        completed_count.value += 1
                        completed_index = completed_count.value
                        print(
                            f"[{completed_index}/{total_tasks}] {status}, "
                            f"GPU: {gpu_id}, process: {pid}, "
                            f"task: {input_video.stem}",
                            file=original_stdout,
                            flush=True,
                        )
                    task_queue.task_done()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def launch_processes(
    process_num,
    task_queue,
    gpu_list,
    input_dir,
    output_dir,
    config,
    log_dir,
    completed_count,
    total_tasks,
):
    """为每张GPU启动固定数量的worker。"""
    processes = []
    for gpu_id in gpu_list:
        for _ in range(process_num):
            process = mp.Process(
                target=worker,
                args=(
                    task_queue,
                    gpu_id,
                    input_dir,
                    output_dir,
                    config,
                    log_dir,
                    completed_count,
                    total_tasks,
                ),
                daemon=True,
            )
            process.start()
            processes.append(process)
    return processes


def run_on_multigpu(input_videos, input_dir, output_dir, config):
    """使用多GPU、每卡多进程处理全部视频。"""
    gpu_list = config["device"]
    process_num = config["process_num"]

    if not isinstance(gpu_list, list) or not gpu_list:
        raise ValueError("config.yml中的device必须是非空GPU编号列表")
    if not isinstance(process_num, int) or process_num <= 0:
        raise ValueError("config.yml中的process_num必须是正整数")

    if not input_videos:
        print("未找到需要处理的mp4视频")
        return

    log_dir = PROJECT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    task_queue = collect_tasks(input_videos)
    completed_count = mp.Value("i", 0)
    total_tasks = len(input_videos)
    print(
        f"***** total {total_tasks} videos, GPUs: {gpu_list}, "
        f"processes per GPU: {process_num} *****"
    )
    processes = launch_processes(
        process_num,
        task_queue,
        gpu_list,
        input_dir,
        output_dir,
        config,
        log_dir,
        completed_count,
        total_tasks,
    )

    task_queue.join()
    for process in processes:
        process.join(timeout=4)
        if process.is_alive():
            process.terminate()
            process.join()
            print(f"***** process: {process.pid} has been forcefully terminated *****")

    print(f"***** all tasks finished; logs: {log_dir} *****")


def main():
    with (PROJECT_DIR / "config.yml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    input_dir = PROJECT_DIR / config["input_dir"]
    output_dir = PROJECT_DIR / config["output_dir"]
    input_videos = sorted(input_dir.rglob("*.mp4"))
    run_on_multigpu(input_videos, input_dir, output_dir, config)


if __name__ == "__main__":
    '''
    注意，这个脚本是多卡多进程的，谨慎运行
    启动命令：nohup python -u main.py > main.log 2>&1 &
    进程组id会写在上面的log里面：假设是12345
    杀死进程组：kill -TERM -12345
    '''
    print(f"进程组：{os.getpgrp()}", flush=True)
    mp.set_start_method("spawn", force=True)
    main()
