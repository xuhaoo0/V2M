'''
检测是否存在悬空和穿地

思路：
- 先取所有帧的y，排序后取较小的一半，取均值，得到地面。把所有帧减去这个值，即把地面放在y=0。
- 判断穿地：定义一个穿地阈值，如果该帧的最低点小于这个阈值，就认为存在穿地。
- 判断悬空：难点在于如何区分“走着走着突然浮空了”和“跳跃”。利用现象：浮空动作最后不会回到地面；跳跃动作会回到地面。
同样定义一个悬空阈值，如果该帧最低点先大于一个阈值，认为在空中了，标记为空中状态；后续始终没有帧小于这个阈值，就判定为浮空。

args:
- smpl_file，例如a/b/c.pt
- device, 例如0

输出：a/b/c.json

字段"float"
- "exist"表示存在悬空或穿地的错误
- "nonexist"表示不存在
字段"groundpnt"
- "exist"表示存在悬空或穿地的错误
- "nonexist"表示不存在

字段"float_info"
- "N"，表示判定从第N帧开始浮空
字段"groundpnt_info"
- "3, 4, 9"，表示第3,4,9帧存在穿地
注意：如果json存在就不要新建了、字段存在就覆盖写
'''

import argparse
import json
import time
from pathlib import Path

import torch

from GVHMR.hmr4d.utils.smplx_utils import make_smplx


@torch.inference_mode()
def get_frame_min_y(smpl_file, device, batch_size):
    """用较低一半帧的平均高度移动trans，并返回移动后的最低点。"""
    params = torch.load(smpl_file, map_location="cpu", weights_only=True)["smpl_params_global"]
    model = make_smplx("supermotion").eval().to(device)
    frame_min_y = []

    for start in range(0, len(params["body_pose"]), batch_size):
        batch_params = {
            name: value[start:start + batch_size].to(device)
            for name, value in params.items()
        }
        vertices = model(**batch_params).vertices
        frame_min_y.append(vertices[..., 1].amin(dim=1).cpu())

    frame_min_y = torch.cat(frame_min_y)
    sorted_min_y = frame_min_y.sort().values
    ground_y = sorted_min_y[:len(sorted_min_y) // 2].mean()

    # 整个序列沿y轴平移，最低点也减去相同的距离
    params["transl"][:, 1] -= ground_y
    return frame_min_y - ground_y, ground_y


def detect_groundpnt(frame_min_y, json_file, penetration_threshold):
    """检测穿地并记录所有穿地帧。"""
    penetration_frames = torch.where(frame_min_y < penetration_threshold)[0].tolist()
    result = json.loads(json_file.read_text(encoding="utf-8")) if json_file.exists() else {}
    result["groundpnt"] = "exist" if penetration_frames else "nonexist"
    result["groundpnt_info"] = ", ".join(map(str, penetration_frames))
    result.pop("penetration_info", None)
    json_file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"穿地检测：{result['groundpnt']}，共{len(penetration_frames)}帧")
    return penetration_frames


def detect_float(frame_min_y, json_file, float_threshold):
    """检测最后一次离地后是否始终没有落地。"""
    float_start = None
    for frame, min_y in enumerate(frame_min_y):
        if min_y > float_threshold and float_start is None:
            float_start = frame
        elif min_y <= float_threshold:
            float_start = None

    result = json.loads(json_file.read_text(encoding="utf-8")) if json_file.exists() else {}
    result["float"] = "exist" if float_start is not None else "nonexist"
    result["float_info"] = str(float_start) if float_start is not None else ""
    json_file.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"悬空检测：{result['float']}，起始帧：{float_start if float_start is not None else '无'}")
    return float_start


def parse_cli_args(smpl_file: Path, device: int) -> tuple[Path, str]:
    """从命令行读取参数，未传入的参数沿用测试值。"""
    parser = argparse.ArgumentParser(description="检测人体悬空和穿地")
    parser.add_argument(
        "--smpl_file",
        "--smpl-file",
        dest="smpl_file",
        type=Path,
        default=smpl_file,
        help="待检测的 SMPL 参数文件",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=device,
        help="GPU 编号，例如 0 或 1",
    )
    args = parser.parse_args()
    return args.smpl_file, f"cuda:{args.device}"


if __name__ == "__main__":
    start_time = time.perf_counter()

    # 【用于测试】
    smpl_file = Path("gvhmr_out/xk/xk.pt")
    device = 0

    # 从外部获取参数
    smpl_file, device = parse_cli_args(smpl_file, device)

    json_file = smpl_file.with_suffix(".json")
    batch_size = 64
    penetration_threshold = -0.2  # 这两个阈值有待修改
    float_threshold = 0.2

    frame_min_y, ground_y = get_frame_min_y(smpl_file, device, batch_size)
    print(f"地面高度：{ground_y:.4f} m，trans的y已整体移动{-ground_y:.4f} m")
    print(f"最低点范围：{frame_min_y.min():.4f} m ～ {frame_min_y.max():.4f} m")

    detect_groundpnt(frame_min_y, json_file, penetration_threshold)
    detect_float(frame_min_y, json_file, float_threshold)
    print(f"检测结果：{json_file}")

    elapsed_seconds = round(time.perf_counter() - start_time)
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"float运行时间：{hours}时{minutes}分{seconds}秒")
