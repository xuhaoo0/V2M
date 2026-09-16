'''
gvhmr内部会用到这个脚本的方法
'''

from pathlib import Path

import torch


# COCO17 中左右髋、膝、踝的索引
LEFT_LEG = [11, 13, 15]
RIGHT_LEG = [12, 14, 16]


def swap_leg(frame):
    """交换单帧的左右腿关键点。"""
    result = frame.clone()
    result[LEFT_LEG] = frame[RIGHT_LEG]
    result[RIGHT_LEG] = frame[LEFT_LEG]
    return result


def leg_delta(previous, current):
    """计算两帧左右腿六个二维关键点的平均位移。"""
    left = torch.norm(current[LEFT_LEG, :2] - previous[LEFT_LEG, :2], dim=-1)
    right = torch.norm(current[RIGHT_LEG, :2] - previous[RIGHT_LEG, :2], dim=-1)
    return torch.cat([left, right]).mean()


def clean_transitions(transitions, max_continuous_frames=10):
    """反复移除超长修改区间的起始突变帧，直到所有区间都符合限制。"""
    cleaned = transitions.clone()

    while True:
        transition_frames = torch.nonzero(cleaned, as_tuple=False).flatten().tolist()
        overlong_starts = []

        # 第 0、2、4……个突变帧分别是修改区间的起点，下一个突变帧是终点。
        # 如果没有下一个突变帧，则该区间的终点是序列末尾。
        for pair_id in range(0, len(transition_frames), 2):
            start = transition_frames[pair_id]
            end = (
                transition_frames[pair_id + 1]
                if pair_id + 1 < len(transition_frames)
                else len(cleaned)
            )
            if end - start > max_continuous_frames:
                overlong_starts.append(start)

        if not overlong_starts:
            break

        # 删除后突变帧会重新配对，因此需要进入下一轮重新计算修改区间。
        cleaned[overlong_starts] = False

    return cleaned


def correct_2d_leg(keypoints, threshold=1.0, max_continuous_frames=10):
    """检测左右腿状态切换，并纠正整个二维关键点序列。"""
    if keypoints.ndim != 3 or keypoints.shape[1:] != (17, 3):
        raise ValueError(f"期望输入形状为 (F, 17, 3)，实际为 {tuple(keypoints.shape)}")

    # 第一遍只使用未经修改的原始序列，记录状态发生切换的帧
    transitions = torch.zeros(len(keypoints), dtype=torch.bool)
    for frame_id in range(1, len(keypoints)):
        previous = keypoints[frame_id - 1]
        current = keypoints[frame_id]
        delta1 = leg_delta(previous, current)
        delta2 = leg_delta(previous, swap_leg(current))
        transitions[frame_id] = delta2 < delta1 * threshold

    # 真正修改前，反复去掉连续修改超过指定帧数的区间起点。
    transitions = clean_transitions(transitions, max_continuous_frames)

    # 假设第一帧正确；每遇到一次切换，后续交换状态取反
    corrected = keypoints.clone()
    should_swap = False
    modified_frames = []
    for frame_id in range(len(keypoints)):
        if transitions[frame_id]:
            should_swap = not should_swap
        if should_swap:
            corrected[frame_id] = swap_leg(keypoints[frame_id])
            modified_frames.append(frame_id)

    frame_text = ", ".join(map(str, modified_frames)) if modified_frames else "无"
    print(f"修改帧：{frame_text}")

    return corrected


if __name__ == "__main__":
    # 超参数
    VITPOSE_PATH = Path(  # 需要处理的2D关键点文件
        "outputs/demo/李济钤-2026-07-22-7665295003677351538-002/preprocess/vitpose.pt"
    )
    THRESHOLD = 1.0  # 比较的阈值
    MAX_CONTINUOUS_FRAMES = 10  # 封闭修改区间超过该长度时，视其起点为误判


    origin_path = VITPOSE_PATH.with_name("vitpose_origin.pt")  # 原始文件重命名，最后输出的文件就叫vitpose.pt
    # 重复运行时始终纠正vitpose_origin.pt，所以每次运行的结果一样
    if not origin_path.exists():
        if not VITPOSE_PATH.exists():
            raise FileNotFoundError(f"找不到输入文件：{VITPOSE_PATH}")
        VITPOSE_PATH.rename(origin_path)

    keypoints = torch.load(origin_path, map_location="cpu")
    corrected = correct_2d_leg(
        keypoints,
        threshold=THRESHOLD,
        max_continuous_frames=MAX_CONTINUOUS_FRAMES,
    )
    torch.save(corrected, VITPOSE_PATH)
