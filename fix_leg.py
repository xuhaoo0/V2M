'''
检测左右腿是否互换，还可以选择修复

args:
- smpl_file，例如a/b/c.pt

detect_leg函数：
输出：a/b/c.json
字段"leg"
- "exist"表示存在左右腿互换的错误
- "nonexist"表示不存在
字段"leg_info"
- "3, 4, 8, 9"，记录所有需要修改的帧，如果不存在则值为空
注意：如果json存在就不要新建了、字段存在就覆盖写

fix_leg函数：
输出：a/b/c_fix_leg.pt，即修复之后，交换左右hip、knee、ankle、foot旋转参数的结果。
'''

import json
from pathlib import Path

import torch

from GVHMR.hmr4d.utils.smplx_utils import make_smplx


# body_pose 不包含 pelvis，下面是其中的左右关节编号
LEG_PAIRS = ((0, 1), (3, 4), (6, 7), (9, 10))

# SMPL-X 完整关节中的 knee、ankle、foot 编号
LEFT_JOINTS = (4, 7, 10)
RIGHT_JOINTS = (5, 8, 11)


def swap_leg_pose(body_pose):
    """交换所有帧的左右 hip、knee、ankle、foot 旋转参数。"""
    swapped = body_pose.clone()
    pose = swapped.reshape(*swapped.shape[:-1], 21, 3)
    original = pose.clone()

    for left, right in LEG_PAIRS:
        pose[..., left, :] = original[..., right, :]
        pose[..., right, :] = original[..., left, :]

    return swapped


def get_joints(smplx, params, body_pose, device):
    """去掉人物整体平移和朝向，只计算姿态产生的关节位置。"""
    with torch.inference_mode():
        output = smplx(
            body_pose=body_pose.to(device),
            betas=params["betas"].to(device),
            global_orient=torch.zeros_like(params["global_orient"], device=device),
            transl=torch.zeros_like(params["transl"], device=device),
        )
    return output.joints[:, :22].cpu()


def detect_switch_frames(params, device):
    """逐帧比较原姿态和交换腿部参数后的姿态，返回状态切换帧。"""
    smplx = make_smplx("supermotion").eval().to(device)
    body_pose = params["body_pose"]

    joints = get_joints(smplx, params, body_pose, device)
    swapped_joints = get_joints(smplx, params, swap_leg_pose(body_pose), device)

    previous_left = joints[:-1, LEFT_JOINTS]
    previous_right = joints[:-1, RIGHT_JOINTS]

    delta1 = torch.cat(
        [
            torch.linalg.vector_norm(joints[1:, LEFT_JOINTS] - previous_left, dim=-1),
            torch.linalg.vector_norm(joints[1:, RIGHT_JOINTS] - previous_right, dim=-1),
        ],
        dim=-1,
    ).mean(dim=-1)

    delta2 = torch.cat(
        [
            torch.linalg.vector_norm(swapped_joints[1:, LEFT_JOINTS] - previous_left, dim=-1),
            torch.linalg.vector_norm(swapped_joints[1:, RIGHT_JOINTS] - previous_right, dim=-1),
        ],
        dim=-1,
    ).mean(dim=-1)

    # 差分的第0项对应原序列第1帧，所以帧号需要加1
    return (torch.nonzero(delta1 > delta2).flatten() + 1).tolist()


def get_fix_ranges(switch_frames, frame_count):
    """根据切换帧得到需要修改的左闭右开区间。"""
    ranges = []
    for index in range(0, len(switch_frames), 2):
        start = switch_frames[index]
        end = switch_frames[index + 1] if index + 1 < len(switch_frames) else frame_count
        ranges.append((start, end))
    return ranges


def clean_switch_frames(switch_frames, frame_count, max_fix_frames):
    """反复删除会产生过长修改区间的切换帧。"""
    switch_frames = switch_frames.copy()

    while True:
        long_range = next(
            (
                (start, end)
                for start, end in get_fix_ranges(switch_frames, frame_count)
                if end - start > max_fix_frames
            ),
            None,
        )
        if long_range is None:
            return switch_frames

        start, end = long_range
        index = switch_frames.index(start)

        # 延续到结尾的区间只删除起点；普通区间同时删除起点和终点
        if end == frame_count:
            del switch_frames[index]
        else:
            del switch_frames[index:index + 2]


def apply_fix(body_pose, fix_frames):
    """只在指定帧使用交换后的腿部旋转参数。"""
    fixed = body_pose.clone()
    swapped = swap_leg_pose(body_pose)
    fixed[fix_frames] = swapped[fix_frames]
    return fixed


def update_json(smpl_file, fix_frames):
    json_file = smpl_file.with_suffix(".json")
    info = json.loads(json_file.read_text(encoding="utf-8")) if json_file.exists() else {}
    info["leg"] = "exist" if fix_frames else "nonexist"
    info["leg_info"] = ", ".join(map(str, fix_frames))
    json_file.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def detect_leg(smpl_file, max_fix_frames, device):
    """检测需要交换左右腿的帧，并更新json。"""
    result = torch.load(smpl_file, map_location="cpu", weights_only=True)
    params = result["smpl_params_global"]
    frame_count = len(params["body_pose"])

    switch_frames = detect_switch_frames(params, device)
    switch_frames = clean_switch_frames(switch_frames, frame_count, max_fix_frames)
    fix_ranges = get_fix_ranges(switch_frames, frame_count)
    fix_frames = [frame for start, end in fix_ranges for frame in range(start, end)]

    update_json(smpl_file, fix_frames)

    print(f"切换帧：{switch_frames}")
    print(f"修改区间：{fix_ranges}")
    print(f"需要修改的帧：{fix_frames}")
    return fix_frames


def fix_leg(smpl_file, fix_frames):
    """交换指定帧的左右腿旋转参数，并保存修复结果。"""
    result = torch.load(smpl_file, map_location="cpu", weights_only=True)

    # global和incam的body_pose相同，但文件中保存了两份，都需要修改
    for name in ("smpl_params_global", "smpl_params_incam"):
        result[name]["body_pose"] = apply_fix(result[name]["body_pose"], fix_frames)

    output_file = smpl_file.with_name(f"{smpl_file.stem}_fix_leg.pt")
    torch.save(result, output_file)
    print(f"修复结果：{output_file}")


if __name__ == "__main__":
    smpl_file = Path("gvhmr_out/leg/leg.pt")
    max_fix_frames = 10
    device = "cuda"

    fix_frames = detect_leg(smpl_file, max_fix_frames, device)  # 检测功能
    fix_leg(smpl_file, fix_frames)  # 修复功能
