'''
用VolumetricSMPL检测并修复是否穿模【注意需要先安装对应的包】

args:
- smpl_file，例如a/b/c.pt
- fix，例如False
- device，例如0

detect_inpenet函数：
输出：a/b/c.json
字段"inpenet"
- "exist"表示存在穿模的错误
- "nonexist"表示不存在

字段"inpenet_info"
- "3, 4, 9"，表示判定这些帧存在穿模
注意：如果json存在就不要新建了、字段存在就覆盖写

fix_inpenet函数：
第一阶段逐帧修复，输出：a/b/c.pt

fix_smooth函数：
第二阶段联合平滑连续帧，并覆盖a/b/c.pt
'''

import argparse
import json
import shutil
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import torch
import torch.nn.functional as F
from pytorch3d.transforms import axis_angle_to_matrix
from VolumetricSMPL import attach_volume

from GVHMR.hmr4d.utils.smplx_utils import make_smplx


# body_pose 不包含 pelvis；pelvis 的旋转属于 global_orient，本脚本不会优化它。
# 固定 spine1、spine2、spine3、neck、left/right_collar、head，只优化其余关节。
OPTIMIZED_JOINTS = [
    0, 1,        # left/right hip
    3, 4,        # left/right knee
    6, 7,        # left/right ankle
    9, 10,       # left/right foot
    15, 16,      # left/right shoulder
    17, 18,      # left/right elbow
    19, 20,      # left/right wrist
]

def make_model():
    """创建与GVHMR参数一致的SMPL-X模型，并加载体积模型。"""
    model = make_smplx("supermotion")
    attach_volume(model.bm, device=device)
    model = model.eval().to(device)

    # 只优化输入姿态，不计算模型参数的梯度
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def forward_body(body_pose, betas):
    """生成检测自穿模所需的完整SMPL-X输出。"""
    zeros = torch.zeros_like(body_pose[:, :3])
    return model(
        body_pose=body_pose,
        betas=betas,
        global_orient=zeros,
        transl=zeros,
        return_verts=True,
        return_full_pose=True,
    )


def get_self_collision(output, ret_samples=True):
    """计算自穿模损失，并隐藏第三方库内部的调试输出。"""
    model.bm.volume.detach_cache()
    with redirect_stdout(StringIO()):
        return model.bm.volume.self_collision_loss(
            output,
            n_points_uniform=n_points_uniform,
            at_least_n_samples=min_collision_samples,
            ret_samples=ret_samples,
        )


def update_json(smpl_file, inpenet_frames):
    """更新输入文件旁边的检测结果。"""
    json_file = smpl_file.with_suffix(".json")
    info = json.loads(json_file.read_text(encoding="utf-8")) if json_file.exists() else {}
    info["inpenet"] = "exist" if inpenet_frames else "nonexist"
    info["inpenet_info"] = ", ".join(map(str, inpenet_frames))
    json_file.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return json_file


def detect_inpenet(smpl_file):
    """检测所有穿模帧，更新json并返回帧编号。"""
    torch.manual_seed(seed)
    params = torch.load(smpl_file, map_location="cpu", weights_only=True)["smpl_params_global"]
    inpenet_frames = []

    with torch.inference_mode():
        for start in range(0, len(params["body_pose"]), batch_size):
            end = min(start + batch_size, len(params["body_pose"]))
            body_pose = params["body_pose"][start:end].to(device)
            betas = params["betas"][start:end].to(device)

            output = forward_body(body_pose, betas)
            _, samples = get_self_collision(output)

            inpenet_frames.extend(
                start + index
                for index, points in enumerate(samples)
                if points is not None and len(points) >= min_collision_samples
            )

    json_file = update_json(smpl_file, inpenet_frames)
    print(f"穿模帧：{inpenet_frames}")
    print(f"检测结果：{json_file}")
    return inpenet_frames


def optimize_pose(initial_pose, betas):
    """第一阶段：固定躯干，使用穿模和姿态损失优化单帧。"""
    initial_joints = initial_pose.detach().reshape(-1, 21, 3)
    optimized_pose = initial_joints[:, OPTIMIZED_JOINTS].clone().requires_grad_(True)
    optimizer = torch.optim.SGD([optimized_pose], lr=lr)

    def compose_pose():
        """将待优化的四肢参数放回原姿态。"""
        pose = initial_joints.clone()
        pose[:, OPTIMIZED_JOINTS] = optimized_pose
        return pose.reshape_as(initial_pose)

    best_pose = initial_pose.detach().clone()
    best_loss = float("inf")
    clear_count = 0

    for _ in range(max_iters):
        optimizer.zero_grad()
        pose = compose_pose()
        output = forward_body(pose, betas)
        collision_loss, samples = get_self_collision(output)
        collision_loss = collision_loss[0]

        if samples[0] is None or len(samples[0]) < min_collision_samples:
            clear_count += 1
            if clear_count >= clear_checks:
                return pose.detach().clone()
            continue

        clear_count = 0
        pose_loss = F.mse_loss(
            optimized_pose,
            initial_joints[:, OPTIMIZED_JOINTS],
        )
        loss = selfpen_weight * collision_loss + pose_weight * pose_loss
        loss_value = loss.item()
        if loss_value < best_loss:
            best_loss = loss_value
            best_pose = pose.detach().clone()

        loss.backward()
        optimizer.step()

    return best_pose


def verify_pose(body_pose, betas):
    """多次复检单帧姿态，返回是否仍穿模及观测到的最大碰撞信息。"""
    max_collision_count = 0
    max_collision_loss = 0.0
    cuda_devices = [body_pose.device.index] if body_pose.is_cuda else []

    # 复检不应消耗主流程的随机数，避免改变后续帧的优化结果。
    with torch.random.fork_rng(devices=cuda_devices), torch.inference_mode():
        for _ in range(verify_checks):
            output = forward_body(body_pose, betas)
            collision_loss, samples = get_self_collision(output)
            collision_count = 0 if samples[0] is None else len(samples[0])
            max_collision_count = max(max_collision_count, collision_count)
            max_collision_loss = max(max_collision_loss, collision_loss[0].item())

    return (
        max_collision_count >= min_collision_samples,
        max_collision_count,
        max_collision_loss,
    )


def optimize_sequence(initial_pose, original_pose, betas, optimize_mask):
    """第二阶段：固定上下文帧，用旋转修正量平滑连续穿模片段。"""
    initial_joints = initial_pose.detach().reshape(-1, 21, 3)
    original_joints = original_pose.detach().reshape(-1, 21, 3)
    optimize_mask = optimize_mask.to(device=initial_pose.device, dtype=torch.bool)
    optimized_frames = torch.nonzero(optimize_mask).flatten()
    optimized_joint_ids = torch.tensor(OPTIMIZED_JOINTS, device=initial_pose.device)
    optimized_pose = initial_joints[
        optimized_frames[:, None],
        optimized_joint_ids[None, :],
    ].clone().requires_grad_(True)
    optimizer = torch.optim.SGD([optimized_pose], lr=lr)

    def compose_pose():
        """放回可优化参数；固定关节和上下文帧始终使用第一阶段结果。"""
        pose = initial_joints.clone()
        pose[
            optimized_frames[:, None],
            optimized_joint_ids[None, :],
        ] = optimized_pose
        return pose.reshape_as(initial_pose)

    optimized_frame_list = optimized_frames.tolist()

    # 原始局部旋转作为时序参考，只计算一次且不参与梯度传播。
    with torch.no_grad():
        original_rotations = axis_angle_to_matrix(
            original_joints[:, OPTIMIZED_JOINTS]
        )

    for _ in range(smooth_iters):
        optimizer.zero_grad()

        # 第二阶段不再根据采样点判断是否修复完成，只计算并反传穿模损失。
        for pose_index, frame_index in enumerate(optimized_frame_list):
            frame_pose = initial_joints[frame_index:frame_index + 1].clone()
            frame_pose[:, OPTIMIZED_JOINTS] = optimized_pose[pose_index:pose_index + 1]
            frame_pose = frame_pose.reshape(1, -1)

            output = forward_body(frame_pose, betas[frame_index:frame_index + 1])
            collision_loss = get_self_collision(output, ret_samples=False)[0]
            if collision_loss.requires_grad:
                (selfpen_weight * collision_loss).backward()

        pose_difference = (
            optimized_pose
            - initial_joints[
                optimized_frames[:, None],
                optimized_joint_ids[None, :],
            ]
        )
        pose_loss = pose_difference.square().mean(dim=(1, 2)).sum()
        (pose_weight * pose_loss).backward()

        pose_joints = compose_pose().reshape(-1, 21, 3)
        if len(pose_joints) >= 2:
            current_rotations = axis_angle_to_matrix(
                pose_joints[:, OPTIMIZED_JOINTS]
            )
            correction_rotations = (
                current_rotations @ original_rotations.transpose(-1, -2)
            )
            correction_velocity = (
                correction_rotations[1:] - correction_rotations[:-1]
            )
            # 每个相邻帧内部求平均、时间维度求和，避免长片段梯度被稀释。
            smooth_loss = correction_velocity.square().mean(dim=(1, 2, 3)).sum()
            (smooth_weight * smooth_loss).backward()

        optimizer.step()

    return compose_pose().detach()


def get_consecutive_ranges(frames):
    """将帧编号整理为左闭右闭的连续区间。"""
    frames = sorted(set(frames))
    if not frames:
        return []

    ranges = []
    start = previous = frames[0]
    for frame in frames[1:]:
        if frame != previous + 1:
            ranges.append((start, previous))
            start = frame
        previous = frame
    ranges.append((start, previous))
    return ranges


def fix_inpenet(smpl_file, inpenet_frames):
    """第一阶段：逐帧修复穿模并保存结果。"""
    backup_file = smpl_file.with_name(f"{smpl_file.stem}_before_fix_inpenet.pt")
    shutil.copy2(smpl_file, backup_file)
    result = torch.load(backup_file, map_location="cpu", weights_only=True)
    params = result["smpl_params_global"]
    fixed_pose = params["body_pose"].clone()

    for frame in inpenet_frames:
        initial_pose = fixed_pose[frame:frame + 1].to(device)
        betas = params["betas"][frame:frame + 1].to(device)
        optimized_pose = optimize_pose(initial_pose, betas)
        fixed_pose[frame] = optimized_pose[0].cpu()

        still_colliding, collision_count, collision_loss = verify_pose(
            optimized_pose,
            betas,
        )
        if still_colliding:
            print(
                f"警告：第{frame}帧优化后仍然穿模，"
                f"最大碰撞点数={collision_count}，最大碰撞loss={collision_loss:.6f}"
            )
        else:
            print(f"第一阶段已修复第{frame}帧")

    for name in ("smpl_params_global", "smpl_params_incam"):
        result[name]["body_pose"] = fixed_pose.clone()

    torch.save(result, smpl_file)
    print(f"修复前备份：{backup_file}")
    print(f"第一阶段修复结果：{smpl_file}")
    return smpl_file


def fix_smooth(original_file, fixed_file, inpenet_frames):
    """第二阶段：联合平滑连续片段，并覆盖第一阶段的输出文件。"""
    original_result = torch.load(original_file, map_location="cpu", weights_only=True)
    result = torch.load(fixed_file, map_location="cpu", weights_only=True)
    original_params = original_result["smpl_params_global"]
    params = result["smpl_params_global"]
    fixed_pose = params["body_pose"].clone()

    frame_count = len(fixed_pose)
    for start, end in get_consecutive_ranges(inpenet_frames):
        # 前后各加入一帧作为固定边界，只将[start, end]内的穿模帧交给优化器。
        context_start = max(0, start - 1)
        context_end = min(frame_count, end + 2)
        initial_pose = fixed_pose[context_start:context_end].to(device)
        original_pose = original_params["body_pose"][context_start:context_end].to(device)
        betas = params["betas"][context_start:context_end].to(device)

        optimize_mask = torch.zeros(len(initial_pose), dtype=torch.bool, device=device)
        local_start = start - context_start
        local_end = end - context_start + 1
        optimize_mask[local_start:local_end] = True

        optimized_segment = optimize_sequence(initial_pose, original_pose, betas, optimize_mask)
        fixed_pose[start:end + 1] = optimized_segment[local_start:local_end].cpu()
        print(f"第二阶段已平滑第{start}-{end}帧")

    for name in ("smpl_params_global", "smpl_params_incam"):
        result[name]["body_pose"] = fixed_pose.clone()

    torch.save(result, fixed_file)
    print(f"第二阶段平滑结果：{fixed_file}")
    return fixed_file


def parse_cli_args(
    smpl_file: Path,
    fix: bool,
    device: int,
) -> tuple[Path, bool, str]:
    """从命令行读取参数，未传入的参数沿用测试值。"""

    def parse_bool(value: str) -> bool:
        normalized = value.lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
        raise argparse.ArgumentTypeError(f"无法解析布尔值：{value}")

    parser = argparse.ArgumentParser(description="检测并修复人体穿模")
    parser.add_argument(
        "--smpl_file",
        "--smpl-file",
        dest="smpl_file",
        type=Path,
        default=smpl_file,
        help="待检测的 SMPL 参数文件",
    )
    parser.add_argument(
        "--fix_inpenet",
        "--fix-inpenet",
        dest="fix",
        type=parse_bool,
        nargs="?",
        const=True,
        default=fix,
        help="是否修复人体穿模；可省略值，或传入 true/false",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=device,
        help="GPU 编号，例如 0 或 1",
    )
    args = parser.parse_args()
    return args.smpl_file, args.fix, f"cuda:{args.device}"


if __name__ == "__main__":
    start_time = time.perf_counter()

    # 【用于测试】
    smpl_file = Path("gvhmr_out/xk/xk.pt")
    fix = False  # 命令行叫fix_inpenet
    device = 0

    # 从外部获取参数
    smpl_file, fix, device = parse_cli_args(smpl_file, fix, device)

    # 损失权重
    selfpen_weight = 1.0
    pose_weight = 100.0
    smooth_weight = 200000.0

    batch_size = 8
    n_points_uniform = 300
    min_collision_samples = 2  # 【需要调整】
    seed = 0

    max_iters = 200  # 第一阶段最大迭代次数
    clear_checks = 3
    verify_checks = 3  # 第一阶段完成后，每帧独立复检次数
    smooth_iters = 20  # 第二阶段的迭代次数
    lr = 1e-5

    model = make_model()
    inpenet_frames = detect_inpenet(smpl_file)  # 检测，几乎不耗时间
    if fix:
        backup_file = smpl_file.with_name(f"{smpl_file.stem}_before_fix_inpenet.pt")
        fixed_file = fix_inpenet(smpl_file, inpenet_frames)  # 第一阶段，解决穿模
        detect_inpenet(fixed_file)  # 检测第一阶段的结果
        fix_smooth(backup_file, fixed_file, inpenet_frames)  # 第二阶段，加入平滑
        detect_inpenet(fixed_file)  # 检测第二阶段的结果

    elapsed_seconds = round(time.perf_counter() - start_time)
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"inpenet运行时间：{hours}时{minutes}分{seconds}秒")
