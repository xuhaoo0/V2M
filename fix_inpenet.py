'''
用VolumetricSMPL检测并修复是否穿模

args:
- smpl_file，例如a/b/c.pt

detect_inpenet函数：
输出：a/b/c.json
字段"inpenet"
- "exist"表示存在穿模的错误
- "nonexist"表示不存在

字段"inpenet_info"
- "3, 4, 9"，表示判定这些帧存在穿模
注意：如果json存在就不要新建了、字段存在就覆盖写

fix_inpenet函数：
输出：a/b/c_fix_inpenet.pt，表示修复后的smpl参数
'''

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import torch
import torch.nn.functional as F
from VolumetricSMPL import attach_volume

from GVHMR.hmr4d.utils.smplx_utils import make_smplx


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


def get_self_collision(output):
    """计算自穿模损失，并隐藏第三方库内部的调试输出。"""
    model.bm.volume.detach_cache()
    with redirect_stdout(StringIO()):
        return model.bm.volume.self_collision_loss(
            output,
            n_points_uniform=n_points_uniform,
            at_least_n_samples=min_collision_samples,
            ret_samples=True,
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
    """在尽量保持原姿态的前提下修复单帧穿模。"""
    pose = initial_pose.detach().clone().requires_grad_(True)
    optimizer = torch.optim.SGD([pose], lr=lr)
    best_pose = pose.detach().clone()
    best_collision = float("inf")
    clear_count = 0

    for _ in range(max_iters):
        optimizer.zero_grad()
        output = forward_body(pose, betas)
        collision_loss, samples = get_self_collision(output)
        collision_loss = collision_loss[0]

        if samples[0] is None or len(samples[0]) < min_collision_samples:
            clear_count += 1
            if clear_count >= clear_checks:
                return pose.detach().clone()
            continue

        clear_count = 0
        if collision_loss.item() < best_collision:
            best_collision = collision_loss.item()
            best_pose = pose.detach().clone()

        pose_loss = F.mse_loss(pose, initial_pose)
        loss = selfpen_weight * collision_loss + pose_weight * pose_loss
        loss.backward()
        optimizer.step()

    return best_pose


def fix_inpenet(smpl_file, inpenet_frames):
    """修复指定帧，并保存新的SMPL参数文件。"""
    result = torch.load(smpl_file, map_location="cpu", weights_only=True)
    params = result["smpl_params_global"]
    fixed_pose = params["body_pose"].clone()

    for frame in inpenet_frames:
        initial_pose = fixed_pose[frame:frame + 1].to(device)
        betas = params["betas"][frame:frame + 1].to(device)
        fixed_pose[frame] = optimize_pose(initial_pose, betas)[0].cpu()
        print(f"已修复第{frame}帧")

    # global和incam的body_pose相同，两份都需要同步修改
    for name in ("smpl_params_global", "smpl_params_incam"):
        result[name]["body_pose"] = fixed_pose.clone()

    output_file = smpl_file.with_name(f"{smpl_file.stem}_fix_inpenet.pt")
    torch.save(result, output_file)
    print(f"修复结果：{output_file}")
    return output_file


if __name__ == "__main__":
    smpl_file = Path("gvhmr_out/inpenet/inpenet.pt")
    device = "cuda:0"

    batch_size = 8
    n_points_uniform = 300
    min_collision_samples = 2
    seed = 0

    max_iters = 200
    clear_checks = 3
    lr = 1e-5
    selfpen_weight = 0.1
    pose_weight = 100.0

    model = make_model()
    inpenet_frames = detect_inpenet(smpl_file)
    fix_inpenet(smpl_file, inpenet_frames)
