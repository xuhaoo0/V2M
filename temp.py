"""对比交换不同腿部关节旋转参数后的前 20 帧。"""

from pathlib import Path

from visualize_smplx import get_params, visualize


INPUT_PATH = Path("gvhmr_out/person1/person1.pt")
FRAME_COUNT = 20

WIDTH = 720
HEIGHT = 1280
FPS = 30
DEVICE = "cuda"
CAMERA_BETA = 3.0
AXIS_LENGTH = 0.5

# body_pose 不包含 pelvis，形状为 (F, 21, 3)。
# 对应关节顺序：left/right hip、left/right knee、left/right ankle、
# left/right foot。
HIP_PAIR = (0, 1)
KNEE_PAIR = (3, 4)
ANKLE_PAIR = (6, 7)
FOOT_PAIR = (9, 10)


def swap_joint_pairs(smplx_params, joint_pairs):
    """复制参数，并交换指定的左右关节轴角旋转。"""
    swapped_params = {name: value.clone() for name, value in smplx_params.items()}
    body_pose = swapped_params["body_pose"].reshape(-1, 21, 3)
    original_pose = body_pose.clone()

    for left_joint, right_joint in joint_pairs:
        body_pose[:, left_joint] = original_pose[:, right_joint]
        body_pose[:, right_joint] = original_pose[:, left_joint]

    swapped_params["body_pose"] = body_pose.reshape(-1, 63)
    return swapped_params


def render(smplx_params, output_path):
    visualize(
        smplx_params,
        output_path,
        WIDTH,
        HEIGHT,
        FPS,
        DEVICE,
        CAMERA_BETA,
        AXIS_LENGTH,
    )


if __name__ == "__main__":
    # 只保留输入序列的前 20 帧，原始 pt 文件不会被修改。
    params = get_params(INPUT_PATH, DEVICE)
    params = {name: value[:FRAME_COUNT].clone() for name, value in params.items()}

    hip_params = swap_joint_pairs(
        params,
        (HIP_PAIR, KNEE_PAIR, ANKLE_PAIR, FOOT_PAIR),
    )
    render(hip_params, INPUT_PATH.parent / "hip.mp4")

    non_hip_params = swap_joint_pairs(
        params,
        (KNEE_PAIR, ANKLE_PAIR, FOOT_PAIR),
    )
    render(non_hip_params, INPUT_PATH.parent / "non-hip.mp4")
