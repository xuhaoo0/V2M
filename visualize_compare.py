'''
左侧是原视频、右侧是mesh。

args:
- smpl_file，例如 a/b/c.pt
- device，例如 0
- render_scale，例如 0.5，表示宽高都缩小为原来的 50%

main 中可配置：
- facing_direction，右侧第一帧朝向，可选 +X 或 +Z

输入视频：a/b/c.mp4
输出视频：a/b/c-compare.mp4

左侧是在原视频上叠加相机坐标系人体，右侧是带地面的全局坐标系人体。
'''

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from GVHMR.hmr4d.utils.geo.hmr_cam import create_camera_sensor
from GVHMR.hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from GVHMR.hmr4d.utils.smplx_utils import make_smplx
from GVHMR.hmr4d.utils.video_io_utils import get_video_lwh, get_video_reader, get_writer
from GVHMR.hmr4d.utils.vis.renderer import (
    Renderer,
    get_global_cameras_static,
    get_ground_params_from_points,
)


PROJECT_DIR = Path(__file__).resolve().parent
GVHMR_DIR = PROJECT_DIR / "GVHMR"
CRF = 23


def get_input_output_paths(smpl_file):
    """根据 PT 路径得到同名原视频和对比视频路径。"""
    input_video = smpl_file.with_suffix(".mp4")
    output_video = smpl_file.with_name(f"{smpl_file.stem}-compare.mp4")
    return input_video, output_video


def get_render_size(width, height, render_scale):
    """按照给定比例缩放画面，并确保输出高度满足常见视频编码要求。"""
    if not 0 < render_scale <= 1:
        raise ValueError(
            f"render_scale 必须在 (0, 1] 范围内，当前值为 {render_scale}"
        )

    render_width = max(1, round(width * render_scale))
    render_height = max(2, round(height * render_scale / 2) * 2)
    return render_width, render_height


def move_params_to_device(params, device):
    """把一组 SMPL-X 参数移动到指定设备。"""
    return {
        name: value.to(device)
        for name, value in params.items()
    }


@torch.inference_mode()
def get_vertices(result, device):
    """根据相机坐标系和全局坐标系参数生成人体网格。"""
    smplx = make_smplx("supermotion").eval().to(device)
    smplx2smpl = torch.load(
        GVHMR_DIR / "hmr4d/utils/body_model/smplx2smpl_sparse.pt",
        map_location="cpu",
        weights_only=True,
    ).to(device)

    incam_output = smplx(**move_params_to_device(result["smpl_params_incam"], device))
    incam_vertices = torch.stack([
        torch.matmul(smplx2smpl, vertices)
        for vertices in incam_output.vertices
    ])
    del incam_output

    global_output = smplx(**move_params_to_device(result["smpl_params_global"], device))
    global_vertices = torch.stack([
        torch.matmul(smplx2smpl, vertices)
        for vertices in global_output.vertices
    ])
    del global_output

    return incam_vertices, global_vertices


def standardize_global_vertices(global_vertices, joint_regressor, facing_direction):
    """估计整段动作的地面，并以第一帧为基准平移和旋转。"""
    if facing_direction not in {"+X", "+Z"}:
        raise ValueError(
            f"不支持的第一帧朝向：{facing_direction}，可选值为 +X 或 +Z"
        )

    global_joints = torch.einsum(
        "jv,lvi->lji",
        joint_regressor,
        global_vertices,
    )

    # 第一帧根关节移动到 x=0、z=0；y 方向使用整段动作估计出的地面。
    offset = global_joints[0, 0].clone()
    frame_min_y = global_vertices[..., 1].amin(dim=1)
    sorted_min_y = frame_min_y.sort().values
    lower_half_count = max(1, len(sorted_min_y) // 2)
    ground_y = sorted_min_y[:lower_half_count].mean()
    offset[1] = ground_y
    global_vertices = global_vertices - offset
    global_joints = global_joints - offset

    # 先让第一帧面朝 +Z。
    transform = compute_T_ayfz2ay(global_joints[[0]], inverse=True)[0]
    global_vertices = apply_T_on_points(global_vertices, transform)
    global_joints = apply_T_on_points(global_joints, transform)

    # 如有需要，再绕 y 轴把 +Z 旋转到 +X。
    if facing_direction == "+X":
        rotate_z_to_x = torch.eye(
            4,
            dtype=global_vertices.dtype,
            device=global_vertices.device,
        )
        rotate_z_to_x[0, 0] = 0
        rotate_z_to_x[0, 2] = 1
        rotate_z_to_x[2, 0] = -1
        rotate_z_to_x[2, 2] = 0
        global_vertices = apply_T_on_points(global_vertices, rotate_z_to_x)
        global_joints = apply_T_on_points(global_joints, rotate_z_to_x)

    return global_vertices, global_joints, ground_y


@torch.inference_mode()
def visualize_compare(
    smpl_file,
    input_video,
    output_video,
    device,
    facing_direction="+Z",
    render_scale=1.0,
):
    """生成左侧 incam、右侧 global 的横向对比视频。"""
    result = torch.load(smpl_file, map_location="cpu", weights_only=True)
    incam_vertices, global_vertices = get_vertices(result, device)

    joint_regressor = torch.load(
        GVHMR_DIR / "hmr4d/utils/body_model/smpl_neutral_J_regressor.pt",
        map_location="cpu",
        weights_only=True,
    ).to(device)
    global_vertices, global_joints, ground_y = standardize_global_vertices(
        global_vertices,
        joint_regressor,
        facing_direction,
    )

    frame_min_y = global_vertices[..., 1].amin(dim=1).cpu()
    print(f"估计地面高度：{ground_y:.4f} m")
    print(f"右侧第一帧朝向：{facing_direction}")
    print(
        f"右侧所有帧最低点范围："
        f"[{frame_min_y.min():.4f}, {frame_min_y.max():.4f}] m"
    )
    print(
        f"右侧穿地帧数："
        f"{int((frame_min_y < -1e-4).sum())}/{len(frame_min_y)}"
    )

    frame_count, source_width, source_height = get_video_lwh(input_video)
    width, height = get_render_size(
        source_width,
        source_height,
        render_scale,
    )
    scale_x = width / source_width
    scale_y = height / source_height
    print(
        f"渲染分辨率：{source_width}x{source_height} -> {width}x{height} "
        f"（缩放参数：{render_scale}）"
    )

    result_frame_count = len(incam_vertices)
    if len(global_vertices) != result_frame_count:
        raise ValueError(
            f"incam 和 global 帧数不一致："
            f"{result_frame_count} != {len(global_vertices)}"
        )
    if frame_count != result_frame_count:
        raise ValueError(
            f"视频和 PT 帧数不一致：{frame_count} != {result_frame_count}"
        )

    faces = make_smplx("smpl").faces
    incam_intrinsics = result["K_fullimg"][0].clone()
    incam_intrinsics[0] *= scale_x
    incam_intrinsics[1] *= scale_y
    incam_renderer = Renderer(
        width,
        height,
        device=device,
        faces=faces,
        K=incam_intrinsics,
    )

    _, _, global_intrinsics = create_camera_sensor(width, height, 24)
    global_renderer = Renderer(
        width,
        height,
        device=device,
        faces=faces,
        K=global_intrinsics,
    )
    ground_size, ground_x, ground_z = get_ground_params_from_points(
        global_joints[:, 0].cpu(),
        global_vertices.cpu(),
    )
    global_renderer.set_ground(
        ground_size * GLOBAL_GROUND_SCALE,
        ground_x,
        ground_z,
    )

    global_rotation, global_translation, global_lights = get_global_cameras_static(
        global_vertices.cpu(),
        beta=GLOBAL_CAMERA_BETA,
        cam_height_degree=20,
        target_center_height=1.0,
        device=device,
    )
    color = torch.ones(3, device=device) * 0.8

    reader = get_video_reader(input_video)
    writer = get_writer(output_video, fps=30, crf=CRF)
    try:
        for frame, image in tqdm(
            enumerate(reader),
            total=frame_count,
            desc="Rendering Compare",
        ):
            if image.shape[1] != width or image.shape[0] != height:
                image = cv2.resize(
                    image,
                    (width, height),
                    interpolation=cv2.INTER_AREA,
                )
            incam_image = incam_renderer.render_mesh(
                incam_vertices[frame],
                image,
                [0.8, 0.8, 0.8],
            )
            global_camera = global_renderer.create_camera(
                global_rotation[frame],
                global_translation[frame],
            )
            global_image = global_renderer.render_with_ground(
                global_vertices[[frame]],
                color[None],
                global_camera,
                global_lights,
            )
            writer.write_frame(np.concatenate([incam_image, global_image], axis=1))
    finally:
        writer.close()
        reader.close()

    print(f"对比视频：{output_video}")
    return output_video


def parse_cli_args(
    smpl_file: Path,
    device: int,
    render_scale: float,
) -> tuple[Path, str, float]:
    """从命令行读取参数，未传入的参数沿用测试值。"""
    parser = argparse.ArgumentParser(description="生成 GVHMR 相机/全局横向对比视频")
    parser.add_argument(
        "--smpl_file",
        "--smpl-file",
        dest="smpl_file",
        type=Path,
        default=smpl_file,
        help="GVHMR 输出的 PT 文件",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=device,
        help="GPU 编号，例如 0 或 1",
    )
    parser.add_argument(
        "--render_scale",
        "--render-scale",
        dest="render_scale",
        type=float,
        default=render_scale,
        help="输出单侧画面的缩放比例，范围为 (0, 1]，例如 0.5",
    )
    args = parser.parse_args()
    return args.smpl_file, f"cuda:{args.device}", args.render_scale


if __name__ == "__main__":
    start_time = time.perf_counter()
    GLOBAL_CAMERA_BETA = 3.0  # 数值越大，相机越远，人物越小
    GLOBAL_GROUND_SCALE = 2.0  # 原来为1.5，控制地板放大的倍数
    facing_direction = "+Z"  # 右侧第一帧朝向，可选："+X" 或 "+Z"
    render_scale = 0.25  # 缩小分辨率，提高渲染速度

    # 【用于测试】
    smpl_file = Path("origin_data/test_visualcompare/张资晃/武当张资恍-2026-08-28-7679082047969286810-001.pt")
    device = 1

    # 从外部获取参数
    smpl_file, device, render_scale = parse_cli_args(
        smpl_file,
        device,
        render_scale,
    )
    input_video, output_video = get_input_output_paths(smpl_file)

    if not smpl_file.is_file():
        raise FileNotFoundError(f"PT 文件不存在：{smpl_file}")
    if not input_video.is_file():
        raise FileNotFoundError(f"同名原视频不存在：{input_video}")

    print(f"SMPL 文件：{smpl_file}")
    print(f"原视频：{input_video}")
    print(f"输出视频：{output_video}")
    visualize_compare(
        smpl_file,
        input_video,
        output_video,
        device,
        facing_direction,
        render_scale,
    )

    elapsed_seconds = round(time.perf_counter() - start_time)
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"compare运行时间：{hours}时{minutes}分{seconds}秒")
