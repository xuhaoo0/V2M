"""可视化smplx动作。

标准化规则：
1. 将第一帧的根关节移动到 x=0、z=0；
2. 将所有帧最低点排序，取较低一半的平均值作为地面并移动到 y=0；
3. 绕 y 轴旋转整段动作，根据参数使第一帧面朝 +X 或 +Z。【可在main里面选择】

后续帧不会再次落地，因此可以直接观察穿地和浮空。
"""

from pathlib import Path

import torch
from tqdm import tqdm

from GVHMR.hmr4d.utils.geo.hmr_cam import create_camera_sensor
from GVHMR.hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from GVHMR.hmr4d.utils.smplx_utils import make_smplx
from GVHMR.hmr4d.utils.video_io_utils import get_writer
from GVHMR.hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points


def get_render_size(width, height, render_scale):
    """按照给定比例缩放画面，并确保输出高度满足常见视频编码要求。"""
    if not 0 < render_scale <= 1:
        raise ValueError(
            f"render_scale 必须在 (0, 1] 范围内，当前值为 {render_scale}"
        )

    render_width = max(1, round(width * render_scale))
    render_height = max(2, round(height * render_scale / 2) * 2)
    return render_width, render_height


def standardize_motion(vertices, joints, facing_direction):
    """根据整段动作估计地面，并以第一帧为基准平移和旋转。"""
    offset = joints[0, 0].clone()
    frame_min_y = vertices[..., 1].amin(dim=1)
    sorted_min_y = frame_min_y.sort().values
    ground_y = sorted_min_y[:len(sorted_min_y) // 2].mean()
    offset[1] = ground_y
    vertices = vertices - offset
    joints = joints - offset

    # 先将第一帧标准化为面朝+Z
    transform = compute_T_ayfz2ay(joints[[0]], inverse=True)[0]
    vertices = apply_T_on_points(vertices, transform)
    joints = apply_T_on_points(joints, transform)

    if facing_direction == "+X":
        rotate_z_to_x = torch.eye(4, dtype=vertices.dtype, device=vertices.device)
        rotate_z_to_x[0, 0] = 0
        rotate_z_to_x[0, 2] = 1
        rotate_z_to_x[2, 0] = -1
        rotate_z_to_x[2, 2] = 0
        vertices = apply_T_on_points(vertices, rotate_z_to_x)
        joints = apply_T_on_points(joints, rotate_z_to_x)
    return vertices, joints, ground_y


def create_coordinate_axes(length, device):
    """创建 +X、+Y、+Z 三个彩色箭头网格。"""
    radius = length * 0.025
    head_length = length * 0.18
    head_radius = radius * 2.5
    shaft_length = length - head_length

    axes = [
        ([1, 0, 0], [0, 1, 0], [0, 0, 1], [0.9, 0.1, 0.1]),  # +X：红色
        ([0, 1, 0], [1, 0, 0], [0, 0, 1], [0.1, 0.8, 0.1]),  # +Y：绿色
        ([0, 0, 1], [1, 0, 0], [0, 1, 0], [0.1, 0.25, 0.95]),  # +Z：蓝色
    ]
    box_faces = torch.tensor(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        device=device,
    )
    head_faces = torch.tensor([[0, 2, 1], [0, 3, 2], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], device=device)

    all_vertices, all_faces, all_colors = [], [], []
    vertex_offset = 0
    for direction, side_u, side_v, color in axes:
        direction = torch.tensor(direction, dtype=torch.float32, device=device)
        side_u = torch.tensor(side_u, dtype=torch.float32, device=device)
        side_v = torch.tensor(side_v, dtype=torch.float32, device=device)
        color = torch.tensor(color, dtype=torch.float32, device=device)

        corners = [-side_u - side_v, side_u - side_v, side_u + side_v, -side_u + side_v]
        shaft = torch.stack(
            [direction * distance + corner * radius for distance in [0, shaft_length] for corner in corners]
        )
        head_center = direction * shaft_length
        head = torch.stack([head_center + corner * head_radius for corner in corners] + [direction * length])

        vertices = torch.cat([shaft, head])
        faces = torch.cat([box_faces, head_faces + len(shaft)]) + vertex_offset
        colors = color.repeat(len(vertices), 1)

        all_vertices.append(vertices)
        all_faces.append(faces)
        all_colors.append(colors)
        vertex_offset += len(vertices)

    return torch.cat(all_vertices), torch.cat(all_faces), torch.cat(all_colors)


def add_coordinate_axes(renderer, axis_length, device):
    """把坐标轴网格加入已有的地面网格。"""
    ground_vertices, ground_faces, ground_colors = renderer.ground_geometry
    axis_vertices, axis_faces, axis_colors = create_coordinate_axes(axis_length, device)
    axis_faces = axis_faces + len(ground_vertices)

    renderer.ground_geometry = [
        torch.cat([ground_vertices, axis_vertices]),
        torch.cat([ground_faces, axis_faces.to(ground_faces)]),
        torch.cat([ground_colors[:, :3], axis_colors]),
    ]


def get_params(input_path, device):
    """读取 GVHMR 的 pt 文件，返回 SMPL-X 参数。"""
    result = torch.load(input_path, map_location="cpu", weights_only=True)
    smplx_params = result["smpl_params_global"]
    smplx_params = {
        name: smplx_params[name].to(device)
        for name in ["body_pose", "betas", "global_orient", "transl"]
    }
    return smplx_params


@torch.inference_mode()
def visualize(
    smplx_params,
    output_path,
    width,
    height,
    fps,
    device,
    camera_beta,
    axis_length,
    facing_direction,
    render_scale=1.0,
):
    width, height = get_render_size(width, height, render_scale)
    print(
        f"渲染分辨率：{width}x{height} "
        f"（缩放参数：{render_scale}）"
    )

    # 由参数生成人体网格和关节
    smplx = make_smplx("supermotion").eval().to(device)
    smplx_output = smplx(**smplx_params)
    vertices = smplx_output.vertices
    joints = smplx_output.joints[:, :22]

    # 使用整段动作估计地面，并按参数统一整段动作的朝向
    vertices, joints, ground_y = standardize_motion(vertices, joints, facing_direction)
    vertices_cpu = vertices.cpu()
    joints_cpu = joints.cpu()

    frame_min_y = vertices_cpu[..., 1].amin(dim=1)
    print(f"估计地面高度：{ground_y:.4f} m")
    print(f"第一帧朝向：{facing_direction}")
    print(f"所有帧最低点范围：[{frame_min_y.min():.4f}, {frame_min_y.max():.4f}] m")
    print(f"穿地帧数：{int((frame_min_y < -1e-4).sum())}/{len(frame_min_y)}")

    # 静态相机只负责构图，不再修改人体动作
    camera_R, camera_T, lights = get_global_cameras_static(
        vertices_cpu,
        beta=camera_beta,
        cam_height_degree=20,
        target_center_height=1.0,
        device=device,
    )
    _, _, K = create_camera_sensor(width, height, 24)
    renderer = Renderer(width, height, device=device, faces=smplx.faces, K=K)

    # 棋盘格始终位于 y=0，只根据动作范围调整 x、z 方向的大小和中心
    ground_size, ground_x, ground_z = get_ground_params_from_points(joints_cpu[:, 0], vertices_cpu)
    renderer.set_ground(ground_size * 1.5, ground_x, ground_z)
    add_coordinate_axes(renderer, axis_length, device)

    # 每个 bin 都预留容纳整幅场景全部面片的空间，避免粗光栅化溢出
    body_face_count = len(smplx.faces)
    ground_and_axis_face_count = len(renderer.ground_geometry[1])
    renderer.renderer.rasterizer.raster_settings.max_faces_per_bin = body_face_count + ground_and_axis_face_count

    color = torch.full((1, 3), 0.8, device=device)
    writer = get_writer(output_path, fps=fps, crf=23)
    for frame_idx in tqdm(range(len(vertices)), desc="正在渲染"):
        camera = renderer.create_camera(camera_R[frame_idx], camera_T[frame_idx])
        image = renderer.render_with_ground(vertices[[frame_idx]], color, camera, lights)
        writer.write_frame(image)
    writer.close()
    print(f"视频已保存到：{output_path}")


if __name__ == "__main__":
    # 直接在这里修改输入路径
    input_path = Path("origin_data/test_visualcompare/张资晃/武当张资恍-2026-08-28-7679082047969286810-001.pt")
    output_path = input_path.with_name(f"{input_path.stem}-smpl.mp4")
    device = "cuda"  # 没有显卡时改成 "cpu"
    facing_direction = "+Z"  # 可选："+X"或"+Z"
    render_scale = 0.5  # 缩小分辨率，提高渲染速度

    width = 720
    height = 1280
    fps = 30
    camera_beta = 3.0  # 数值越大，人物在画面中越小
    axis_length = 0.5  # 坐标轴长度，单位为米

    smplx_params = get_params(input_path, device)
    visualize(
        smplx_params,
        output_path,
        width,
        height,
        fps,
        device,
        camera_beta,
        axis_length,
        facing_direction,
        render_scale,
    )
