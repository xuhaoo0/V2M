from ultralytics import YOLO
from hmr4d import PROJ_ROOT

import torch
from tqdm import tqdm

from hmr4d.utils.seq_utils import (
    get_frame_id_list_from_mask,
    linear_interpolate_frame_ids,
    frame_id_to_mask,
    rearrange_by_mask,
)
from hmr4d.utils.video_io_utils import get_video_lwh
from hmr4d.utils.net_utils import moving_average_smooth


class Tracker:
    def __init__(self) -> None:
        # https://docs.ultralytics.com/modes/predict/
        self.yolo = YOLO(PROJ_ROOT / "inputs/checkpoints/yolo/yolov8x.pt")

    def detect(self, video_path):
        cfg = {
            "device": "cuda",
            "conf": 0.5,  # default 0.25, wham 0.5
            "classes": 0,  # human
            "verbose": False,
            "stream": True,
        }
        results = self.yolo.predict(video_path, **cfg)
        detections = []
        for result in tqdm(results, total=get_video_lwh(video_path)[0], desc="YoloV8 Detection"):
            if result.boxes is None or len(result.boxes) == 0:
                detections.append(None)
                continue

            bbx_xyxy = result.boxes.xyxy.cpu()  # (N, 4)
            bbx_wh = bbx_xyxy[:, 2:] - bbx_xyxy[:, :2]
            largest_idx = (bbx_wh[:, 0] * bbx_wh[:, 1]).argmax()
            detections.append(bbx_xyxy[largest_idx])

        return detections

    def get_one_track(self, video_path, track_id=0):
        if track_id != 0:
            raise ValueError("Per-frame main-person detection only supports track_id=0")

        detections = self.detect(video_path)
        frame_ids = torch.tensor([i for i, bbx in enumerate(detections) if bbx is not None])
        if len(frame_ids) == 0:
            print("[Tracker] No person detected; skipping video")
            return None
        bbx_xyxys = torch.stack([bbx for bbx in detections if bbx is not None])

        # interpolate missing frames
        mask = frame_id_to_mask(frame_ids, get_video_lwh(video_path)[0])
        bbx_xyxy_one_track = rearrange_by_mask(bbx_xyxys, mask)  # (F, 4), missing filled with 0
        missing_frame_id_list = get_frame_id_list_from_mask(~mask)  # list of list
        bbx_xyxy_one_track = linear_interpolate_frame_ids(bbx_xyxy_one_track, missing_frame_id_list)
        assert (bbx_xyxy_one_track.sum(1) != 0).all()

        bbx_xyxy_one_track = moving_average_smooth(bbx_xyxy_one_track, window_size=5, dim=0)
        bbx_xyxy_one_track = moving_average_smooth(bbx_xyxy_one_track, window_size=5, dim=0)

        return bbx_xyxy_one_track
