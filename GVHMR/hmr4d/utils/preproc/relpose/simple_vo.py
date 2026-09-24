import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from .utils import focal_length_from_mm
from .matcher_wrapper import Matcher
from .solver_two_view import TwoPairSolver, CameraParams, interpolate_missing_frames
from tqdm import tqdm

from hmr4d.utils.video_io_utils import get_video_lwh, read_video_np


class SimpleVO:
    def __init__(self, video_path, scale=0.5, step=8, method="sift", f_mm=None, num_workers=1):
        self.video_path = video_path
        self.scale = scale
        self.step = step
        self.method = method
        self.f_mm = 24 if f_mm is None else f_mm  # fullframe camera focal length in mm
        self.num_workers = num_workers

    def compute(self):
        # Read video
        frames = read_video_np(self.video_path, scale=self.scale)

        # Downsample frames, and interpolate missing frames
        F_all = frames.shape[0]
        sample_idxs = np.arange(0, F_all, self.step)
        if sample_idxs[-1] != F_all - 1:
            sample_idxs = np.concatenate([sample_idxs, [F_all - 1]])
        frames = frames[sample_idxs]
        F, H, W, C = frames.shape
        print(f"[SimpleVO] Choosen frames shape: {frames.shape}")

        matcher: Matcher = Matcher(self.method)
        camera_params = CameraParams(W, H, focal_length=focal_length_from_mm(W, H, self.f_mm))
        solver: TwoPairSolver = TwoPairSolver(camera_params, solver="pycolmap")

        # TODO:We should use different pipelines for different methods
        print("gvhmr: start visual odometry", flush=True)
        T_w2c_list = self.process_video_T_w2c_list_np(frames, matcher, solver)

        # Interpolate missing frames
        T_w2c_list = interpolate_missing_frames(T_w2c_list, sample_idxs)

        return T_w2c_list

    @staticmethod
    def solve_frame_pair(prev_frame, curr_frame, matcher: Matcher, solver: TwoPairSolver):
        try:
            pts0, pts1 = matcher.match_np(prev_frame, curr_frame)
        except cv2.error:
            print("Warning: feature matching failed; using previous camera pose")
            return None

        if len(pts0) < 10:
            print(
                f"Warning: only {len(pts0)} matches found; "
                "using previous camera pose"
            )
            return None
        return solver.solve(pts0, pts1)

    def process_video_T_w2c_list_np(self, frames, matcher: Matcher, solver: TwoPairSolver):
        if self.num_workers <= 1:
            # Serial path: identical to the pre-parallel implementation, including
            # incremental accumulation (so a mid-loop exception leaves a partial
            # T_w2c_list, matching prior behavior).
            T_w2c_list = [np.eye(4)]
            prev_frame = frames[0]
            for frame_idx in tqdm(range(1, len(frames))):
                curr_frame = frames[frame_idx]
                T_delta = self.solve_frame_pair(prev_frame, curr_frame, matcher, solver)
                if T_delta is None:
                    T_w2c_list.append(T_w2c_list[-1].copy())
                else:
                    T_w2c_list.append(T_delta @ T_w2c_list[-1])
                prev_frame = curr_frame
            return T_w2c_list

        # Parallel path: each thread owns its own Matcher/TwoPairSolver to avoid
        # sharing cv2.SIFT / pycolmap internals across threads. Per-thread
        # instances are cloned from the passed-in objects so configuration
        # (matcher type/args, solver backend, camera params) carries over.
        #
        # Note: this is NOT bit-for-bit equivalent to the serial path. pycolmap
        # RANSAC is non-deterministic, and accumulation is deferred until all
        # deltas are collected, so partial-failure behavior also differs.
        n_pairs = len(frames) - 1
        tls = threading.local()

        def get_worker_state():
            if not hasattr(tls, "matcher"):
                tls.matcher = matcher.clone()
                tls.solver = solver.clone()
            return tls.matcher, tls.solver

        def solve_pair(i):
            m, s = get_worker_state()
            return self.solve_frame_pair(frames[i], frames[i + 1], m, s)

        T_deltas = [None] * n_pairs
        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            for i, T in tqdm(
                zip(range(n_pairs), ex.map(solve_pair, range(n_pairs))),
                total=n_pairs,
            ):
                T_deltas[i] = T

        T_w2c_list = [np.eye(4)]
        for T_delta in T_deltas:
            if T_delta is None:
                T_w2c_list.append(T_w2c_list[-1].copy())
            else:
                T_w2c_list.append(T_delta @ T_w2c_list[-1])
        return T_w2c_list
