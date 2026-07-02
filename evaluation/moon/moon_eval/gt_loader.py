"""
moon_eval/gt_loader.py — Ground-truth loading utilities.

Loads per-image GT data:
  - .npz files: intrinsics (3x3) + cam2world (4x4)
  - .exr files: depth maps (single channel Y, float32)

Backprojects depth to world-frame 3D points.
"""

import numpy as np
from pathlib import Path
from typing import Tuple

import OpenEXR
import Imath


def read_exr(path) -> np.ndarray:
    """Read a single-channel EXR depth file.

    Returns the depth array of shape (H, W), dtype float32.
    """
    ex = OpenEXR.InputFile(str(path))
    dw = ex.header()["dataWindow"]
    W0 = dw.max.x - dw.min.x + 1
    H0 = dw.max.y - dw.min.y + 1
    buf = ex.channel("Y", Imath.PixelType(Imath.PixelType.FLOAT))
    depth = np.frombuffer(buf, dtype=np.float32).reshape(H0, W0)
    return depth


def load_gt_view(
    gt_folder: Path,
    img_stem: str,
    Hc: int = 384,
    Wc: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load GT for a single view.

    Reads the .npz (intrinsics, cam2world) and .exr (depth) for `img_stem`,
    center-crops the depth to (Hc, Wc), backprojects to world-frame 3D points.

    Returns:
        pts_world  : (Hc*Wc, 3)  3D points in world frame
        depth_map  : (Hc, Wc)    center-cropped depth
        Kc         : (3, 3)      intrinsics adjusted for the crop
        T_w_c      : (4, 4)      cam2world transform
        mask_valid : (Hc*Wc,)    bool, True where pts_world is finite
    """
    gt_folder = Path(gt_folder)
    data = np.load(gt_folder / f"{img_stem}.npz")
    K_gt = data["intrinsics"]
    T_w_c = data["cam2world"]

    depth_full = read_exr(gt_folder / f"{img_stem}.exr")
    y0 = (depth_full.shape[0] - Hc) // 2
    x0 = (depth_full.shape[1] - Wc) // 2
    depth_map = depth_full[y0:y0 + Hc, x0:x0 + Wc]

    # Adjust principal point for the crop
    Kc = K_gt.copy()
    Kc[0, 2] -= x0
    Kc[1, 2] -= y0

    # Back-project depth map to camera frame
    u, v = np.meshgrid(np.arange(Wc), np.arange(Hc))
    z = depth_map
    x_c = (u - Kc[0, 2]) * z / Kc[0, 0]
    y_c = (v - Kc[1, 2]) * z / Kc[1, 1]
    pts_cam = np.stack([x_c, y_c, z], axis=-1).reshape(-1, 3)

    # Transform to world frame
    hom = np.concatenate([pts_cam, np.ones((Hc * Wc, 1))], axis=1).T
    pts_world = (T_w_c @ hom)[:3].T
    mask_valid = np.isfinite(pts_world).all(axis=1)

    return pts_world, depth_map, Kc, T_w_c, mask_valid


def get_gt_relative_pose(
    gt_folder: Path,
    stem0: str,
    stem1: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute GT relative pose between two views.

    Loads cam2world for both views from their .npz files and computes:
        T_rel = inv(T_wc1) @ T_wc0

    Returns:
        R_gt : (3, 3) rotation matrix
        t_gt : (3,)   translation vector
    """
    gt_folder = Path(gt_folder)
    data0 = np.load(gt_folder / f"{stem0}.npz")
    data1 = np.load(gt_folder / f"{stem1}.npz")
    T_wc0 = data0["cam2world"].astype(np.float64)
    T_wc1 = data1["cam2world"].astype(np.float64)
    T_rel = np.linalg.inv(T_wc1) @ T_wc0
    R_gt = T_rel[:3, :3]
    t_gt = T_rel[:3, 3]
    return R_gt, t_gt
