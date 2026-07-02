"""
moon_eval/metrics/camera.py — Camera pose evaluation metrics.

Implements:
  - RRA / RTA (Relative Rotation/Translation Accuracy)
  - AUC of the pose error recall curve (standard DUSt3R / VGGT protocol)
  - Essential-matrix pose estimation from 2D-2D matches
  - Match extraction using MASt3R's fast_reciprocal_NNs
  - Direct pose comparison from global_aligner output
"""

import numpy as np
import cv2
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..gt_loader import get_gt_relative_pose


# ─────────────────────────────────────────────────────────────────────────────
# Basic pose error functions  (ported from eval_emat.py)
# ─────────────────────────────────────────────────────────────────────────────

def rra_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """Relative Rotation Accuracy in degrees."""
    R_err = R_est @ R_gt.T
    return float(np.degrees(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))))


def rta_deg(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    """Relative Translation Accuracy in degrees (angular error of unit vectors)."""
    te = t_est / (np.linalg.norm(t_est) + 1e-12)
    tg = t_gt / (np.linalg.norm(t_gt) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(np.dot(te, tg), -1, 1))))


def compute_rra_rta(
    R_est: np.ndarray,
    t_est: np.ndarray,
    R_gt: np.ndarray,
    t_gt: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute RRA, RTA, and max pose error.

    Args:
        R_est, t_est : estimated rotation (3×3) and translation (3,)
        R_gt,  t_gt  : ground-truth rotation (3×3) and translation (3,)

    Returns:
        (rra, rta, pose_error) in degrees
        pose_error = max(rra, rta)
    """
    rra = rra_deg(R_est, R_gt)
    rta = rta_deg(t_est, t_gt)
    return rra, rta, max(rra, rta)


def compute_auc(
    errors: List[float],
    thresholds: Tuple[int, ...] = (5, 10, 20),
) -> Dict[str, float]:
    """AUC of the pose error recall curve (DUSt3R / VGGT standard).

    For each threshold t, the recall curve plots the fraction of pairs with
    pose error ≤ x, for x ∈ [0, t].  AUC is the area under that curve,
    normalised by t, expressed as a percentage.

    Args:
        errors     : list of max(RRA, RTA) per pair (inf for failures)
        thresholds : angular thresholds in degrees

    Returns:
        dict with keys 'AUC@{t}' for each t (0–100 scale)
    """
    errors_arr = np.array(errors, dtype=np.float64)
    result: Dict[str, float] = {}
    for t in thresholds:
        n_bins = t * 10
        x = np.linspace(0, t, n_bins + 1)
        recalls = np.array([(errors_arr <= xi).mean() for xi in x])
        auc = float(np.trapz(recalls, x) / t * 100)
        result[f"AUC@{t}"] = round(auc, 2)
    return result


def compute_vcre_auc(
    vcre_errors: List[float],
    thresholds: Tuple[int, ...] = (50, 100, 200),
) -> Dict[str, float]:
    """AUC of the VCRE recall curve (pixel thresholds), MASt3R-style.

    For each threshold t (in pixels), computes the fraction of pairs with
    VCRE_median ≤ x for x ∈ [0, t], then returns the normalised AUC (0–100).

    Also computes Precision@t = fraction of pairs with VCRE < t.
    """
    errors_arr = np.array(vcre_errors, dtype=np.float64)
    result: Dict[str, float] = {}
    for t in thresholds:
        # Precision
        prec = float((errors_arr < t).mean() * 100)
        result[f"VCRE_Prec@{t}"] = round(prec, 1)
        # AUC
        n_bins = max(int(t * 2), 100)
        x = np.linspace(0, t, n_bins + 1)
        recalls = np.array([(errors_arr <= xi).mean() for xi in x])
        auc = float(np.trapz(recalls, x) / t * 100)
        result[f"VCRE_AUC@{t}"] = round(auc, 2)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Essential-matrix pose estimation
# ─────────────────────────────────────────────────────────────────────────────

def compute_pose_from_essential(
    m0: np.ndarray,
    m1: np.ndarray,
    K: np.ndarray,
    threshold: float = 1.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """5-point RANSAC Essential matrix pose estimation.

    Args:
        m0, m1    : (N, 2) matched pixel coordinates in image 0 and 1
        K         : (3, 3) camera intrinsics
        threshold : RANSAC inlier threshold in pixels

    Returns:
        R         : (3, 3) rotation matrix or None on failure
        t         : (3,) unit translation vector or None on failure
        n_inliers : number of RANSAC inliers
    """
    pts1 = m0.astype(np.float64)
    pts2 = m1.astype(np.float64)
    if len(pts1) < 8:
        return None, None, 0

    E, mask_e = cv2.findEssentialMat(
        pts1, pts2, K.astype(np.float64),
        method=cv2.RANSAC,
        prob=0.9999,
        threshold=threshold,
    )
    if E is None or mask_e is None:
        return None, None, 0

    n_inliers, R, t, _ = cv2.recoverPose(
        E, pts1, pts2, K.astype(np.float64), mask=mask_e
    )
    return R, t.flatten(), int(n_inliers)


# ─────────────────────────────────────────────────────────────────────────────
# Match extraction from MASt3R model
# ─────────────────────────────────────────────────────────────────────────────

def extract_matches_from_model(
    model: torch.nn.Module,
    device,
    img0_path,
    img1_path,
    n_matches: int = 2000,
    border: int = 3,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Run MASt3R inference and extract 2D-2D matches via fast_reciprocal_NNs.

    Follows the same procedure as eval_emat.py:
      1. load_images at size=512
      2. inference (single pair)
      3. fast_reciprocal_NNs with subsample_or_initxy1=8, dist='dot'
      4. border filter
      5. top-N by combined confidence (min of desc_conf * conf_3d for each view)

    Args:
        model     : loaded MASt3R model (eval mode)
        device    : torch device
        img0_path : path to image 0
        img1_path : path to image 1
        n_matches : number of top-confidence matches to keep
        border    : pixel border exclusion

    Returns:
        m0       : (K, 2) matched coords in image 0
        m1       : (K, 2) matched coords in image 1
        n_total  : total matches before top-N selection
    """
    # Lazy imports — must not load before mast3r.utils.path_to_dust3r is set up
    from dust3r.utils.image import load_images
    from dust3r.inference import inference
    from mast3r.fast_nn import fast_reciprocal_NNs

    images = load_images([str(img0_path), str(img1_path)], size=512, verbose=False)
    with torch.no_grad():
        output = inference([tuple(images)], model, device, batch_size=1, verbose=False)

    pred1, pred2 = output["pred1"], output["pred2"]
    view1, view2 = output["view1"], output["view2"]

    desc1 = pred1["desc"].squeeze(0).detach()
    desc2 = pred2["desc"].squeeze(0).detach()

    matches_im0, matches_im1 = fast_reciprocal_NNs(
        desc1, desc2,
        subsample_or_initxy1=8,
        device=device,
        dist="dot",
        block_size=2 ** 13,
    )

    H0, W0 = view1["true_shape"][0]
    H1, W1 = view2["true_shape"][0]
    valid = (
        (matches_im0[:, 0] >= border)
        & (matches_im0[:, 0] < int(W0) - border)
        & (matches_im0[:, 1] >= border)
        & (matches_im0[:, 1] < int(H0) - border)
        & (matches_im1[:, 0] >= border)
        & (matches_im1[:, 0] < int(W1) - border)
        & (matches_im1[:, 1] >= border)
        & (matches_im1[:, 1] < int(H1) - border)
    )
    matches_im0 = matches_im0[valid]
    matches_im1 = matches_im1[valid]
    n_total = len(matches_im0)

    # Compute combined confidence: min(desc_conf * conf_3d) over both views
    conf_im0 = pred1["conf"].squeeze(0).detach().cpu().numpy()
    conf_im1 = pred2["conf"].squeeze(0).detach().cpu().numpy()
    desc_conf_im0 = pred1["desc_conf"].squeeze(0).detach().cpu().numpy()
    desc_conf_im1 = pred2["desc_conf"].squeeze(0).detach().cpu().numpy()

    comb0 = (
        desc_conf_im0[matches_im0[:, 1], matches_im0[:, 0]]
        * conf_im0[matches_im0[:, 1], matches_im0[:, 0]]
    )
    comb1 = (
        desc_conf_im1[matches_im1[:, 1], matches_im1[:, 0]]
        * conf_im1[matches_im1[:, 1], matches_im1[:, 0]]
    )
    combined_conf = np.minimum(comb0, comb1)

    n_keep = min(n_matches, n_total)
    top_idx = np.argsort(combined_conf)[::-1][:n_keep]
    m0 = matches_im0[top_idx]
    m1 = matches_im1[top_idx]

    return m0, m1, n_total


# ─────────────────────────────────────────────────────────────────────────────
# Pose from global aligner (direct comparison, no Essential matrix)
# ─────────────────────────────────────────────────────────────────────────────

def compute_pose_from_aligner(
    poses: np.ndarray,
    gt_folder: Path,
    stem0: str,
    stem1: str,
) -> Dict[str, float]:
    """Compare global_aligner predicted relative pose to GT.

    The global aligner produces cam2world poses for each view.  The predicted
    relative pose is T_rel_pred = inv(poses[1]) @ poses[0].  This is compared
    to the GT relative pose from the NPZ files.

    Args:
        poses     : (2, 4, 4) cam2world poses from ga.get_im_poses()
        gt_folder : path to the GT folder containing .npz files
        stem0     : image stem for view 0 (e.g. 'im_00000')
        stem1     : image stem for view 1 (e.g. 'im_00001')

    Returns:
        dict with keys: rra_aligner, rta_aligner, pose_error_aligner
    """
    try:
        R_gt, t_gt = get_gt_relative_pose(gt_folder, stem0, stem1)

        T_rel_pred = np.linalg.inv(poses[1].astype(np.float64)) @ poses[0].astype(np.float64)
        R_pred = T_rel_pred[:3, :3]
        t_pred = T_rel_pred[:3, 3]

        rra, rta, pe = compute_rra_rta(R_pred, t_pred, R_gt, t_gt)
        return {
            "rra_aligner": rra,
            "rta_aligner": rta,
            "pose_error_aligner": pe,
        }
    except Exception as e:
        return {
            "rra_aligner": float("inf"),
            "rta_aligner": float("inf"),
            "pose_error_aligner": float("inf"),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Full camera metrics for one pair
# ─────────────────────────────────────────────────────────────────────────────

def compute_camera_metrics_for_pair(
    model: torch.nn.Module,
    device,
    img0_path,
    img1_path,
    gt_folder: Path,
    K_GT: np.ndarray,
    n_matches: int = 2000,
    threshold: float = 1.0,
    poses_from_aligner: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Full camera pose evaluation for one image pair.

    Runs Essential-matrix pose estimation from MASt3R matches and, optionally,
    compares the global_aligner predicted poses to GT directly.

    Args:
        model               : loaded MASt3R model
        device              : torch device
        img0_path, img1_path: image paths
        gt_folder           : path to GT folder
        K_GT                : (3, 3) camera intrinsics
        n_matches           : number of top matches for Essential matrix
        threshold           : RANSAC threshold (pixels)
        poses_from_aligner  : (2, 4, 4) or None — if provided, also computes
                              rra_aligner / rta_aligner

    Returns:
        flat dict with keys:
          rra_emat, rta_emat, pose_error_emat, n_inliers_emat, n_matches_emat,
          AUC@5, AUC@10, AUC@20  (from this single pair's pose error),
          rra_aligner, rta_aligner, pose_error_aligner  (if poses provided)
    """
    stem0 = Path(img0_path).stem
    stem1 = Path(img1_path).stem
    result: Dict[str, float] = {}

    # ── Essential matrix path ─────────────────────────────────────────────────
    try:
        R_gt, t_gt = get_gt_relative_pose(gt_folder, stem0, stem1)
        m0, m1, n_total = extract_matches_from_model(
            model, device, img0_path, img1_path, n_matches=n_matches, border=3
        )
        result["n_matches_emat"] = int(len(m0))

        R_est, t_est, n_inl = compute_pose_from_essential(
            m0.astype(np.float64), m1.astype(np.float64), K_GT, threshold=threshold
        )
        result["n_inliers_emat"] = int(n_inl)

        if R_est is not None:
            rra, rta, pe = compute_rra_rta(R_est, t_est, R_gt, t_gt)
        else:
            rra = rta = pe = float("inf")
        result["rra_emat"] = rra
        result["rta_emat"] = rta
        result["pose_error_emat"] = pe

        # AUC for this single pair (single-element list)
        auc = compute_auc([pe])
        result.update(auc)

    except Exception as e:
        result.update({
            "rra_emat": float("inf"),
            "rta_emat": float("inf"),
            "pose_error_emat": float("inf"),
            "n_inliers_emat": 0,
            "n_matches_emat": 0,
        })

    # ── Aligner pose path ─────────────────────────────────────────────────────
    if poses_from_aligner is not None:
        aligner_metrics = compute_pose_from_aligner(
            poses_from_aligner, gt_folder, stem0, stem1
        )
        result.update(aligner_metrics)

        # ── VCRE (Virtual Correspondence Reprojection Error) ──────────────
        try:
            vcre_metrics = compute_vcre(
                poses_from_aligner, gt_folder, stem0, stem1, K_GT,
            )
            result.update(vcre_metrics)
        except Exception:
            pass

    return result


# ─────────────────────────────────────────────────────────────────────────────
# VCRE  (Virtual Correspondence Reprojection Error)
# ─────────────────────────────────────────────────────────────────────────────

def _make_virtual_cube(center: np.ndarray, half_extent: float, n_per_axis: int = 6):
    """Create a grid of virtual 3D points in a cube around *center*.

    Returns (N, 3) array with N = n_per_axis^3 points.
    """
    lin = np.linspace(-half_extent, half_extent, n_per_axis)
    gx, gy, gz = np.meshgrid(lin, lin, lin)
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1)  # (N, 3)
    return pts + center


def compute_vcre(
    poses_pred: np.ndarray,
    gt_folder: Path,
    stem0: str,
    stem1: str,
    K: np.ndarray,
    n_per_axis: int = 6,
    image_size: Tuple[int, int] = (512, 384),
) -> Dict[str, float]:
    """Compute VCRE between predicted and GT relative poses.

    Standard Map-free relocalization metric (Arnold et al., ECCV 2022):
    place a cube of virtual 3D points in front of the cameras, project them
    with both the GT and predicted *relative* poses, and measure the median
    reprojection error in pixels.

    The cube is centred on the GT 3D scene centroid at median-depth distance
    along the camera Z axis, with half-extent equal to half the depth range
    so that it spans the visible scene.

    Args:
        poses_pred : (2, 4, 4) cam2world from global_aligner
        gt_folder  : path to GT folder with .npz files
        stem0, stem1 : image stems
        K          : (3, 3) camera intrinsics
        n_per_axis : points per cube axis (total = n^3)
        image_size : (W, H) used to normalise by image diagonal

    Returns:
        dict with vcre_median_px, vcre_mean_px, vcre_pct (normalised by diag)
    """
    gt_folder = Path(gt_folder)

    # GT cam2world
    T_wc0_gt = np.load(gt_folder / f"{stem0}.npz")["cam2world"].astype(np.float64)
    T_wc1_gt = np.load(gt_folder / f"{stem1}.npz")["cam2world"].astype(np.float64)

    # Relative poses: cam1 ← cam0
    T_rel_gt   = np.linalg.inv(T_wc1_gt) @ T_wc0_gt
    T_rel_pred = np.linalg.inv(poses_pred[1].astype(np.float64)) @ poses_pred[0].astype(np.float64)

    # The global_aligner outputs poses in an arbitrary scale.  The virtual
    # cube is placed at GT depth (metres), so we must rescale the predicted
    # translation to match the GT baseline magnitude.  This isolates the
    # *directional* translation error (and rotation error) from the scale.
    t_gt_norm   = np.linalg.norm(T_rel_gt[:3, 3])
    t_pred_norm = np.linalg.norm(T_rel_pred[:3, 3])
    if t_pred_norm > 1e-12 and t_gt_norm > 1e-12:
        T_rel_pred[:3, 3] *= t_gt_norm / t_pred_norm

    # Build virtual cube in cam0 frame at scene depth.
    # Load GT depth to place the cube at the correct distance.
    from ..gt_loader import load_gt_view
    _, depth_gt, _, _, mask_gt = load_gt_view(gt_folder, stem0, 384, 512)
    valid_depths = depth_gt[mask_gt.reshape(384, 512)]
    valid_depths = valid_depths[np.isfinite(valid_depths) & (valid_depths > 0)]
    if len(valid_depths) < 10:
        return {}
    median_depth = float(np.median(valid_depths))
    depth_span = float(np.percentile(valid_depths, 95) - np.percentile(valid_depths, 5))
    # Cube centre: straight ahead at median depth, half-extent = half the depth span
    center = np.array([0.0, 0.0, median_depth])
    half_ext = max(depth_span * 0.5, median_depth * 0.1)  # at least 10% of depth

    pts_cam0 = _make_virtual_cube(center, half_ext, n_per_axis)  # (N, 3)

    # Only keep points with positive Z in both GT views
    # In cam1_gt frame:
    pts_cam1_gt = (T_rel_gt[:3, :3] @ pts_cam0.T).T + T_rel_gt[:3, 3]
    valid = (pts_cam0[:, 2] > 0) & (pts_cam1_gt[:, 2] > 0)
    if valid.sum() < 4:
        return {}

    pts_cam0 = pts_cam0[valid]

    # Project into cam1 with GT relative pose → 2D GT
    pts_cam1_gt = (T_rel_gt[:3, :3] @ pts_cam0.T).T + T_rel_gt[:3, 3]
    proj_gt = (K @ pts_cam1_gt.T).T
    uv_gt = proj_gt[:, :2] / proj_gt[:, 2:3]

    # Project into cam1 with predicted relative pose → 2D pred
    pts_cam1_pred = (T_rel_pred[:3, :3] @ pts_cam0.T).T + T_rel_pred[:3, 3]
    valid2 = pts_cam1_pred[:, 2] > 0
    if valid2.sum() < 4:
        return {}

    proj_pred = (K @ pts_cam1_pred[valid2].T).T
    uv_pred = proj_pred[:, :2] / proj_pred[:, 2:3]

    # Reprojection error in pixels
    err_px = np.linalg.norm(uv_pred - uv_gt[valid2], axis=1)

    W, H = image_size
    diag = np.sqrt(W ** 2 + H ** 2)

    return {
        "vcre_median_px": float(np.median(err_px)),
        "vcre_mean_px": float(np.mean(err_px)),
        "vcre_pct": float(np.median(err_px) / diag * 100),
    }
