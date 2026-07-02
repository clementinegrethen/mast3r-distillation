"""
moon_eval/metrics/classic.py — Classic 3D reconstruction metrics.

Functions:
  compute_accuracy_completeness — directional NN distances (pred→GT, GT→pred)
  compute_depth_metrics         — MAE, RMSE, Pearson, SILog, delta thresholds
  compute_3d_metrics            — RMSE, MAE, Pearson-Z on aligned 3D points
  compute_profile_metrics       — Central-row profile correlation and MAE
  compute_overlap_consistency   — Inter-view overlap quality metric
"""

import numpy as np
from scipy.stats import pearsonr
from scipy.spatial import cKDTree
from skimage.metrics import structural_similarity as _ssim
from typing import Dict, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy / Completeness / Chamfer
# ─────────────────────────────────────────────────────────────────────────────

def compute_accuracy_completeness(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
    max_pts: int = 20000,
    thresholds: Tuple[float, ...] = (0.5, 1.0, 2.0),
) -> Dict[str, float]:
    """Nearest-neighbour accuracy, completeness, and Chamfer distance.

    Directional semantics:
        accuracy    = mean distance from each pred point to its nearest GT point
                      (measures how close the prediction is to the GT surface)
        completeness = mean distance from each GT point to its nearest pred point
                      (measures how much of the GT surface is covered)
        chamfer     = (accuracy + completeness) / 2

    Additionally computes:
        acc_pct_under_{t}   : % of pred pts within t units of GT (for t in thresholds)
        compl_pct_under_{t} : % of GT pts within t units of pred

    Args:
        pred_pts   : (N, 3) valid aligned predicted points
        gt_pts     : (M, 3) valid GT points
        max_pts    : subsample both sets to at most max_pts (random, seeded)
        thresholds : distance thresholds for recall-style percentages

    Returns:
        dict with keys: accuracy, completeness, chamfer,
                        acc_median, compl_median,
                        acc_pct_under_{t}, compl_pct_under_{t} per threshold
    """
    rng = np.random.RandomState(42)
    if len(pred_pts) > max_pts:
        pred_pts = pred_pts[rng.choice(len(pred_pts), max_pts, replace=False)]
    if len(gt_pts) > max_pts:
        gt_pts = gt_pts[rng.choice(len(gt_pts), max_pts, replace=False)]

    tree_gt = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)

    # pred → GT  (accuracy)
    dist_pred2gt, _ = tree_gt.query(pred_pts, k=1)
    # GT → pred  (completeness)
    dist_gt2pred, _ = tree_pred.query(gt_pts, k=1)

    acc = float(np.mean(dist_pred2gt))
    compl = float(np.mean(dist_gt2pred))
    chamfer = (acc + compl) / 2.0

    result = {
        "accuracy": acc,
        "completeness": compl,
        "chamfer": chamfer,
        "acc_median": float(np.median(dist_pred2gt)),
        "compl_median": float(np.median(dist_gt2pred)),
    }

    for t in thresholds:
        result[f"acc_pct_under_{t}"] = float((dist_pred2gt < t).mean() * 100)
        result[f"compl_pct_under_{t}"] = float((dist_gt2pred < t).mean() * 100)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Depth map metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_depth_metrics(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    """2D depth map evaluation metrics.

    Includes standard metrics used in monocular depth estimation literature:
      - depth_mae    : mean absolute error
      - depth_rmse   : root mean square error
      - depth_pearson: Pearson correlation coefficient
      - depth_ssim   : SSIM with adaptive data_range = max(GT ∪ pred) − min
                       (same formula as MASt3REvaluator.ssim_adaptive)
      - silog        : scale-invariant log error (std of log-ratio)
                       Formula: sqrt(mean(d²) - mean(d)²) where d = log(pred/gt)
      - delta1/delta2/delta3: % pixels where max(pred/gt, gt/pred) < 1.25^k

    Args:
        pred_depth : (H, W) predicted depth values  (must be 2-D for SSIM)
        gt_depth   : (H, W) GT depth values
        mask       : (H, W) bool, True = valid pixel

    Returns:
        dict with keys: depth_mae, depth_rmse, depth_pearson, depth_ssim,
                        silog, delta1, delta2, delta3
    """
    # Keep 2-D shape for SSIM before ravelling
    pred_2d = np.asarray(pred_depth)
    gt_2d = np.asarray(gt_depth)
    mask_2d = np.asarray(mask).astype(bool)

    pred_depth = pred_2d.ravel()
    gt_depth = gt_2d.ravel()
    mask = mask_2d.ravel()

    if mask.sum() < 10:
        return {}

    pred_v = pred_depth[mask]
    gt_v = gt_depth[mask]

    result: Dict[str, float] = {}
    result["depth_mae"] = float(np.mean(np.abs(pred_v - gt_v)))
    result["depth_rmse"] = float(np.sqrt(np.mean((pred_v - gt_v) ** 2)))

    if len(gt_v) > 1:
        r, _ = pearsonr(gt_v, pred_v)
        result["depth_pearson"] = float(r)
    else:
        result["depth_pearson"] = float("nan")

    # Depth SSIM (adaptive data_range over valid pixels, applied to full map)
    if pred_2d.ndim == 2 and pred_2d.shape == gt_2d.shape:
        try:
            valid_gt = gt_2d[mask_2d]
            valid_pred = pred_2d[mask_2d]
            lo = min(float(valid_gt.min()), float(valid_pred.min()))
            hi = max(float(valid_gt.max()), float(valid_pred.max()))
            dr = hi - lo if hi > lo else 1.0
            win = min(7, min(pred_2d.shape) // 2 * 2 - 1)
            win = max(win, 3)
            s = _ssim(gt_2d, pred_2d, data_range=dr, win_size=win)
            result["depth_ssim"] = float(s)
        except Exception:
            pass

    # Scale-invariant log error (requires positive depths)
    pos_mask = (pred_v > 0) & (gt_v > 0)
    if pos_mask.sum() >= 10:
        d = np.log(pred_v[pos_mask]) - np.log(gt_v[pos_mask])
        silog = float(np.sqrt(np.mean(d ** 2) - np.mean(d) ** 2))
        result["silog"] = silog

        # Delta thresholds
        ratio = np.maximum(pred_v[pos_mask] / gt_v[pos_mask],
                           gt_v[pos_mask] / pred_v[pos_mask])
        result["delta1"] = float((ratio < 1.25).mean() * 100)
        result["delta2"] = float((ratio < 1.25 ** 2).mean() * 100)
        result["delta3"] = float((ratio < 1.25 ** 3).mean() * 100)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3D point-to-point metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_3d_metrics(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    """Point-to-point 3D metrics on aligned predictions.

    Both pred_pts and gt_pts should already be in the same coordinate frame
    (i.e., pred_pts has been Sim(3)-aligned to GT frame).

    Args:
        pred_pts : (N, 3) aligned predicted points
        gt_pts   : (N, 3) GT corresponding points
        mask     : (N,) bool, True = valid

    Returns:
        dict with keys: rmse, mae_3d, pearson_z
    """
    mask = mask.astype(bool)
    if mask.sum() < 10:
        return {}

    pred_v = pred_pts[mask]
    gt_v = gt_pts[mask]
    diff = pred_v - gt_v

    result: Dict[str, float] = {}
    result["rmse"] = float(np.sqrt((diff ** 2).mean()))
    result["mae_3d"] = float(np.mean(np.linalg.norm(diff, axis=1)))

    if len(gt_v) > 1:
        r, _ = pearsonr(gt_v[:, 2], pred_v[:, 2])
        result["pearson_z"] = float(r)
    else:
        result["pearson_z"] = float("nan")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Profile metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_profile_metrics(
    depth_pred_map: np.ndarray,
    depth_gt_map: np.ndarray,
    mask_2d: np.ndarray,
    n_rows: int = 1,
) -> Dict[str, float]:
    """Central-row profile correlation and MAE.

    Extracts one (or more) horizontal profiles from the depth maps and
    computes correlation and MAE. This captures whether the coarse terrain
    shape along a transect is preserved.

    Args:
        depth_pred_map : (H, W) predicted depth
        depth_gt_map   : (H, W) GT depth
        mask_2d        : (H, W) bool
        n_rows         : 1 → central row only; >1 → n evenly-spaced rows

    Returns:
        dict with keys: profile_mae, profile_corr
        For n_rows > 1 also adds: profile_mae_mean, profile_corr_mean,
                                    profile_corr_std
    """
    H, W = depth_gt_map.shape
    result: Dict[str, float] = {}

    if n_rows == 1:
        rows = [H // 2]
    else:
        rows = np.linspace(H // 4, 3 * H // 4, n_rows, dtype=int).tolist()

    maes, corrs = [], []
    for row in rows:
        valid_row = (
            mask_2d[row, :]
            & np.isfinite(depth_gt_map[row, :])
            & np.isfinite(depth_pred_map[row, :])
        )
        if valid_row.sum() < 10:
            continue
        gt_prof = depth_gt_map[row, valid_row]
        pred_prof = depth_pred_map[row, valid_row]
        maes.append(float(np.mean(np.abs(pred_prof - gt_prof))))
        if len(gt_prof) > 1:
            corrs.append(float(np.corrcoef(gt_prof, pred_prof)[0, 1]))

    if maes:
        result["profile_mae"] = maes[0] if n_rows == 1 else float(np.mean(maes))
    if corrs:
        result["profile_corr"] = corrs[0] if n_rows == 1 else float(np.mean(corrs))

    if n_rows > 1 and maes:
        result["profile_mae_mean"] = float(np.mean(maes))
        result["profile_corr_mean"] = float(np.mean(corrs)) if corrs else float("nan")
        result["profile_corr_std"] = float(np.std(corrs)) if len(corrs) > 1 else float("nan")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Overlap consistency
# ─────────────────────────────────────────────────────────────────────────────

def compute_overlap_consistency(
    pred_v0: np.ndarray,
    pred_v1: np.ndarray,
    terrain_span: float,
) -> Dict[str, float]:
    """Inter-view overlap quality metric.

    After scene-level Sim(3) alignment, both views should agree in their
    overlapping region. This function measures the residual discrepancy.

    Points from each view that are within 5% of terrain_span of the other
    view are considered 'overlapping'. Their mutual NN distances are reported.

    Args:
        pred_v0      : (N, 3) aligned predicted points from view 0
        pred_v1      : (M, 3) aligned predicted points from view 1
        terrain_span : norm of scene bounding box diagonal (for threshold scaling)

    Returns:
        dict with keys: overlap_n_v0, overlap_n_v1,
                        overlap_pct_v0, overlap_pct_v1,
                        overlap_consistency_median, overlap_consistency_mean,
                        overlap_consistency_pct
    """
    tree_v1 = cKDTree(pred_v1)
    dist_0to1, _ = tree_v1.query(pred_v0, k=1)
    tree_v0 = cKDTree(pred_v0)
    dist_1to0, _ = tree_v0.query(pred_v1, k=1)

    overlap_thresh = terrain_span * 0.05
    ov0 = dist_0to1 < overlap_thresh
    ov1 = dist_1to0 < overlap_thresh

    if ov0.sum() < 10 or ov1.sum() < 10:
        return {"overlap_n_v0": 0, "overlap_n_v1": 0, "overlap_consistency_median": float("nan")}

    overlap_dist = np.concatenate([dist_0to1[ov0], dist_1to0[ov1]])
    return {
        "overlap_n_v0": int(ov0.sum()),
        "overlap_n_v1": int(ov1.sum()),
        "overlap_pct_v0": float(ov0.mean() * 100),
        "overlap_pct_v1": float(ov1.mean() * 100),
        "overlap_consistency_median": float(np.median(overlap_dist)),
        "overlap_consistency_mean": float(np.mean(overlap_dist)),
        "overlap_consistency_pct": float(np.median(overlap_dist) / terrain_span * 100),
    }
