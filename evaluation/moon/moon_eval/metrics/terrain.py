"""
moon_eval/metrics/terrain.py — Terrain and relief quality metrics.

Designed for lunar surface evaluation where detecting relief features
(craters, slopes, ridges) matters as much as absolute accuracy.

Functions:
  compute_slope_map      — Slope (°) and aspect (°) from depth map via Sobel
  compute_slope_metrics  — Slope MAE, RMSE, correlation
  compute_hda_metrics    — Hazard Detection & Avoidance slope classification
  compute_curvature_maps — Mean and Gaussian curvature
  compute_roughness_map  — Local roughness (std in sliding window)
  compute_relief_metrics — Full relief/terrain feature quality suite
"""

import numpy as np
from scipy.ndimage import sobel, uniform_filter
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
from typing import Dict, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Slope and aspect
# ─────────────────────────────────────────────────────────────────────────────

def compute_slope_map(
    depth_map: np.ndarray,
    pixel_size: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute slope (°) and aspect (°) from a depth map using Sobel filters.

    Slope is the magnitude of the gradient direction in 3D:
        slope = arctan(sqrt(gx² + gy²))   [degrees]

    Aspect is the compass direction (0=East, CCW):
        aspect = arctan2(-gy, gx)          [degrees, 0–360]

    Args:
        depth_map  : (H, W) depth values
        pixel_size : physical size of one pixel (for gradient scaling)

    Returns:
        slope_deg  : (H, W) slope in degrees
        aspect_deg : (H, W) aspect in degrees [0, 360)
    """
    gx = sobel(depth_map, axis=1) / (8.0 * pixel_size)
    gy = sobel(depth_map, axis=0) / (8.0 * pixel_size)

    slope_deg = np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2)))

    aspect_rad = np.arctan2(-gy, gx)
    aspect_deg = (np.degrees(aspect_rad) + 360.0) % 360.0

    return slope_deg, aspect_deg


# ─────────────────────────────────────────────────────────────────────────────
# Slope metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_slope_metrics(
    slope_pred: np.ndarray,
    slope_gt: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    """Slope correlation, MAE, RMSE, and mean values.

    Args:
        slope_pred : (H, W) predicted slope in degrees
        slope_gt   : (H, W) GT slope in degrees
        mask       : (H, W) bool, valid pixels

    Returns:
        dict with keys: slope_corr, slope_mae, slope_rmse,
                        slope_gt_mean, slope_pred_mean
    """
    valid = mask & np.isfinite(slope_gt) & np.isfinite(slope_pred)
    if valid.sum() < 10:
        return {}

    sg = slope_gt[valid]
    sp = slope_pred[valid]
    result: Dict[str, float] = {
        "slope_mae": float(np.mean(np.abs(sp - sg))),
        "slope_rmse": float(np.sqrt(np.mean((sp - sg) ** 2))),
        "slope_gt_mean": float(sg.mean()),
        "slope_pred_mean": float(sp.mean()),
    }

    if len(sg) > 1:
        result["slope_corr"] = float(np.corrcoef(sg, sp)[0, 1])
    else:
        result["slope_corr"] = float("nan")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# HDA (Hazard Detection & Avoidance) metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_hda_metrics(
    slope_pred: np.ndarray,
    slope_gt: np.ndarray,
    mask: np.ndarray,
    thresholds: Tuple[int, ...] = (5, 10, 15, 20),
) -> Dict[str, float]:
    """Hazard Detection & Avoidance classification metrics.

    For each slope threshold t, classifies pixels as safe (slope < t) or
    unsafe (slope ≥ t) and computes agreement, false alarm rate, and miss rate.

        agree      : % of pixels where both pred and GT agree on safe/unsafe
        false_alarm: % of GT-safe pixels wrongly predicted as unsafe
                     (landing aborted unnecessarily)
        miss       : % of GT-unsafe pixels wrongly predicted as safe
                     (dangerous — the model misses a real hazard)

    Args:
        slope_pred : (H, W) predicted slope in degrees
        slope_gt   : (H, W) GT slope in degrees
        mask       : (H, W) bool
        thresholds : slope thresholds in degrees

    Returns:
        flat dict with keys: slope_agree_{t}deg, slope_false_alarm_{t}deg,
                             slope_miss_{t}deg  for each threshold
    """
    valid = mask & np.isfinite(slope_gt) & np.isfinite(slope_pred)
    if valid.sum() < 10:
        return {}

    sg = slope_gt[valid]
    sp = slope_pred[valid]
    result: Dict[str, float] = {}

    for t in thresholds:
        gt_safe = sg < t
        pred_safe = sp < t
        agree = gt_safe == pred_safe
        result[f"slope_agree_{t}deg"] = float(agree.mean() * 100)

        if gt_safe.sum() > 0:
            false_alarm = ((~pred_safe) & gt_safe).sum() / gt_safe.sum()
            result[f"slope_false_alarm_{t}deg"] = float(false_alarm * 100)

        if (~gt_safe).sum() > 0:
            miss = (pred_safe & (~gt_safe)).sum() / (~gt_safe).sum()
            result[f"slope_miss_{t}deg"] = float(miss * 100)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Curvature
# ─────────────────────────────────────────────────────────────────────────────

def compute_curvature_maps(
    depth_map: np.ndarray,
    pixel_size: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and Gaussian curvature of a depth surface.

    Uses the standard differential geometry formulas for a height field z(x,y):
        p = dz/dx,  q = dz/dy
        r = d²z/dx², s = d²z/dxdy, t = d²z/dy²

        mean_K    = -((1+q²)r - 2pqs + (1+p²)t) / (2*(1+p²+q²)^(3/2))
        gauss_K   = (rt - s²) / (1+p²+q²)²

    Args:
        depth_map  : (H, W) depth values
        pixel_size : physical pixel size for gradient scaling

    Returns:
        mean_curv  : (H, W) mean curvature
        gauss_curv : (H, W) Gaussian curvature
    """
    dz_dx = np.gradient(depth_map, axis=1) / pixel_size
    dz_dy = np.gradient(depth_map, axis=0) / pixel_size
    d2z_dx2 = np.gradient(dz_dx, axis=1) / pixel_size
    d2z_dy2 = np.gradient(dz_dy, axis=0) / pixel_size
    d2z_dxdy = np.gradient(dz_dx, axis=0) / pixel_size

    p, q = dz_dx, dz_dy
    r, s, t = d2z_dx2, d2z_dxdy, d2z_dy2

    denom_mean = 2.0 * (1.0 + p ** 2 + q ** 2) ** 1.5
    mean_curv = -((1.0 + q ** 2) * r - 2.0 * p * q * s + (1.0 + p ** 2) * t) / (
        denom_mean + 1e-12
    )
    denom_gauss = (1.0 + p ** 2 + q ** 2) ** 2
    gauss_curv = (r * t - s ** 2) / (denom_gauss + 1e-12)

    return mean_curv, gauss_curv


# ─────────────────────────────────────────────────────────────────────────────
# Roughness
# ─────────────────────────────────────────────────────────────────────────────

def compute_roughness_map(
    depth_map: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """Local roughness = local standard deviation of depth in a sliding window.

    Uses the identity: Var(X) = E[X²] - E[X]²

    Args:
        depth_map   : (H, W)
        window_size : half-window size for uniform_filter

    Returns:
        roughness   : (H, W) local std of depth
    """
    mean = uniform_filter(depth_map.astype(float), size=window_size)
    mean_sq = uniform_filter(depth_map.astype(float) ** 2, size=window_size)
    variance = np.maximum(mean_sq - mean ** 2, 0.0)
    return np.sqrt(variance)


# ─────────────────────────────────────────────────────────────────────────────
# Full relief / terrain feature metrics
# ─────────────────────────────────────────────────────────────────────────────

def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection over Union of two boolean masks."""
    intersection = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    if union == 0:
        return float("nan")
    return float(intersection / union)


def compute_relief_metrics(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    mask: np.ndarray,
    curvature_threshold: float = 0.0,   # 0 = auto (80th percentile)
    roughness_window: int = 5,
) -> Dict[str, float]:
    """Comprehensive terrain relief quality metrics.

    Goes beyond absolute accuracy to ask: *does the model capture terrain
    features like craters, ridges, and slopes correctly?*

    The metrics are designed to be scale-invariant or structure-based:
    - **Roughness correlation**: does the model preserve local texture?
    - **Curvature correlation**: are concave/convex features detected?
    - **Crater IoU**: are crater locations correct? (curvature binarization)
    - **Scale-invariant relief**: detrend and correlate (removes offset/scale)
    - **Slope SSIM**: structural similarity of gradient maps
    - **Aspect histogram correlation**: do slope directions match?
    - **Topographic prominence**: are peaks/valleys in the right places?

    Args:
        depth_pred          : (H, W) aligned predicted depth
        depth_gt            : (H, W) GT depth
        mask                : (H, W) bool, valid pixels
        curvature_threshold : binarization threshold for craters
                              0 (default) → use 80th percentile of |curv_gt|
        roughness_window    : window size for roughness computation

    Returns:
        flat dict with all terrain metric values (NaN if computation fails)
    """
    result: Dict[str, float] = {}
    mask = mask.astype(bool)
    H, W = depth_gt.shape

    if mask.sum() < 50:
        return result

    # ── Roughness ─────────────────────────────────────────────────────────────
    try:
        rough_gt = compute_roughness_map(depth_gt, roughness_window)
        rough_pred = compute_roughness_map(depth_pred, roughness_window)
        r_gt = rough_gt[mask]
        r_pred = rough_pred[mask]
        if len(r_gt) > 1:
            result["roughness_corr"] = float(np.corrcoef(r_gt, r_pred)[0, 1])
            result["roughness_mae"] = float(np.mean(np.abs(r_pred - r_gt)))
    except Exception:
        pass

    # ── Curvature ────────────────────────────────────────────────────────────
    try:
        mc_gt, _ = compute_curvature_maps(depth_gt)
        mc_pred, _ = compute_curvature_maps(depth_pred)
        mc_gt_v = mc_gt[mask]
        mc_pred_v = mc_pred[mask]
        # Remove extreme outliers (top/bottom 1%)
        lo, hi = np.percentile(mc_gt_v, 1), np.percentile(mc_gt_v, 99)
        valid_curv = (mc_gt_v >= lo) & (mc_gt_v <= hi)
        if valid_curv.sum() > 10:
            r, _ = pearsonr(mc_gt_v[valid_curv], mc_pred_v[valid_curv])
            result["curvature_corr"] = float(r)
            result["curvature_mae"] = float(
                np.mean(np.abs(mc_pred_v[valid_curv] - mc_gt_v[valid_curv]))
            )

        # ── Crater IoU (curvature-based) ──────────────────────────────────────
        # Determine threshold adaptively
        abs_mc_gt = np.abs(mc_gt_v)
        if curvature_threshold <= 0:
            ct = float(np.percentile(abs_mc_gt, 80))
            ct = max(ct, 0.005)
        else:
            ct = curvature_threshold

        # Rim regions: strong positive curvature (convex, crater rim)
        rim_gt = (mc_gt > ct) & mask
        rim_pred = (mc_pred > ct) & mask
        result["crater_rim_iou"] = _iou(rim_gt, rim_pred)

        # Interior regions: strong negative curvature (concave, crater floor)
        interior_gt = (mc_gt < -ct) & mask
        interior_pred = (mc_pred < -ct) & mask
        result["crater_interior_iou"] = _iou(interior_gt, interior_pred)

        # Combined: any strong curvature feature
        combined_gt = (np.abs(mc_gt) > ct) & mask
        combined_pred = (np.abs(mc_pred) > ct) & mask
        result["crater_combined_iou"] = _iou(combined_gt, combined_pred)

    except Exception:
        pass

    # ── Scale-invariant relief (detrended depth correlation) ─────────────────
    try:
        ys, xs = np.where(mask)
        zs_gt = depth_gt[mask]
        # Fit plane to GT: z = ax + by + c
        A = np.column_stack([xs, ys, np.ones(len(xs))])
        coeffs, _, _, _ = np.linalg.lstsq(A, zs_gt, rcond=None)
        plane_gt = coeffs[0] * xs + coeffs[1] * ys + coeffs[2]

        detrended_gt = zs_gt - plane_gt
        detrended_pred = depth_pred[mask] - plane_gt  # use GT plane for both

        if len(detrended_gt) > 10:
            r, _ = pearsonr(detrended_gt, detrended_pred)
            result["relief_corr"] = float(r)
    except Exception:
        pass

    # ── Slope SSIM ───────────────────────────────────────────────────────────
    try:
        slope_gt, aspect_gt = compute_slope_map(depth_gt)
        slope_pred, aspect_pred = compute_slope_map(depth_pred)

        # Normalize slope maps to [0, 1] using GT max
        slope_max = float(np.nanpercentile(slope_gt[mask], 99))
        if slope_max > 0:
            slope_gt_n = np.clip(slope_gt / slope_max, 0, 1)
            slope_pred_n = np.clip(slope_pred / slope_max, 0, 1)

            # Crop to bounding box of mask to reduce zero-padding effect
            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]
            r0, r1 = rows[0], rows[-1] + 1
            c0, c1 = cols[0], cols[-1] + 1

            patch_gt = slope_gt_n[r0:r1, c0:c1]
            patch_pred = slope_pred_n[r0:r1, c0:c1]

            if min(patch_gt.shape) >= 7:
                win = min(7, min(patch_gt.shape) // 2 * 2 - 1)
                win = max(win, 3)
                slope_ssim = ssim(
                    patch_gt, patch_pred, data_range=1.0, win_size=win
                )
                result["slope_ssim"] = float(slope_ssim)
    except Exception:
        pass

    # ── Aspect histogram correlation ─────────────────────────────────────────
    try:
        if "slope_gt" not in dir():
            slope_gt, aspect_gt = compute_slope_map(depth_gt)
            slope_pred, aspect_pred = compute_slope_map(depth_pred)

        # Only use pixels where slope is meaningful (> 2°) in both
        valid_aspect = (
            mask
            & np.isfinite(aspect_gt)
            & np.isfinite(aspect_pred)
            & (slope_gt > 2.0)
            & (slope_pred > 2.0)
        )
        if valid_aspect.sum() >= 100:
            n_bins = 16
            hist_gt, _ = np.histogram(aspect_gt[valid_aspect], bins=n_bins,
                                       range=(0, 360))
            hist_pred, _ = np.histogram(aspect_pred[valid_aspect], bins=n_bins,
                                         range=(0, 360))
            # Normalize
            hist_gt = hist_gt.astype(float) / (hist_gt.sum() + 1e-12)
            hist_pred = hist_pred.astype(float) / (hist_pred.sum() + 1e-12)
            if hist_gt.std() > 0 and hist_pred.std() > 0:
                result["aspect_hist_corr"] = float(
                    np.corrcoef(hist_gt, hist_pred)[0, 1]
                )
        else:
            result["aspect_hist_corr"] = float("nan")
    except Exception:
        pass

    # ── Topographic prominence (peaks / valleys) ──────────────────────────────
    try:
        from skimage.feature import peak_local_max

        # Use masked depth (set invalid to NaN then fill for peak detection)
        depth_gt_masked = depth_gt.copy().astype(float)
        depth_gt_masked[~mask] = np.nan
        depth_pred_masked = depth_pred.copy().astype(float)
        depth_pred_masked[~mask] = np.nan

        # Fill NaN with local mean for peak_local_max (which needs finite array)
        from scipy.ndimage import generic_filter
        def nanmean_fill(arr):
            """Replace NaN with local nanmean."""
            filled = arr.copy()
            nan_mask = np.isnan(filled)
            if nan_mask.any():
                col_means = np.nanmean(filled, axis=0)
                col_means = np.where(np.isnan(col_means), np.nanmean(filled), col_means)
                filled[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
            return filled

        dgt_filled = nanmean_fill(depth_gt_masked)
        dpred_filled = nanmean_fill(depth_pred_masked)

        min_dist = 8

        # Peaks (local maxima = ridges, mounds)
        peaks_gt = peak_local_max(dgt_filled, min_distance=min_dist,
                                  exclude_border=False)
        peaks_pred = peak_local_max(dpred_filled, min_distance=min_dist,
                                    exclude_border=False)

        # Valleys (local minima = crater interiors) — negate depth
        valleys_gt = peak_local_max(-dgt_filled, min_distance=min_dist,
                                    exclude_border=False)
        valleys_pred = peak_local_max(-dpred_filled, min_distance=min_dist,
                                      exclude_border=False)

        result["peak_count_gt"] = int(len(peaks_gt))
        result["peak_count_pred"] = int(len(peaks_pred))
        result["valley_count_gt"] = int(len(valleys_gt))
        result["valley_count_pred"] = int(len(valleys_pred))

        recall_radius = 10  # pixels

        def _location_recall(
            gt_coords: np.ndarray, pred_coords: np.ndarray, radius: float
        ) -> float:
            if len(gt_coords) == 0:
                return float("nan")
            if len(pred_coords) == 0:
                return 0.0
            from scipy.spatial import cKDTree as _KD
            tree = _KD(pred_coords)
            dists, _ = tree.query(gt_coords, k=1)
            return float((dists < radius).mean() * 100)

        result["peak_location_recall"] = _location_recall(
            peaks_gt, peaks_pred, recall_radius
        )
        result["valley_location_recall"] = _location_recall(
            valleys_gt, valleys_pred, recall_radius
        )

    except ImportError:
        pass  # skimage.feature not available
    except Exception:
        pass

    return result
