"""
moon_eval/alignment.py — Sim(3) alignment utilities + ICP refinement.

Functions:
  umeyama_sim3            — Closed-form Sim(3) (Umeyama 1991)
  ransac_umeyama_sim3     — RANSAC-robust Umeyama with outlier rejection
  align_sim3_ransac       — Open3D RANSAC-based Sim(3)
  apply_sim3              — Apply 4×4 Sim(3) transform to points
  build_T4x4_from_srt     — Compose scale, rotation, translation into 4×4
  improved_gt_alignment   — Multi-stage: RANSAC-Umeyama → ICP refinement (→ fallback)
"""

import numpy as np
import open3d as o3d
from typing import Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Core Sim(3) primitives  (ported from eval_per_view.py)
# ─────────────────────────────────────────────────────────────────────────────

def umeyama_sim3(
    src: np.ndarray,
    dst: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Closed-form Sim(3) alignment (Umeyama 1991).

    Finds scale s, rotation R, translation t such that:
        dst ≈ s * R @ src + t

    Args:
        src : (N, 3) source points
        dst : (N, 3) destination points

    Returns:
        s   : scalar scale
        R   : (3, 3) rotation matrix
        t   : (3,) translation vector
    """
    n, d = src.shape
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    var_src = np.sum(src_c ** 2) / n
    cov = (dst_c.T @ src_c) / n
    U, S, Vt = np.linalg.svd(cov)
    det_sign = np.linalg.det(U) * np.linalg.det(Vt)
    D = np.eye(d)
    if det_sign < 0:
        D[-1, -1] = -1
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / var_src
    t = mu_dst - s * R @ mu_src
    return s, R, t


def ransac_umeyama_sim3(
    src: np.ndarray,
    dst: np.ndarray,
    n_iter: int = 200,
    sample_size: int = 500,
    inlier_fraction_target: float = 0.5,
    verbose: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """RANSAC-robust Umeyama Sim(3): reject outliers before final fit.

    Procedure:
      1. Random subsample → Umeyama on subsample.
      2. Apply to ALL points, measure residuals.
      3. Count inliers (residual < adaptive threshold).
      4. Keep the best (most inliers) hypothesis.
      5. Re-fit Umeyama on ALL inliers for the best hypothesis.

    The adaptive threshold is median_residual * 3 (MAD-style) for each
    hypothesis, clipped to [p10, p75] of the residuals.

    Args:
        src : (N, 3) source points
        dst : (N, 3) destination points
        n_iter           : number of RANSAC iterations
        sample_size      : points per sample (>= 4 for Sim(3))
        inlier_fraction_target : early-stop if this fraction is reached
        verbose          : print info

    Returns:
        s           : scale
        R           : (3, 3) rotation
        t           : (3,) translation
        inlier_mask : (N,) bool mask of inlier correspondences
    """
    n = len(src)
    sample_size = min(sample_size, n)
    rng = np.random.RandomState(42)

    best_n_inliers = 0
    best_inlier_mask = np.ones(n, dtype=bool)
    best_s, best_R, best_t = None, None, None

    for it in range(n_iter):
        # Random subsample
        idx = rng.choice(n, sample_size, replace=False)
        s_h, R_h, t_h = umeyama_sim3(src[idx], dst[idx])

        # Apply to ALL points
        aligned = s_h * (R_h @ src.T).T + t_h
        residuals = np.linalg.norm(aligned - dst, axis=1)

        # Adaptive threshold: 3 * median (MAD-style), clipped
        med_res = np.median(residuals)
        thresh = np.clip(med_res * 3.0,
                         np.percentile(residuals, 10),
                         np.percentile(residuals, 75))
        inlier_mask = residuals < thresh
        n_inliers = inlier_mask.sum()

        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_inlier_mask = inlier_mask
            best_s, best_R, best_t = s_h, R_h, t_h

            if n_inliers >= n * inlier_fraction_target and it >= 20:
                break  # Good enough, early stop

    # Re-fit on all inliers for optimal estimate
    if best_n_inliers >= 10:
        best_s, best_R, best_t = umeyama_sim3(
            src[best_inlier_mask], dst[best_inlier_mask]
        )

    if verbose:
        print(f"  [ransac_umeyama] {best_n_inliers}/{n} inliers "
              f"({100*best_n_inliers/n:.1f}%), scale={best_s:.4f}")

    return best_s, best_R, best_t, best_inlier_mask


def align_sim3_ransac(
    src: np.ndarray,
    dst: np.ndarray,
    Nsub: int = 5000,
) -> Tuple[np.ndarray, float, int]:
    """RANSAC-based Sim(3) alignment using Open3D.

    Args:
        src  : (N, 3) source points
        dst  : (N, 3) destination (target) points
        Nsub : max number of points to use (random subsample)

    Returns:
        T        : (4, 4) Sim(3) transform matrix
        scale    : estimated scale (cbrt of det of upper-left 3x3)
        n_inliers: number of RANSAC inliers
    """
    n = len(src)
    Nsub = min(Nsub, n)
    rng = np.random.RandomState(42)
    idxs = rng.choice(n, Nsub, replace=False)

    pcd_A = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src[idxs]))
    pcd_B = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst[idxs]))
    corr = o3d.utility.Vector2iVector(
        np.stack([np.arange(Nsub), np.arange(Nsub)], axis=1)
    )

    est = o3d.pipelines.registration.TransformationEstimationPointToPoint(
        with_scaling=True
    )
    res = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        pcd_A, pcd_B, corr, 1e5,
        estimation_method=est,
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(1e5)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )

    T = res.transformation
    M = T[:3, :3]
    scale = float(np.cbrt(np.linalg.det(M)))
    return T, scale, len(res.correspondence_set)


def apply_sim3(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4×4 Sim(3) transform to (N, 3) points.

    Returns transformed points of shape (N, 3).
    """
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def build_T4x4_from_srt(
    s: float,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Compose scale s, rotation R, translation t into a 4×4 matrix.

    The upper-left 3×3 block is s*R and the last column is [t; 1].

    Returns:
        T : (4, 4) Sim(3) transform
    """
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


# ─────────────────────────────────────────────────────────────────────────────
# Improved alignment: Umeyama → ICP PointToPlane refinement
# ─────────────────────────────────────────────────────────────────────────────

def improved_gt_alignment(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
    conf: Optional[np.ndarray] = None,
    voxel_size_factor: float = 0.02,
    icp_max_iter: int = 100,
    icp_fitness_min: float = 0.02,
    verbose: bool = False,
) -> Tuple[np.ndarray, float, str]:
    """Multi-stage robust alignment: RANSAC-Umeyama → ICP refinement (→ fallback).

    Pipeline:
      1. RANSAC-Umeyama Sim(3) with outlier rejection.
         - Runs many random subsamples of correspondences.
         - For each: compute Umeyama → count inliers → keep best.
         - Re-fit Umeyama on all inliers only (outlier-free).
         This is critical for noisy students (S2, S5, …).
      2. (Optional) If conf is provided, try also a confidence-weighted
         alignment on top-50% confident points and keep whichever is better.
      3. Apply Umeyama transform to pred_pts → pred_init.
      4. Voxel-downsample both clouds, estimate normals.
      5. ICP PointToPlane (rigid) refines residual misalignment.
         ICP threshold = max(3 * median_error, 2 * voxel_size).
      6. If ICP fitness < icp_fitness_min → try PointToPoint ICP.
      7. If both ICP fail → fall back to RANSAC-Umeyama result (no ICP).
      8. Compose final T = T_icp @ T_umeyama.

    NOTE on scale: ICP is rigid (no scale). The scale from Umeyama is baked
    into T_umeyama[:3,:3]. ICP refines rotation & translation around the
    already-scaled space.

    Args:
        pred_pts          : (N, 3) valid predicted points (pre-filtered)
        gt_pts            : (N, 3) corresponding GT points (pre-filtered)
        conf              : (N,) optional confidence weights per point
        voxel_size_factor : fraction of terrain span used for voxel size
        icp_max_iter      : ICP max iterations
        icp_fitness_min   : minimum fitness to consider ICP successful
        verbose           : print alignment info

    Returns:
        T_final     : (4, 4) final Sim(3) transform  (pred → GT)
        scale       : estimated scale factor
        method_used : 'ransac_umeyama+icp' | 'ransac_umeyama' |
                      'conf_umeyama+icp' | 'ransac_fallback'
    """
    n = len(pred_pts)
    if n < 10:
        raise ValueError(f"Too few points for alignment: {n}")

    # ── Stage 1: RANSAC-Umeyama (outlier-robust) ────────────────────────────
    # For large point clouds, subsample to keep RANSAC fast
    max_pts_for_ransac = min(n, 50000)
    if n > max_pts_for_ransac:
        rng_sub = np.random.RandomState(0)
        sub_idx = rng_sub.choice(n, max_pts_for_ransac, replace=False)
        src_sub, dst_sub = pred_pts[sub_idx], gt_pts[sub_idx]
    else:
        src_sub, dst_sub = pred_pts, gt_pts

    s_r, R_r, t_r, inlier_mask_sub = ransac_umeyama_sim3(
        src_sub, dst_sub,
        n_iter=300,
        sample_size=min(800, len(src_sub)),
        verbose=verbose,
    )
    T_ransac = build_T4x4_from_srt(s_r, R_r, t_r)
    pred_init_r = apply_sim3(pred_pts, T_ransac)
    med_err_r = float(np.median(np.linalg.norm(pred_init_r - gt_pts, axis=1)))

    if verbose:
        inl_pct = 100 * inlier_mask_sub.sum() / len(inlier_mask_sub)
        print(f"  [align] RANSAC-Umeyama: s={s_r:.4f}, med_err={med_err_r:.4f}, "
              f"inliers={inl_pct:.1f}%")

    # ── Stage 1b: Confidence-weighted alignment (if conf provided) ──────────
    # Try alignment on only high-confidence points — may beat RANSAC-Umeyama
    T_best = T_ransac
    s_best = s_r
    med_err_best = med_err_r
    best_method_prefix = "ransac_umeyama"

    if conf is not None and len(conf) == n:
        try:
            # Take top-50% confident points
            conf_thresh = np.percentile(conf, 50)
            hi_mask = conf >= conf_thresh
            n_hi = hi_mask.sum()
            if n_hi >= 100:
                s_c, R_c, t_c = umeyama_sim3(pred_pts[hi_mask], gt_pts[hi_mask])
                T_conf = build_T4x4_from_srt(s_c, R_c, t_c)
                pred_init_c = apply_sim3(pred_pts, T_conf)
                med_err_c = float(np.median(np.linalg.norm(pred_init_c - gt_pts, axis=1)))
                if verbose:
                    print(f"  [align] Conf-Umeyama: s={s_c:.4f}, med_err={med_err_c:.4f} "
                          f"(top {n_hi} pts)")
                if med_err_c < med_err_best:
                    T_best = T_conf
                    s_best = s_c
                    med_err_best = med_err_c
                    best_method_prefix = "conf_umeyama"
        except Exception:
            pass  # Confidence-weighted alignment failed, keep RANSAC result

    pred_init = apply_sim3(pred_pts, T_best)

    # ── Stage 2: Prepare ICP ─────────────────────────────────────────────────
    terrain_span = float(np.linalg.norm(np.ptp(gt_pts, axis=0)))
    voxel_size = max(terrain_span * voxel_size_factor, 1e-3)
    median_err = float(np.median(np.linalg.norm(pred_init - gt_pts, axis=1)))
    icp_thresh = max(median_err * 3.0, voxel_size * 2.0)

    icp_ok = False
    try:
        src_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred_init))
        dst_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_pts))

        # Voxel-downsample for normal estimation
        src_ds = src_pcd.voxel_down_sample(voxel_size)
        dst_ds = dst_pcd.voxel_down_sample(voxel_size)

        normal_radius = voxel_size * 2.0
        normal_nn = 30
        src_ds.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_nn)
        )
        dst_ds.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_nn)
        )

        # ── Stage 3a: ICP PointToPlane ─────────────────────────────────────────
        icp_res = o3d.pipelines.registration.registration_icp(
            src_ds,
            dst_ds,
            icp_thresh,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=icp_max_iter
            ),
        )

        if verbose:
            print(
                f"  [align] ICP P2Plane: fitness={icp_res.fitness:.4f}, "
                f"rmse={icp_res.inlier_rmse:.4f}, thresh={icp_thresh:.4f}"
            )

        if icp_res.fitness >= icp_fitness_min:
            T_icp = icp_res.transformation
            T_final = T_icp @ T_best
            scale = float(np.cbrt(np.linalg.det(T_final[:3, :3])))
            method = f"{best_method_prefix}+icp"
            icp_ok = True
        else:
            # ── Stage 3b: Fallback — ICP PointToPoint (more tolerant) ──────────
            if verbose:
                print(f"  [align] P2Plane ICP low fitness, trying PointToPoint...")
            icp_res2 = o3d.pipelines.registration.registration_icp(
                src_ds,
                dst_ds,
                icp_thresh * 1.5,  # Slightly more tolerant
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=icp_max_iter
                ),
            )
            if verbose:
                print(
                    f"  [align] ICP P2Point: fitness={icp_res2.fitness:.4f}, "
                    f"rmse={icp_res2.inlier_rmse:.4f}"
                )
            if icp_res2.fitness >= icp_fitness_min:
                T_icp = icp_res2.transformation
                T_final = T_icp @ T_best
                scale = float(np.cbrt(np.linalg.det(T_final[:3, :3])))
                method = f"{best_method_prefix}+icp_p2p"
                icp_ok = True

    except Exception as e:
        if verbose:
            print(f"  [align] ICP failed ({e})")

    if not icp_ok:
        # ── ICP failed — use RANSAC-Umeyama result directly ──────────────────
        # This is already much better than old code which fell back to
        # Open3D RANSAC Sim(3) (unreliable for large scale gaps).
        T_final = T_best
        scale = s_best
        method = best_method_prefix

        if verbose:
            print(f"  [align] Using {method} without ICP (median_err={med_err_best:.2f})")

    return T_final, scale, method
