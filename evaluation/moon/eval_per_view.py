#!/usr/bin/env python3
"""
eval_per_view.py — Per-view evaluation for MASt3R Teacher + Students.

Three reconstruction modes:
  --mode raw           : Raw MASt3R output + per-view independent Sim(3) alignment
  --mode sparse_ga     : sparse_global_alignment (refine) + single scene-level Sim(3)
  --mode sparse_ga_depth : sparse_ga with depth optimisation (refine+depth)

Metrics computed per-view then aggregated:
  - Accuracy / Completeness / Chamfer distance
  - Slope correlation, MAE, RMSE + HDA detection (miss/false alarm)
  - Profile MAE/corr, Depth Pearson r
  - Overlap consistency (mismatch between v0 and v1)

Usage:
    python eval_per_view.py --mode sparse_ga --max_pairs 3 --gt_folders landing
    python eval_per_view.py --mode raw --max_pairs 5 --gt_folders nadir pitch landing
"""

import sys
import os
import argparse
import time
import json
import tempfile

import numpy as np
import torch
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from scipy.ndimage import sobel
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mast3r.utils.path_to_dust3r  # noqa
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from eval_emat import (
    STUDENT_CONFIGS,
    GT_FOLDERS,
    K_GT,
    load_teacher,
    load_students,
)
from MAST3RUtils import MAST3RUtils
from dust3r.utils.device import to_numpy
import OpenEXR
import Imath
import open3d as o3d


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_exr(path):
    ex = OpenEXR.InputFile(str(path))
    dw = ex.header()["dataWindow"]
    W0, H0 = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
    buf = ex.channel("Y", Imath.PixelType(Imath.PixelType.FLOAT))
    d = np.frombuffer(buf, dtype=np.float32).reshape(H0, W0)
    return d


def load_gt_view(gt_folder, img_stem, Hc=384, Wc=512):
    """Load GT depth + cam2world + intrinsics for one view.
    Returns pts_world (Hc*Wc, 3), depth_map (Hc, Wc), K_crop, T_w_c.
    """
    data = np.load(gt_folder / f"{img_stem}.npz")
    K_gt = data["intrinsics"]
    T_w_c = data["cam2world"]

    depth_full = read_exr(gt_folder / f"{img_stem}.exr")
    y0 = (depth_full.shape[0] - Hc) // 2
    x0 = (depth_full.shape[1] - Wc) // 2
    depth_map = depth_full[y0:y0 + Hc, x0:x0 + Wc]

    # Back-project to world
    u, v = np.meshgrid(np.arange(Wc), np.arange(Hc))
    Kc = K_gt.copy()
    Kc[0, 2] -= x0
    Kc[1, 2] -= y0

    z = depth_map
    x = (u - Kc[0, 2]) * z / Kc[0, 0]
    y = (v - Kc[1, 2]) * z / Kc[1, 1]
    pts_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    hom = np.concatenate([pts_cam, np.ones((Hc * Wc, 1))], axis=1).T
    pts_world = (T_w_c @ hom)[:3].T
    mask_valid = np.isfinite(pts_world).all(axis=1)

    return pts_world, depth_map, Kc, T_w_c, mask_valid


# ─────────────────────────────────────────────────────────────────────────────
# Per-view metric functions
# ─────────────────────────────────────────────────────────────────────────────

def umeyama_sim3(src, dst):
    """Closed-form Sim(3) alignment (Umeyama 1991)."""
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


def align_sim3_ransac(src, dst, Nsub=5000):
    """RANSAC-based Sim(3) alignment using Open3D."""
    n = len(src)
    Nsub = min(Nsub, n)
    rng = np.random.RandomState(42)
    idxs = rng.choice(n, Nsub, replace=False)

    pcd_A = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src[idxs]))
    pcd_B = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst[idxs]))
    corr = o3d.utility.Vector2iVector(np.stack([np.arange(Nsub), np.arange(Nsub)], axis=1))

    est = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
    res = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        pcd_A, pcd_B, corr, 1e5,
        estimation_method=est,
        ransac_n=4,
        checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(1e5)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )

    T = res.transformation
    M = T[:3, :3]
    scale = np.cbrt(np.linalg.det(M))
    return T, scale, len(res.correspondence_set)


def apply_sim3(pts, T):
    """Apply 4x4 Sim(3) transform to (N,3) points."""
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def compute_slope(depth_map, pixel_size=1.0):
    """Compute slope in degrees from a depth map."""
    grad_x = sobel(depth_map, axis=1) / (8 * pixel_size)
    grad_y = sobel(depth_map, axis=0) / (8 * pixel_size)
    slope_deg = np.degrees(np.arctan(np.sqrt(grad_x**2 + grad_y**2)))
    return slope_deg


def compute_accuracy_completeness(pred_pts, gt_pts, max_pts=20000):
    """Compute accuracy and completeness (nearest-neighbor distances).

    Accuracy: mean dist from pred to closest GT point
    Completeness: mean dist from GT to closest pred point
    """
    rng = np.random.RandomState(42)
    if len(pred_pts) > max_pts:
        pred_pts = pred_pts[rng.choice(len(pred_pts), max_pts, replace=False)]
    if len(gt_pts) > max_pts:
        gt_pts = gt_pts[rng.choice(len(gt_pts), max_pts, replace=False)]

    tree_gt = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)

    dist_pred2gt, _ = tree_gt.query(pred_pts, k=1)
    dist_gt2pred, _ = tree_pred.query(gt_pts, k=1)

    acc = np.mean(dist_pred2gt)
    compl = np.mean(dist_gt2pred)
    chamfer = (acc + compl) / 2.0

    return {
        "accuracy": float(acc),
        "completeness": float(compl),
        "chamfer": float(chamfer),
        "acc_median": float(np.median(dist_pred2gt)),
        "compl_median": float(np.median(dist_gt2pred)),
    }


def compute_per_view_metrics(pred_pts_aligned, gt_pts, depth_pred_map, depth_gt_map, mask_ok):
    """Compute all metrics for a single aligned view.

    Args:
        pred_pts_aligned: (H*W, 3) aligned predicted points in GT frame
        gt_pts: (H*W, 3) GT points in world frame
        depth_pred_map: (H, W) predicted depth (Z component of aligned pts)
        depth_gt_map: (H, W) GT depth
        mask_ok: (H*W,) boolean mask for valid points
    """
    m = {}

    pred_valid = pred_pts_aligned[mask_ok]
    gt_valid = gt_pts[mask_ok]

    # --- 3D point-to-point errors ---
    diff = pred_valid - gt_valid
    m["rmse"] = float(np.sqrt((diff ** 2).mean()))
    m["mae_3d"] = float(np.mean(np.linalg.norm(diff, axis=1)))

    # Pearson on Z
    m["pearson_z"], _ = pearsonr(gt_valid[:, 2], pred_valid[:, 2])
    m["pearson_z"] = float(m["pearson_z"])

    # --- Accuracy / Completeness / Chamfer ---
    ac_metrics = compute_accuracy_completeness(pred_valid, gt_valid)
    m.update(ac_metrics)

    # --- 2D depth map metrics ---
    H, W = depth_gt_map.shape
    mask_2d = mask_ok.reshape(H, W)
    gt_z = depth_gt_map[mask_2d]
    pred_z = depth_pred_map[mask_2d]

    if len(gt_z) > 10:
        m["depth_mae"] = float(np.mean(np.abs(pred_z - gt_z)))
        m["depth_rmse"] = float(np.sqrt(np.mean((pred_z - gt_z) ** 2)))
        m["depth_pearson"], _ = pearsonr(gt_z, pred_z)
        m["depth_pearson"] = float(m["depth_pearson"])

    # --- Slope metrics ---
    slope_gt = compute_slope(depth_gt_map)
    slope_pred = compute_slope(depth_pred_map)
    valid_slope = mask_2d & np.isfinite(slope_gt) & np.isfinite(slope_pred)
    if valid_slope.any():
        sg = slope_gt[valid_slope]
        sp = slope_pred[valid_slope]
        m["slope_corr"] = float(np.corrcoef(sg, sp)[0, 1])
        m["slope_mae"] = float(np.mean(np.abs(sp - sg)))
        m["slope_rmse"] = float(np.sqrt(np.mean((sp - sg) ** 2)))
        m["slope_gt_mean"] = float(sg.mean())
        m["slope_pred_mean"] = float(sp.mean())

        # HDA-relevant: slope detection threshold accuracy
        # For HDA, slopes > 10° are typically hazardous
        for thresh in [5, 10, 15, 20]:
            gt_safe = sg < thresh
            pred_safe = sp < thresh
            agree = (gt_safe == pred_safe)
            m[f"slope_agree_{thresh}deg"] = float(agree.mean() * 100)
            # False alarm: pred says unsafe but GT is safe
            if gt_safe.sum() > 0:
                m[f"slope_false_alarm_{thresh}deg"] = float(
                    ((~pred_safe) & gt_safe).sum() / gt_safe.sum() * 100
                )
            # Miss: pred says safe but GT is unsafe
            if (~gt_safe).sum() > 0:
                m[f"slope_miss_{thresh}deg"] = float(
                    (pred_safe & (~gt_safe)).sum() / (~gt_safe).sum() * 100
                )

    # --- Profile metrics (central row) ---
    row = H // 2
    gt_prof = depth_gt_map[row, :]
    pred_prof = depth_pred_map[row, :]
    valid_prof = mask_2d[row, :] & np.isfinite(gt_prof) & np.isfinite(pred_prof)
    if valid_prof.sum() > 10:
        m["profile_mae"] = float(np.mean(np.abs(pred_prof[valid_prof] - gt_prof[valid_prof])))
        m["profile_corr"] = float(np.corrcoef(gt_prof[valid_prof], pred_prof[valid_prof])[0, 1])

    return m


def compute_overlap_consistency(pred_v0, pred_v1, terrain_span):
    """Measure how well the two views agree in their overlap region."""
    tree_v1 = cKDTree(pred_v1)
    dist_0to1, _ = tree_v1.query(pred_v0, k=1)
    tree_v0 = cKDTree(pred_v0)
    dist_1to0, _ = tree_v0.query(pred_v1, k=1)

    overlap_thresh = terrain_span * 0.05
    ov0 = dist_0to1 < overlap_thresh
    ov1 = dist_1to0 < overlap_thresh

    if ov0.sum() < 10 or ov1.sum() < 10:
        return {"overlap_n": 0, "overlap_consistency": float("nan")}

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


# ─────────────────────────────────────────────────────────────────────────────
# Sparse Global Alignment reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def run_sparse_ga(model, device, img1_path, img2_path, opt_depth=False,
                  matching_conf_thr=5.0, niter1=500, niter2=200):
    """Run sparse_global_alignment on an image pair.

    Returns the SparseGA scene object with optimized poses, intrinsics, depthmaps.
    Both views live in a single consistent world frame.
    """
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    from dust3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    filelist = [str(img1_path), str(img2_path)]
    images = load_images(filelist, size=512, verbose=False)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)

    # Initialize with known intrinsics
    init_dict = {}
    for img_path in filelist:
        init_dict[img_path] = {
            'intrinsics': torch.tensor(K_GT, dtype=torch.float32),
        }

    cache_path = tempfile.mkdtemp(prefix="eval_sparse_ga_")
    scene = sparse_global_alignment(
        filelist, pairs, cache_path, model,
        lr1=0.01, niter1=niter1,
        lr2=0.014, niter2=niter2,
        device=device,
        shared_intrinsics=True,
        matching_conf_thr=matching_conf_thr,
        opt_depth=opt_depth,
        init=init_dict,
    )
    return scene

from dust3r.image_pairs import make_pairs

def evaluate_pair_sparse_ga(model, device, gt_folder, img1_path, img2_path,
                            opt_depth=False, matching_conf_thr=5.0,
                            niter1=500, niter2=200,
                            model_name="Model", output_root="eval_full_multi"):
    """Evaluate one image pair using sparse_global_alignment.

    Key difference from evaluate_pair(): ONE Sim(3) aligns the entire scene,
    ensuring both views are in a consistent coordinate system.
    """
    gt_folder = Path(gt_folder)
    name0 = Path(img1_path).stem
    name1 = Path(img2_path).stem

    # 1) Load GT for both views
    gt_pts0, depth_gt0, Kc0, T_wc0, mask_g0 = load_gt_view(gt_folder, name0)
    gt_pts1, depth_gt1, Kc1, T_wc1, mask_g1 = load_gt_view(gt_folder, name1)

    # 2) Try the demo-style global_aligner flow (build dust3r `output` and call global_aligner)
    pts3d_list = depthmaps = confs = None
    try:
        from dust3r.utils.image import load_images
        from dust3r.inference import inference
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

        images = load_images([str(img1_path), str(img2_path)], size=512, verbose=False)
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        # inference expects a list of tuples of images (batch size 1)
        output = inference(pairs, model, device, batch_size=1, verbose=False)

        print('    Running demo-style global_aligner (PointCloudOptimizer)...')
        ga = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
        # run the global alignment optimization (single lr schedule)
        try:
            ga.compute_global_alignment(init='mst', niter=300, schedule='cosine', lr=0.01)
        except Exception:
            # If compute_global_alignment signature differs, fall back to default call
            ga.compute_global_alignment()

        pts3d_list = to_numpy(ga.get_pts3d())
        depthmaps = to_numpy(ga.get_depthmaps())
        confs = to_numpy(ga.get_masks())
        poses = ga.get_im_poses().detach().cpu().numpy()
        focals = ga.get_focals().detach().cpu().numpy()

    except Exception as e:
        # Fallback: use sparse_global_alignment path
        print(f"    demo-style global_aligner failed ({e}), falling back to sparse_global_alignment()")
        scene = run_sparse_ga(model, device, img1_path, img2_path,
                              opt_depth=opt_depth,
                              matching_conf_thr=matching_conf_thr,
                              niter1=niter1, niter2=niter2)
        pts3d_list, depthmaps, confs = to_numpy(scene.get_dense_pts3d(clean_depth=True))
        poses = scene.get_im_poses().detach().cpu().numpy()
        focals = scene.get_focals().detach().cpu().numpy()

    H, W = confs[0].shape[:2]
    pts3d_v0_world = pts3d_list[0].reshape(-1, 3)
    pts3d_v1_world = pts3d_list[1].reshape(-1, 3)
    conf_v0 = confs[0].reshape(-1)
    conf_v1 = confs[1].reshape(-1)

    # 4) Build combined correspondences for scene-level Sim(3)
    mask_conf0 = conf_v0 >= 1.0
    mask_conf1 = conf_v1 >= 1.0
    mask_ok0 = mask_conf0 & mask_g0 & np.isfinite(pts3d_v0_world).all(axis=1)
    mask_ok1 = mask_conf1 & mask_g1 & np.isfinite(pts3d_v1_world).all(axis=1)

    pred_combined = np.vstack([pts3d_v0_world[mask_ok0], pts3d_v1_world[mask_ok1]])
    gt_combined = np.vstack([gt_pts0[mask_ok0], gt_pts1[mask_ok1]])

    if len(pred_combined) < 100:
        print(f"    Too few valid points ({len(pred_combined)}), skipping")
        return {"pair": f"{name0}_{name1}", "avg_chamfer": float("nan")}

    # 5) Single scene-level Sim(3) alignment
    T_sim3, scale, n_inliers = align_sim3_ransac(pred_combined, gt_combined, Nsub=8000)

    # 6) Apply the SAME transform to both views
    aligned_v0 = apply_sim3(pts3d_v0_world, T_sim3)
    aligned_v1 = apply_sim3(pts3d_v1_world, T_sim3)

    # Save aligned point clouds and transform for inspection
    try:
        out_dir = Path(output_root) / model_name / gt_folder.name / f"{name0}_vs_{name1}"
        out_dir.mkdir(parents=True, exist_ok=True)

        combined_pred = np.vstack([aligned_v0, aligned_v1])
        combined_gt = np.vstack([gt_pts0, gt_pts1])

        pcd_pred = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(combined_pred))
        pcd_gt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(combined_gt))

        o3d.io.write_point_cloud(str(out_dir / "aligned_pred.ply"), pcd_pred)
        o3d.io.write_point_cloud(str(out_dir / "aligned_gt.ply"), pcd_gt)
        np.savetxt(str(out_dir / "transform_sim3.txt"), T_sim3)
    except Exception as e:
        print(f"    Warning: failed to save PLYs/transform: {e}")

    results = {
        "pair": f"{name0}_{name1}",
        "scene_scale": float(scale),
        "scene_scale_err_pct": float(abs(scale - 1) * 100),
        "scene_n_inliers": int(n_inliers),
        "optimized_focal_v0": float(focals[0]),
        "optimized_focal_v1": float(focals[1]),
        "gt_focal": float(K_GT[0, 0]),
    }

    # 7) Per-view metrics using the scene-aligned points
    for view_idx, (aligned, gt_pts, depth_gt, mask_ok) in enumerate([
        (aligned_v0, gt_pts0, depth_gt0, mask_ok0),
        (aligned_v1, gt_pts1, depth_gt1, mask_ok1),
    ]):
        prefix = f"v{view_idx}"
        results[f"{prefix}_n_pts"] = int(mask_ok.sum())

        depth_pred_map = aligned.reshape(Hc, Wc, 3)[..., 2]
        depth_gt_map = gt_pts.reshape(Hc, Wc, 3)[..., 2]

        view_metrics = compute_per_view_metrics(
            aligned, gt_pts, depth_pred_map, depth_gt_map, mask_ok
        )
        for k, v in view_metrics.items():
            results[f"{prefix}_{k}"] = v

    # 8) Average metrics across views
    metric_keys = [
        "rmse", "mae_3d", "accuracy", "completeness", "chamfer",
        "pearson_z", "depth_mae", "depth_pearson",
        "slope_corr", "slope_mae", "slope_rmse",
        "profile_mae", "profile_corr",
        "slope_agree_10deg", "slope_miss_10deg", "slope_false_alarm_10deg",
    ]
    for k in metric_keys:
        v0 = results.get(f"v0_{k}")
        v1 = results.get(f"v1_{k}")
        if v0 is not None and v1 is not None:
            results[f"avg_{k}"] = (v0 + v1) / 2

    # 9) Overlap consistency — now meaningful since same scene-level Sim(3)
    try:
        pred_v0_valid = aligned_v0[mask_ok0]
        pred_v1_valid = aligned_v1[mask_ok1]
        terrain_span = np.linalg.norm(
            np.ptp(np.vstack([gt_pts0[mask_g0], gt_pts1[mask_g1]]), axis=0)
        )
        ov_metrics = compute_overlap_consistency(pred_v0_valid, pred_v1_valid, terrain_span)
        results.update(ov_metrics)
    except Exception as e:
        print(f"    Overlap consistency failed: {e}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Raw MASt3R evaluation (per-view independent Sim(3))
# ─────────────────────────────────────────────────────────────────────────────

Hc, Wc = 384, 512


def evaluate_pair(model, device, gt_folder, img1_path, img2_path):
    """Evaluate one image pair: per-view alignment + metrics + overlap consistency."""
    gt_folder = Path(gt_folder)
    name0 = Path(img1_path).stem
    name1 = Path(img2_path).stem

    # 1) Load GT for both views
    gt_pts0, depth_gt0, Kc0, T_wc0, mask_g0 = load_gt_view(gt_folder, name0)
    gt_pts1, depth_gt1, Kc1, T_wc1, mask_g1 = load_gt_view(gt_folder, name1)

    # 2) Run MASt3R inference
    outputs = MAST3RUtils.getMasterOutput(
        model, device, str(img1_path), str(img2_path),
        n_matches=100000, visualizeMatches=False, verboseFlag=False,
    )
    (_, _, filtered_m0, filtered_m1,
     pts3d_im0, pts3d_im1, conf_im0, conf_im1, *_) = outputs

    pts3d_im0 = to_numpy(pts3d_im0)  # (H, W, 3) in cam0 frame
    pts3d_im1 = to_numpy(pts3d_im1)  # (H, W, 3) in cam0 frame
    conf_im0 = to_numpy(conf_im0)
    conf_im1 = to_numpy(conf_im1)

    results = {"pair": f"{name0}_{name1}"}

    # 3) Per-view evaluation with independent Sim(3) alignment
    for view_idx, (pts3d, conf, gt_pts, depth_gt, mask_g) in enumerate([
        (pts3d_im0, conf_im0, gt_pts0, depth_gt0, mask_g0),
        (pts3d_im1, conf_im1, gt_pts1, depth_gt1, mask_g1),
    ]):
        prefix = f"v{view_idx}"
        flat_pred = pts3d.reshape(-1, 3)
        mask_conf = (conf >= 1).reshape(-1) if conf.ndim == 2 else (conf >= 1).reshape(-1)
        mask_ok = mask_conf & mask_g & np.isfinite(flat_pred).all(axis=1)

        if mask_ok.sum() < 100:
            print(f"    {prefix}: too few valid points ({mask_ok.sum()}), skipping")
            continue

        # Sim(3) alignment: pred -> GT
        T_sim3, scale, n_inliers = align_sim3_ransac(
            flat_pred[mask_ok], gt_pts[mask_ok]
        )
        aligned = apply_sim3(flat_pred, T_sim3)

        results[f"{prefix}_scale"] = float(scale)
        results[f"{prefix}_scale_err_pct"] = float(abs(scale - 1) * 100)
        results[f"{prefix}_n_pts"] = int(mask_ok.sum())

        # Compute depth maps from aligned points
        depth_pred_map = aligned.reshape(Hc, Wc, 3)[..., 2]
        depth_gt_map = gt_pts.reshape(Hc, Wc, 3)[..., 2]

        # Per-view metrics
        view_metrics = compute_per_view_metrics(
            aligned, gt_pts, depth_pred_map, depth_gt_map, mask_ok
        )
        for k, v in view_metrics.items():
            results[f"{prefix}_{k}"] = v

    # 4) Average metrics across views
    metric_keys = [
        "rmse", "mae_3d", "accuracy", "completeness", "chamfer",
        "pearson_z", "depth_mae", "depth_pearson",
        "slope_corr", "slope_mae", "slope_rmse",
        "profile_mae", "profile_corr",
        "slope_agree_10deg", "slope_miss_10deg", "slope_false_alarm_10deg",
    ]
    for k in metric_keys:
        v0 = results.get(f"v0_{k}")
        v1 = results.get(f"v1_{k}")
        if v0 is not None and v1 is not None:
            results[f"avg_{k}"] = (v0 + v1) / 2

    # 5) Overlap consistency (both views in GT frame after their own Sim(3))
    try:
        # Re-align both views for overlap computation
        flat0 = pts3d_im0.reshape(-1, 3)
        mask_ok0 = ((conf_im0 >= 1).reshape(-1)) & mask_g0 & np.isfinite(flat0).all(axis=1)
        T0, _, _ = align_sim3_ransac(flat0[mask_ok0], gt_pts0[mask_ok0])
        aligned0 = apply_sim3(flat0, T0)

        flat1 = pts3d_im1.reshape(-1, 3)
        mask_ok1 = ((conf_im1 >= 1).reshape(-1)) & mask_g1 & np.isfinite(flat1).all(axis=1)
        T1, _, _ = align_sim3_ransac(flat1[mask_ok1], gt_pts1[mask_ok1])
        aligned1 = apply_sim3(flat1, T1)

        pred_v0_valid = aligned0[mask_ok0]
        pred_v1_valid = aligned1[mask_ok1]
        terrain_span = np.linalg.norm(
            np.ptp(np.vstack([gt_pts0[mask_g0], gt_pts1[mask_g1]]), axis=0)
        )

        ov_metrics = compute_overlap_consistency(pred_v0_valid, pred_v1_valid, terrain_span)
        results.update(ov_metrics)
    except Exception as e:
        print(f"    Overlap consistency failed: {e}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Batch evaluation
# ─────────────────────────────────────────────────────────────────────────────

def batch_evaluate_model(
    model,
    model_name,
    device,
    gt_folder,
    folder_key,
    max_pairs=None,
    mode="sparse_ga",
    matching_conf_thr=5.0,
    niter1=500,
    niter2=200,
    opt_depth=False,
    output_root="eval_full_multi",
):
    """Evaluate all pairs for one model on one folder. Routes to different
    evaluation functions depending on `mode`.
    """
    gt_folder = Path(gt_folder)
    images = sorted(gt_folder.glob("*.jpg"))
    pairs = list(zip(images[0::2], images[1::2]))

    if max_pairs is not None and max_pairs < len(pairs):
        step = max(1, len(pairs) // max_pairs)
        pairs = pairs[::step][:max_pairs]

    results = []
    for img1, img2 in pairs:
        print(f"  [{model_name}] {img1.name} vs {img2.name}")
        try:
            if mode.startswith("sparse_ga"):
                row = evaluate_pair_sparse_ga(
                    model,
                    device,
                    gt_folder,
                    img1,
                    img2,
                    opt_depth=opt_depth,
                    matching_conf_thr=matching_conf_thr,
                    niter1=niter1,
                    niter2=niter2,
                    model_name=model_name,
                    output_root=output_root,
                )
            else:
                row = evaluate_pair(model, device, gt_folder, img1, img2)

            row["Model"] = model_name
            row["Folder"] = folder_key
            results.append(row)
        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "Model": model_name,
                "Folder": folder_key,
                "pair": f"{img1.stem}_{img2.stem}",
                "avg_chamfer": float("nan"),
            })

    return results


def print_summary(df):
    """Print summary table per model."""
    key_cols = [
        "avg_accuracy", "avg_completeness", "avg_chamfer",
        "avg_slope_corr", "avg_slope_mae",
        "avg_profile_corr", "avg_profile_mae",
        "avg_depth_pearson",
        "avg_slope_agree_10deg", "avg_slope_miss_10deg",
        "overlap_consistency_median",
    ]
    existing_cols = [c for c in key_cols if c in df.columns]

    print("\n" + "=" * 100)
    print("  PER-VIEW EVALUATION SUMMARY")
    print("=" * 100)

    models = df["Model"].unique()
    rows = []
    for mname in models:
        ms = df[df["Model"] == mname]
        row = {"Model": mname, "N": len(ms)}
        for col in existing_cols:
            vals = ms[col].dropna()
            if len(vals) > 0:
                row[f"{col}_med"] = round(vals.median(), 4)
        rows.append(row)

    df_sum = pd.DataFrame(rows)
    print(df_sum.to_string(index=False))
    return df_sum


def print_hda_summary(df):
    """Print HDA-specific metrics summary."""
    print("\n" + "=" * 100)
    print("  HDA SLOPE DETECTION SUMMARY")
    print("=" * 100)

    hda_cols = []
    for thresh in [5, 10, 15, 20]:
        for metric in ["agree", "miss", "false_alarm"]:
            col = f"avg_slope_{metric}_{thresh}deg"
            if col in df.columns:
                hda_cols.append(col)

    if not hda_cols:
        print("  No HDA metrics available")
        return

    models = df["Model"].unique()
    rows = []
    for mname in models:
        ms = df[df["Model"] == mname]
        row = {"Model": mname}
        for col in hda_cols:
            vals = ms[col].dropna()
            if len(vals) > 0:
                short = col.replace("avg_slope_", "").replace("deg", "°")
                row[short] = f"{vals.median():.1f}%"
        rows.append(row)

    df_hda = pd.DataFrame(rows)
    print(df_hda.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Per-view evaluation for Teacher + Students")
    parser.add_argument("--max_pairs", type=int, default=3,
                        help="Max pairs per GT folder (default: 3 for quick test)")
    parser.add_argument("--output_dir", type=str, default="eval_per_view",
                        help="Output directory")
    parser.add_argument("--gt_folders", nargs="+", default=["landing"],
                        choices=["nadir", "pitch", "landing"],
                        help="GT folders to evaluate on")
    parser.add_argument("--gt_folder_path", type=str, default=None,
                        help="Path to a specific GT folder (use with --img1/--img2)")
    parser.add_argument("--img1", type=str, default=None,
                        help="Path to first image (run single pair)")
    parser.add_argument("--img2", type=str, default=None,
                        help="Path to second image (run single pair)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model names to evaluate (default: Teacher + all students)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mode", type=str, default="sparse_ga",
                        choices=["raw", "sparse_ga", "sparse_ga_depth"],
                        help="Reconstruction mode: raw | sparse_ga | sparse_ga_depth")
    parser.add_argument("--matching_conf_thr", type=float, default=5.0,
                        help="Matching confidence threshold for sparse_ga")
    parser.add_argument("--niter1", type=int, default=500,
                        help="Number of iterations (stage 1) for sparse_ga")
    parser.add_argument("--niter2", type=int, default=200,
                        help="Number of iterations (stage 2) for sparse_ga")
    parser.add_argument("--opt_depth", action="store_true",
                        help="Enable depth optimization in sparse_ga (sparse_ga_depth)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    # Load models
    print("Loading Teacher...")
    teacher = load_teacher(device)
    all_models = {"Teacher": teacher}

    print("Loading Students...")
    students = load_students(device)

    if args.models is not None:
        # Filter to requested models
        for name in args.models:
            if name in students:
                all_models[name] = students[name]
    else:
        all_models.update(students)

    print(f"\nModels to evaluate: {list(all_models.keys())}")
    print(f"GT folders: {args.gt_folders}")
    print(f"Max pairs per folder: {args.max_pairs}")

    all_results = []

    for folder_key in args.gt_folders:
        gt_folder = GT_FOLDERS[folder_key]
        if not gt_folder.exists():
            print(f"WARNING: {gt_folder} does not exist, skipping")
            continue

        print(f"\n{'='*70}")
        print(f"  Evaluating on: {folder_key} ({gt_folder})")
        print(f"{'='*70}")

        # If user provided explicit image pair, run only that pair (use gt_folder_path if provided)
        if args.img1 is not None and args.img2 is not None:
            # Determine GT folder to use: prefer explicit path, else use mapped folder
            if args.gt_folder_path is not None:
                gt_folder_use = Path(args.gt_folder_path)
            else:
                gt_folder_use = gt_folder

            for model_name, model in all_models.items():
                print(f"  [{model_name}] single pair {Path(args.img1).name} vs {Path(args.img2).name}")
                try:
                    if args.mode.startswith("sparse_ga"):
                        row = evaluate_pair_sparse_ga(
                            model,
                            device,
                            gt_folder_use,
                            args.img1,
                            args.img2,
                            opt_depth=args.opt_depth,
                            matching_conf_thr=args.matching_conf_thr,
                            niter1=args.niter1,
                            niter2=args.niter2,
                            model_name=model_name,
                            output_root=args.output_dir,
                        )
                    else:
                        row = evaluate_pair(model, device, gt_folder_use, args.img1, args.img2)

                    row["Model"] = model_name
                    row["Folder"] = folder_key
                    all_results.append(row)
                except Exception as e:
                    print(f"    FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results.append({
                        "Model": model_name,
                        "Folder": folder_key,
                        "pair": f"{Path(args.img1).stem}_{Path(args.img2).stem}",
                        "avg_chamfer": float("nan"),
                    })
            # skip the normal per-folder loop for this folder when single-pair was run
            continue

        for model_name, model in all_models.items():
            t0 = time.time()
            results = batch_evaluate_model(
                model,
                model_name,
                device,
                gt_folder,
                folder_key,
                max_pairs=args.max_pairs,
                mode=args.mode,
                matching_conf_thr=args.matching_conf_thr,
                niter1=args.niter1,
                niter2=args.niter2,
                opt_depth=args.opt_depth,
                output_root=args.output_dir,
            )
            dt = time.time() - t0
            print(f"  {model_name}: {len(results)} pairs in {dt:.1f}s")
            all_results.extend(results)

    # Build DataFrame
    df = pd.DataFrame(all_results)

    # Save full results
    csv_path = Path(args.output_dir) / "per_view_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFull results saved to {csv_path}")

    # Print summaries
    print_summary(df)
    print_hda_summary(df)

    # Save summary JSON
    summary = {}
    for mname in df["Model"].unique():
        ms = df[df["Model"] == mname]
        summary[mname] = {}
        for col in df.columns:
            if col in ["Model", "Folder", "pair"]:
                continue
            vals = ms[col].dropna()
            if len(vals) > 0:
                try:
                    summary[mname][col] = {
                        "median": round(float(vals.median()), 4),
                        "mean": round(float(vals.mean()), 4),
                        "std": round(float(vals.std()), 4),
                    }
                except (TypeError, ValueError):
                    pass

    json_path = Path(args.output_dir) / "per_view_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {json_path}")


if __name__ == "__main__":
    main()
