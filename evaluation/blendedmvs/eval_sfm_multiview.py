#!/usr/bin/env python3
"""
eval_sfm_multiview.py — Multi-view 3D reconstruction evaluation via SfM.

Uses MASt3R's sparse_global_alignment (matches → pose estimation → triangulation)
instead of direct pointmaps. This avoids stitching artifacts and evaluates the
full SfM pipeline: matching quality → pose accuracy → 3D reconstruction.

Metrics:
  - Pose: AUC@5/10/20 of max(RRA, RTA)
  - Depth: abs_rel, delta1, RMSE (per-view, Sim(3) aligned)
  - 3D:   Chamfer distance, completeness (after Sim(3) alignment)

Usage:
    python eval_sfm_multiview.py --models teacher --gt_folders nadir
    python eval_sfm_multiview.py --models teacher S2_ViT-Small --n_views 5 --max_scenes 10
"""

import sys
import os
import argparse
import json
import time
import tempfile
import shutil

import numpy as np
import torch
import cv2
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mast3r.utils.path_to_dust3r  # noqa
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs

from eval_emat import (
    STUDENT_CONFIGS,
    GT_FOLDERS,
    K_GT,
    load_teacher,
    load_students,
    rra_deg,
    rta_deg,
    compute_auc,
)

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


def save_ply(path, pts, colors=None):
    """Save point cloud as PLY file."""
    pts = pts[np.isfinite(pts).all(axis=-1)]
    n = len(pts)
    if n == 0:
        return
    has_color = colors is not None and len(colors) == n
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            line = f"{pts[i,0]:.6f} {pts[i,1]:.6f} {pts[i,2]:.6f}"
            if has_color:
                r, g, b = np.clip(colors[i] * 255, 0, 255).astype(np.uint8)
                line += f" {r} {g} {b}"
            f.write(line + "\n")


# =============================================================================
# GT loading
# =============================================================================

def load_gt_for_image(gt_folder, img_name):
    """Load GT intrinsics, cam2world, depth for one image."""
    npz_path = gt_folder / f"{img_name}.npz"
    exr_path = gt_folder / f"{img_name}.exr"

    gt = np.load(str(npz_path))
    K = gt["intrinsics"]           # (3,3)
    cam2world = gt["cam2world"]    # (4,4)

    depth = None
    if exr_path.exists():
        depth = _read_exr(str(exr_path))

    return K, cam2world, depth


def _read_exr(path):
    """Read single-channel EXR depth map."""
    try:
        import OpenEXR
        import Imath
        f = OpenEXR.InputFile(path)
        dw = f.header()['dataWindow']
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        # Try channel names
        for ch in ['Z', 'Y', 'R', 'depth']:
            if ch in f.header()['channels']:
                data = np.frombuffer(f.channel(ch, pt), dtype=np.float32).reshape(h, w)
                return data
        # Fallback: first channel
        ch = list(f.header()['channels'].keys())[0]
        return np.frombuffer(f.channel(ch, pt), dtype=np.float32).reshape(h, w)
    except ImportError:
        return cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)


def crop_depth_to_dust3r(depth_full, Hc=384, Wc=512):
    """Center-crop depth map to DUSt3R output size."""
    H, W = depth_full.shape[:2]
    y0 = (H - Hc) // 2
    x0 = (W - Wc) // 2
    return depth_full[y0:y0+Hc, x0:x0+Wc], y0, x0


def backproject_depth(depth, K, cam2world):
    """Depth map → world-frame 3D points."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth.astype(np.float64)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pts_cam = np.stack([x, y, z], axis=-1)  # (H, W, 3)
    valid = (z > 0) & np.isfinite(z)

    # To world
    R = cam2world[:3, :3]
    t = cam2world[:3, 3]
    pts_world = (pts_cam @ R.T) + t

    return pts_world, valid


# =============================================================================
# SfM reconstruction
# =============================================================================

def run_sfm_reconstruction(model, device, img_paths, K_gt,
                           matching_conf_thr=5.0, niter1=500, niter2=200):
    """
    Run MASt3R sparse_global_alignment on N images.

    Returns:
        pts3d_list: list of N arrays (H, W, 3) in world frame
        depthmaps:  list of N arrays (H, W)
        confs:      list of N confidence arrays
        poses:      (N, 4, 4) cam2world
        focals:     (N,) estimated focal lengths
    """
    filelist = [str(p) for p in img_paths]
    images = load_images(filelist, size=512, verbose=False)
    pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)

    # Provide GT intrinsics as initialization
    init_dict = {
        p: {"intrinsics": torch.tensor(K_gt, dtype=torch.float32)}
        for p in filelist
    }

    cache_path = tempfile.mkdtemp(prefix="eval_sfm_")
    try:
        scene = sparse_global_alignment(
            filelist, pairs, cache_path, model,
            lr1=0.2, niter1=niter1,
            lr2=0.02, niter2=niter2,
            device=device,
            shared_intrinsics=True,
            matching_conf_thr=matching_conf_thr,
            init=init_dict,
        )

        # Extract results
        pts3d_confs = scene.get_dense_pts3d(clean_depth=True)
        pts3d_list = [p.detach().cpu().numpy() if torch.is_tensor(p) else np.array(p)
                      for p in pts3d_confs[0]]
        confs = [c.detach().cpu().numpy() if torch.is_tensor(c) else np.array(c)
                 for c in pts3d_confs[2]]
        poses = scene.get_im_poses().detach().cpu().numpy()   # (N,4,4) cam2world
        focals = scene.get_focals().detach().cpu().numpy()

        # Compute camera-frame depth from world-frame 3D points + poses
        depthmaps = []
        for i, pts3d_w in enumerate(pts3d_list):
            w2c = np.linalg.inv(poses[i])
            R, t = w2c[:3, :3], w2c[:3, 3]
            shape = pts3d_w.shape[:-1]  # (H, W) or (N,)
            pts_cam = (pts3d_w.reshape(-1, 3) @ R.T + t).reshape(*shape, 3)
            depthmaps.append(pts_cam[..., 2])  # z-depth in camera frame
    finally:
        shutil.rmtree(cache_path, ignore_errors=True)

    return pts3d_list, depthmaps, confs, poses, focals


# =============================================================================
# Metrics
# =============================================================================

def sim3_align(pts_pred, pts_gt, max_pts=10000):
    """Sim(3) alignment via Umeyama. Returns s, R, t and aligned points."""
    # Subsample for speed
    if len(pts_pred) > max_pts:
        idx = np.random.choice(len(pts_pred), max_pts, replace=False)
        pts_p = pts_pred[idx]
        pts_g = pts_gt[idx]
    else:
        pts_p, pts_g = pts_pred, pts_gt

    n = len(pts_p)
    mu_p = pts_p.mean(0)
    mu_g = pts_g.mean(0)
    dp = pts_p - mu_p
    dg = pts_g - mu_g

    # Variance of source
    var_p = (dp ** 2).sum() / n

    # Cross-covariance
    H = dp.T @ dg / n
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ D @ U.T

    # Scale: s = trace(D @ S) / var_p
    s = np.trace(D @ np.diag(S)) / var_p
    t = mu_g - s * R @ mu_p

    aligned = s * (pts_pred @ R.T) + t
    return s, R, t, aligned



def chamfer_distance(pts_a, pts_b, max_pts=50000):
    """Chamfer distance between two point clouds."""
    if len(pts_a) > max_pts:
        pts_a = pts_a[np.random.choice(len(pts_a), max_pts, replace=False)]
    if len(pts_b) > max_pts:
        pts_b = pts_b[np.random.choice(len(pts_b), max_pts, replace=False)]

    from scipy.spatial import cKDTree
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    d_ab, _ = tree_b.query(pts_a, k=1)
    d_ba, _ = tree_a.query(pts_b, k=1)
    return float((d_ab.mean() + d_ba.mean()) / 2)


def pose_metrics_multiview(est_poses, gt_c2w_list):
    """
    Compute pairwise RRA/RTA between all consecutive pairs
    after Sim(3) alignment of the full pose trajectory.
    est_poses: (N, 4, 4) estimated cam2world
    gt_c2w_list: list of N (4,4) GT cam2world
    """
    N = len(est_poses)
    gt_poses = np.stack(gt_c2w_list)  # (N, 4, 4)

    # Align estimated poses to GT via Procrustes on camera centers
    centers_est = est_poses[:, :3, 3]
    centers_gt = gt_poses[:, :3, 3]

    if N >= 3:
        _, _, _, centers_aligned = sim3_align(centers_est, centers_gt)
    else:
        centers_aligned = centers_est

    rra_list, rta_list = [], []
    for i in range(N):
        for j in range(i + 1, N):
            # GT relative pose
            R_gt_i, t_gt_i = gt_poses[i, :3, :3], gt_poses[i, :3, 3]
            R_gt_j, t_gt_j = gt_poses[j, :3, :3], gt_poses[j, :3, 3]
            R_gt_rel = R_gt_j.T @ R_gt_i
            t_gt_rel = R_gt_j.T @ (t_gt_i - t_gt_j)

            # Est relative pose
            R_est_i, t_est_i = est_poses[i, :3, :3], est_poses[i, :3, 3]
            R_est_j, t_est_j = est_poses[j, :3, :3], est_poses[j, :3, 3]
            R_est_rel = R_est_j.T @ R_est_i
            t_est_rel = R_est_j.T @ (t_est_i - t_est_j)

            rra = rra_deg(R_est_rel, R_gt_rel)
            rta = rta_deg(t_est_rel, t_gt_rel)
            rra_list.append(rra)
            rta_list.append(rta)

    errors = [max(r, t) for r, t in zip(rra_list, rta_list)]
    aucs = compute_auc(errors)

    return {
        "rra_median": float(np.median(rra_list)),
        "rta_median": float(np.median(rta_list)),
        "n_pairs": len(errors),
        **aucs,
    }


# =============================================================================
# Scene evaluation
# =============================================================================

def evaluate_scene(model, device, img_paths, gt_folder, K_gt, n_views=5,
                   matching_conf_thr=5.0, niter1=500, niter2=200,
                   save_dir=None, scene_name=None):
    """Evaluate SfM reconstruction for one scene (group of N images)."""
    t0 = time.time()

    # Run SfM
    try:
        pts3d_list, depthmaps, confs, est_poses, focals = run_sfm_reconstruction(
            model, device, img_paths, K_gt,
            matching_conf_thr=matching_conf_thr,
            niter1=niter1, niter2=niter2,
        )
    except Exception as e:
        print(f"    SfM failed: {e}")
        return None

    elapsed = time.time() - t0

    # Load GT for all views
    gt_c2w_list = []
    gt_depths = []
    gt_Ks = []
    for p in img_paths:
        K, c2w, depth = load_gt_for_image(gt_folder, p.stem)
        gt_c2w_list.append(c2w)
        gt_depths.append(depth)
        gt_Ks.append(K)

    N = len(img_paths)

    # --- Save complete SfM output ---
    if save_dir is not None and scene_name is not None:
        scene_dir = Path(save_dir) / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "img_names": np.array([p.stem for p in img_paths]),
            "est_poses": est_poses,                          # (N,4,4) cam2world
            "focals": focals,                                # (N,)
            "gt_poses": np.stack(gt_c2w_list),               # (N,4,4) cam2world
            "gt_intrinsics": np.stack(gt_Ks),                # (N,3,3)
            "K_gt": K_gt,                                    # (3,3)
        }
        for i in range(N):
            save_dict[f"pts3d_{i}"] = pts3d_list[i]          # (Npts,3) world frame
            save_dict[f"depth_{i}"] = depthmaps[i]           # cam-frame z-depth
            save_dict[f"conf_{i}"] = confs[i]
            if gt_depths[i] is not None:
                save_dict[f"gt_depth_{i}"] = gt_depths[i]
        np.savez_compressed(scene_dir / "sfm_output.npz", **save_dict)

        # Save PLY point clouds
        all_pts = []
        for i in range(N):
            p = pts3d_list[i]
            if p is not None:
                all_pts.append(p.reshape(-1, 3))
                save_ply(scene_dir / f"pts3d_{img_paths[i].stem}.ply", p.reshape(-1, 3))
        if all_pts:
            save_ply(scene_dir / "scene_merged.ply", np.concatenate(all_pts))

        # GT PLY
        Hc_, Wc_ = 384, 512
        all_gt_pts = []
        for i in range(N):
            if gt_depths[i] is not None:
                dc, y0_, x0_ = crop_depth_to_dust3r(gt_depths[i], Hc_, Wc_)
                Kc = gt_Ks[i].copy()
                Kc[0, 2] -= x0_
                Kc[1, 2] -= y0_
                gw, gv = backproject_depth(dc, Kc, gt_c2w_list[i])
                gf = gw.reshape(-1, 3)[gv.ravel()]
                save_ply(scene_dir / f"gt_{img_paths[i].stem}.ply", gf)
                all_gt_pts.append(gf)
        if all_gt_pts:
            save_ply(scene_dir / "gt_merged.ply", np.concatenate(all_gt_pts))

        print(f"[saved {scene_dir}] ", end="")

    # --- Pose metrics ---
    pose_m = pose_metrics_multiview(est_poses, gt_c2w_list)

    # --- Collect predicted 3D points (sparse, from SfM) ---
    Hc, Wc = 384, 512
    all_pts_pred = []
    all_pts_gt = []

    for i in range(N):
        # Predicted 3D points (may be sparse anchor points, not H×W grid)
        if pts3d_list[i] is not None:
            pts_pred = pts3d_list[i]
            if pts_pred.ndim >= 2:
                pts_pred = pts_pred.reshape(-1, 3)
            mask_pred = np.isfinite(pts_pred).all(axis=-1) & (np.abs(pts_pred) < 1e6).all(axis=-1)
            if mask_pred.sum() > 100:
                all_pts_pred.append(pts_pred[mask_pred])

        # GT 3D points from depth backprojection
        if gt_depths[i] is not None:
            depth_gt_crop, y0, x0 = crop_depth_to_dust3r(gt_depths[i], Hc, Wc)
            K_crop = gt_Ks[i].copy()
            K_crop[0, 2] -= x0
            K_crop[1, 2] -= y0
            pts_gt_w, valid_gt = backproject_depth(depth_gt_crop, K_crop, gt_c2w_list[i])
            pts_gt_flat = pts_gt_w.reshape(-1, 3)[valid_gt.ravel()]
            if len(pts_gt_flat) > 100:
                all_pts_gt.append(pts_gt_flat)

    # --- 3D Chamfer distance (Sim(3) aligned) ---
    chamfer = np.nan
    if all_pts_pred and all_pts_gt:
        pts_p = np.concatenate(all_pts_pred)
        pts_g = np.concatenate(all_pts_gt)
        try:
            _, _, _, pts_p_aligned = sim3_align(pts_p, pts_g)
            chamfer = chamfer_distance(pts_p_aligned, pts_g)
        except Exception as e:
            print(f"    Chamfer failed: {e}")

    return {
        "n_views": N,
        "time_s": round(elapsed, 1),
        "focal_est": float(np.mean(focals)),
        **pose_m,
        "chamfer": float(chamfer),
    }


# =============================================================================
# Main
# =============================================================================

def get_scene_groups(gt_folder, n_views):
    """Group images into scenes of n_views consecutive images."""
    images = sorted(gt_folder.glob("*.jpg"))
    scenes = []
    for i in range(0, len(images) - n_views + 1, n_views):
        scenes.append(images[i:i + n_views])
    return scenes


def main():
    parser = argparse.ArgumentParser(description="SfM-based multi-view 3D evaluation")
    parser.add_argument("--models", nargs="+", default=["teacher"],
                        help="Models to evaluate: teacher, S2_ViT-Small, etc.")
    parser.add_argument("--gt_folders", nargs="+", default=["nadir"],
                        choices=list(GT_FOLDERS.keys()),
                        help="GT test sets to evaluate on")
    parser.add_argument("--n_views", type=int, default=5,
                        help="Number of views per scene")
    parser.add_argument("--max_scenes", type=int, default=None,
                        help="Max scenes per GT folder")
    parser.add_argument("--matching_conf_thr", type=float, default=5.0)
    parser.add_argument("--niter1", type=int, default=500,
                        help="Coarse alignment iterations")
    parser.add_argument("--niter2", type=int, default=200,
                        help="Refinement iterations")
    parser.add_argument("--output_dir", type=str, default="eval_sfm_multiview")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    loaded = {}
    for name in args.models:
        if name == "teacher":
            loaded["teacher"] = load_teacher(device)
        else:
            # Load one student
            if name not in STUDENT_CONFIGS:
                print(f"Unknown model: {name}. Available: {list(STUDENT_CONFIGS.keys())}")
                continue
            students = load_students(device)
            if name in students:
                loaded[name] = students[name]
            else:
                print(f"Could not load {name} (checkpoint missing?)")

    if not loaded:
        print("No models loaded. Exiting.")
        return

    all_results = {}

    for folder_key in args.gt_folders:
        gt_folder = GT_FOLDERS[folder_key]
        print(f"\n{'='*60}")
        print(f"GT folder: {folder_key} ({gt_folder})")
        print(f"{'='*60}")

        scenes = get_scene_groups(gt_folder, args.n_views)
        if args.max_scenes and len(scenes) > args.max_scenes:
            scenes = scenes[:args.max_scenes]

        print(f"  {len(scenes)} scenes of {args.n_views} views each")

        for model_name, model in loaded.items():
            print(f"\n  --- {model_name} ---")
            scene_results = []

            for si, scene_imgs in enumerate(scenes):
                scene_name = f"{scene_imgs[0].stem}..{scene_imgs[-1].stem}"
                print(f"    [{si+1}/{len(scenes)}] {scene_name}", end=" ", flush=True)

                result = evaluate_scene(
                    model, device, scene_imgs, gt_folder, K_GT,
                    n_views=args.n_views,
                    matching_conf_thr=args.matching_conf_thr,
                    niter1=args.niter1,
                    niter2=args.niter2,
                    save_dir=output_dir / folder_key / model_name,
                    scene_name=scene_name,
                )

                if result is not None:
                    scene_results.append(result)
                    print(f"  AUC@5={result.get('AUC@5','?')} "
                          f"chamfer={result.get('chamfer','?'):.4f} "
                          f"({result['time_s']}s)")
                else:
                    print("  FAILED")

            if not scene_results:
                print(f"    No successful scenes for {model_name}")
                continue

            # Aggregate
            agg = {}
            for key in scene_results[0]:
                vals = [r[key] for r in scene_results
                        if isinstance(r[key], (int, float)) and not np.isnan(r[key])]
                if vals:
                    agg[key] = float(np.mean(vals))

            key = f"{folder_key}/{model_name}"
            all_results[key] = {
                "aggregate": agg,
                "per_scene": scene_results,
            }

            print(f"\n  {model_name} aggregate on {folder_key}:")
            for k, v in agg.items():
                if k in ("n_views", "n_pairs"):
                    print(f"    {k}: {v:.0f}")
                else:
                    print(f"    {k}: {v:.4f}")

    # Save results
    out_path = output_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    header = f"{'Model':<25} {'Folder':<10} {'AUC@5':>7} {'AUC@10':>7} {'AUC@20':>7} {'RRA_med':>8} {'RTA_med':>8} {'Chamfer':>10}"
    print(header)
    print("-" * len(header))
    for key, data in all_results.items():
        folder, model = key.split("/")
        a = data["aggregate"]
        print(f"{model:<25} {folder:<10} "
              f"{a.get('AUC@5', 0):>7.2f} "
              f"{a.get('AUC@10', 0):>7.2f} "
              f"{a.get('AUC@20', 0):>7.2f} "
              f"{a.get('rra_median', 0):>8.2f} "
              f"{a.get('rta_median', 0):>8.2f} "
              f"{a.get('chamfer', 0):>10.4f}")


if __name__ == "__main__":
    main()
