#!/usr/bin/env python3
"""
Multi-view evaluation on BlendedMVS test split.

For each test scene, pick N images, run all-pairs inference + global alignment,
then compare the fused 3D reconstruction against GT depth maps.

Metrics:
  - Per-view depth: abs_rel, delta<1.25, RMSE (after Sim(3) alignment)
  - Scene-level 3D: Chamfer distance, accuracy, completeness
  - Pose: RRA, RTA from optimized poses vs GT

Also exports PLY point clouds for visualization.

Usage:
  python eval_blendedmvs_multiview.py --n_views 5 --max_scenes 10 --include_teacher
"""
import os, sys, json, argparse, time, tempfile
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# ── MASt3R imports ──────────────────────────────────────────────────
import mast3r.utils.path_to_dust3r  # noqa
from mast3r.model import AsymmetricMASt3R
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# ── Student builders ────────────────────────────────────────────────
from distillation_dual import build_vit_student, build_vit_tiny_student

BMVS_ROOT = "Datas/blendedmvs_processed"
TEACHER_CKPT = "mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

STUDENT_CONFIGS = {
    "S2_full": dict(
        builder=build_vit_student,
        kwargs=dict(dec_embed_dim=512, dec_depth=6, dec_num_heads=4, mlp_ratio=1.0,
                    prefer_dinov2=True, backbone_type="dinov2"),
        ckpt="output/general_s2_full/checkpoint-best.pth",
    ),
    "S4_reduced": dict(
        builder=build_vit_student,
        kwargs=dict(dec_embed_dim=256, dec_depth=4, dec_num_heads=4, mlp_ratio=1.0,
                    prefer_dinov2=True, backbone_type="dinov2"),
        ckpt="output/general_s4_reduced/checkpoint-best.pth",
    ),
    "S5_vit_tiny": dict(
        builder=build_vit_tiny_student,
        kwargs=dict(dec_embed_dim=512, dec_depth=6, dec_num_heads=4, mlp_ratio=1.0),
        ckpt="output/general_s5_vit_tiny/checkpoint-best.pth",
    ),
    "S6_no_svd": dict(
        builder=build_vit_student,
        kwargs=dict(dec_embed_dim=512, dec_depth=6, dec_num_heads=4, mlp_ratio=1.0,
                    prefer_dinov2=True, backbone_type="dinov2"),
        ckpt="output/general_s6_no_svd/checkpoint-best.pth",
    ),
    "S7_no_feat": dict(
        builder=build_vit_student,
        kwargs=dict(dec_embed_dim=512, dec_depth=6, dec_num_heads=4, mlp_ratio=1.0,
                    prefer_dinov2=True, backbone_type="dinov2"),
        ckpt="output/general_s7_no_feat/checkpoint-best.pth",
    ),
}


# ── Utilities ───────────────────────────────────────────────────────

def get_test_scenes(bmvs_root):
    """Get test split scenes (seq_low % 10 == 1) from pairs file."""
    pairs = np.load(os.path.join(bmvs_root, 'blendedmvs_pairs.npy'))
    # Filter existing scenes
    existing = np.array([os.path.isdir(os.path.join(bmvs_root, f"{h:08x}{l:016x}"))
                         for h, l in zip(pairs['seq_high'], pairs['seq_low'])])
    pairs = pairs[existing]
    # Test split
    test_mask = (pairs['seq_low'] % 10) == 1
    test_pairs = pairs[test_mask]

    # Group by scene
    scenes = {}
    for p in test_pairs:
        seq = f"{p['seq_high']:08x}{p['seq_low']:016x}"
        if seq not in scenes:
            scenes[seq] = set()
        scenes[seq].add(int(p['img1']))
        scenes[seq].add(int(p['img2']))

    return scenes


def load_gt_for_image(seq_path, img_idx):
    """Load GT depth, intrinsics, cam2world for one image."""
    name = f"{img_idx:08d}"
    npz_path = os.path.join(seq_path, name + ".npz")
    exr_path = os.path.join(seq_path, name + ".exr")
    if not os.path.isfile(npz_path) or not os.path.isfile(exr_path):
        return None
    gt = np.load(npz_path)
    depth = cv2.imread(exr_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    if depth is None:
        return None
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = gt['R_cam2world']
    pose[:3, 3] = gt['t_cam2world']
    K = np.float32(gt['intrinsics'])
    return dict(depth=depth, pose=pose, K=K)


def umeyama_alignment(src, dst):
    """Sim(3) alignment: find s, R, t such that dst ~ s*R@src + t."""
    n, d = src.shape
    mu_s, mu_d = src.mean(0), dst.mean(0)
    src_c, dst_c = src - mu_s, dst - mu_d
    sigma_s = np.mean(np.sum(src_c ** 2, axis=1))
    cov = dst_c.T @ src_c / n
    U, S, Vt = np.linalg.svd(cov)
    det_sign = np.linalg.det(U) * np.linalg.det(Vt)
    D = np.eye(d)
    if det_sign < 0:
        D[-1, -1] = -1
    R = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / sigma_s
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def backproject(depth, K, H_out, W_out):
    """Back-project depth to 3D points using intrinsics K."""
    H_orig, W_orig = depth.shape[:2]
    if (H_orig, W_orig) != (H_out, W_out):
        depth = cv2.resize(depth, (W_out, H_out), interpolation=cv2.INTER_NEAREST)
    sx, sy = W_out / W_orig, H_out / H_orig
    fx, fy = K[0, 0] * sx, K[1, 1] * sy
    cx, cy = K[0, 2] * sx, K[1, 2] * sy
    u, v = np.meshgrid(np.arange(W_out), np.arange(H_out))
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    return np.stack([x, y, depth], axis=-1)


def pose_angular_error(pred_poses, gt_poses):
    """Compute median RRA and RTA across all pairs of views."""
    n = len(pred_poses)
    rra_list, rta_list = [], []
    for i in range(n):
        for j in range(i + 1, n):
            # Relative pose
            gt_rel = gt_poses[j] @ np.linalg.inv(gt_poses[i])
            pred_rel = pred_poses[j] @ np.linalg.inv(pred_poses[i])
            # RRA
            R_err = gt_rel[:3, :3].T @ pred_rel[:3, :3]
            cos_a = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
            rra_list.append(np.degrees(np.arccos(cos_a)))
            # RTA
            gt_t = gt_rel[:3, 3]
            pred_t = pred_rel[:3, 3]
            gn, pn = np.linalg.norm(gt_t), np.linalg.norm(pred_t)
            if gn > 1e-8 and pn > 1e-8:
                cos_t = np.clip(np.dot(gt_t, pred_t) / (gn * pn), -1, 1)
                rta_list.append(np.degrees(np.arccos(cos_t)))
    return rra_list, rta_list


# ── Model loading ───────────────────────────────────────────────────

def load_teacher(device):
    model = AsymmetricMASt3R.from_pretrained(TEACHER_CKPT).to(device)
    model.eval()
    return model


def load_student(name, cfg, device):
    ckpt_path = cfg['ckpt']
    print(f"Loading {name} from {ckpt_path}")
    if not os.path.isfile(ckpt_path):
        print(f"  Checkpoint not found: {ckpt_path}, skipping {name}")
        return None
    student = cfg['builder'](device=device, **cfg['kwargs'])
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model', ckpt)
    missing, unexpected = student.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  WARNING: {len(missing)} missing keys")
    epoch = ckpt.get('epoch', '?')
    print(f"  Loaded epoch {epoch}")
    student.eval()
    return student


# ── Multi-view reconstruction + evaluation ──────────────────────────

def evaluate_scene_multiview(model, model_name, seq_path, img_indices, device,
                             n_views=5, niter=300, export_ply=False, ply_dir=None):
    """
    Pick n_views images from the scene, run global alignment, evaluate against GT.
    """
    seq_name = os.path.basename(seq_path)

    # Pick n_views images that have GT
    valid_imgs = []
    for idx in sorted(img_indices):
        img_path = os.path.join(seq_path, f"{idx:08d}.jpg")
        gt = load_gt_for_image(seq_path, idx)
        if gt is not None and os.path.isfile(img_path):
            valid_imgs.append((idx, img_path, gt))
        if len(valid_imgs) >= n_views:
            break

    if len(valid_imgs) < 2:
        return None

    filelist = [v[1] for v in valid_imgs]
    gt_data = [v[2] for v in valid_imgs]

    # Load images and make pairs
    try:
        images = load_images(filelist, size=512, verbose=False)
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    except Exception as e:
        print(f"  [{model_name}] Scene {seq_name}: image loading failed: {e}")
        return None

    # Inference
    try:
        with torch.no_grad():
            output = inference(pairs, model, device, batch_size=1, verbose=False)
    except Exception as e:
        print(f"  [{model_name}] Scene {seq_name}: inference failed: {e}")
        return None

    # Global alignment
    try:
        ga = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
        ga.compute_global_alignment(init='mst', niter=niter, schedule='cosine', lr=0.01)
        pts3d_list = to_numpy(ga.get_pts3d())
        depthmaps = to_numpy(ga.get_depthmaps())
        masks = to_numpy(ga.get_masks())
        poses_pred = ga.get_im_poses().detach().cpu().numpy()  # (N, 4, 4)
    except Exception as e:
        print(f"  [{model_name}] Scene {seq_name}: alignment failed: {e}")
        return None

    n = len(valid_imgs)
    Hp, Wp = pts3d_list[0].shape[:2]

    # Collect all pred + GT 3D points for scene-level Sim(3)
    all_pred, all_gt, all_valid = [], [], []
    gt_poses = [v['pose'] for v in gt_data]

    for vi in range(n):
        gt_depth = gt_data[vi]['depth']
        gt_K = gt_data[vi]['K']
        gt_pts = backproject(gt_depth, gt_K, Hp, Wp)  # (Hp, Wp, 3)
        # Transform GT to world frame
        pose_w = gt_poses[vi]
        gt_pts_flat = gt_pts.reshape(-1, 3)
        gt_pts_world = (pose_w[:3, :3] @ gt_pts_flat.T).T + pose_w[:3, 3]

        pred_pts_world = pts3d_list[vi].reshape(-1, 3)
        mask_vi = masks[vi].reshape(-1) if masks[vi].dtype == bool else masks[vi].reshape(-1) > 0.5

        # Also filter invalid GT depth
        gt_depth_r = cv2.resize(gt_depth, (Wp, Hp), interpolation=cv2.INTER_NEAREST).reshape(-1)
        valid = mask_vi & (gt_depth_r > 0.01) & (gt_depth_r < 100) & np.isfinite(pred_pts_world).all(axis=1)

        all_pred.append(pred_pts_world[valid])
        all_gt.append(gt_pts_world[valid])
        all_valid.append(valid)

    pred_combined = np.vstack(all_pred)
    gt_combined = np.vstack(all_gt)

    if len(pred_combined) < 100:
        print(f"  [{model_name}] Scene {seq_name}: too few valid points ({len(pred_combined)})")
        return None

    # Scene-level Sim(3) alignment
    n_sub = min(10000, len(pred_combined))
    rng = np.random.default_rng(0)
    idx_sub = rng.choice(len(pred_combined), n_sub, replace=False)
    scale, R, t = umeyama_alignment(pred_combined[idx_sub], gt_combined[idx_sub])

    # Apply alignment to all points
    aligned_combined = scale * (pred_combined @ R.T) + t
    errors_3d = np.linalg.norm(aligned_combined - gt_combined, axis=1)

    # Per-view depth metrics
    depth_abs_rels, depth_delta1s, depth_rmses = [], [], []
    offset = 0
    for vi in range(n):
        nv = all_pred[vi].shape[0]
        if nv < 10:
            offset += nv
            continue
        aligned_v = aligned_combined[offset:offset + nv]
        gt_v = gt_combined[offset:offset + nv]
        # Depth = Z component (in aligned world frame, compare Z)
        pred_z = aligned_v[:, 2]
        gt_z = gt_v[:, 2]
        valid_z = (gt_z > 0.01) & np.isfinite(pred_z)
        if valid_z.sum() > 10:
            pred_z_v = pred_z[valid_z]
            gt_z_v = gt_z[valid_z]
            thresh = np.maximum(gt_z_v / (pred_z_v + 1e-8), pred_z_v / (gt_z_v + 1e-8))
            depth_delta1s.append((thresh < 1.25).mean())
            depth_abs_rels.append(np.mean(np.abs(gt_z_v - pred_z_v) / (gt_z_v + 1e-8)))
            depth_rmses.append(np.sqrt(np.mean((gt_z_v - pred_z_v) ** 2)))
        offset += nv

    # Pose error (using predicted poses vs GT)
    rra_list, rta_list = pose_angular_error(poses_pred, np.array(gt_poses))

    # Chamfer-like metrics: accuracy + completeness
    # accuracy: mean dist from pred to nearest GT
    # completeness: mean dist from GT to nearest pred
    # (subsample for speed)
    n_chamfer = min(50000, len(aligned_combined))
    idx_c = rng.choice(len(aligned_combined), n_chamfer, replace=False)
    from scipy.spatial import cKDTree
    tree_gt = cKDTree(gt_combined[idx_c])
    tree_pred = cKDTree(aligned_combined[idx_c])
    d_pred2gt, _ = tree_gt.query(aligned_combined[idx_c])
    d_gt2pred, _ = tree_pred.query(gt_combined[idx_c])
    accuracy = np.mean(d_pred2gt)
    completeness = np.mean(d_gt2pred)
    chamfer = (accuracy + completeness) / 2

    results = {
        "scene": seq_name,
        "n_views": n,
        "n_points": len(pred_combined),
        "pts3d_mean": float(errors_3d.mean()),
        "pts3d_median": float(np.median(errors_3d)),
        "pts3d_90pct": float(np.percentile(errors_3d, 90)),
        "accuracy": float(accuracy),
        "completeness": float(completeness),
        "chamfer": float(chamfer),
        "sim3_scale": float(scale),
        "abs_rel": float(np.mean(depth_abs_rels)) if depth_abs_rels else float('nan'),
        "delta1": float(np.mean(depth_delta1s)) if depth_delta1s else float('nan'),
        "rmse": float(np.mean(depth_rmses)) if depth_rmses else float('nan'),
        "RRA_median": float(np.median(rra_list)) if rra_list else float('nan'),
        "RTA_median": float(np.median(rta_list)) if rta_list else float('nan'),
        "RRA_mean": float(np.mean(rra_list)) if rra_list else float('nan'),
        "RTA_mean": float(np.mean(rta_list)) if rta_list else float('nan'),
    }

    # Export PLY (colored) + depth images
    if export_ply and ply_dir:
        out_dir = Path(ply_dir) / model_name / seq_name
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            # ── Collect RGB colors from input images ──
            all_colors_pred, all_colors_gt = [], []
            for vi in range(n):
                img = cv2.imread(filelist[vi])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_r = cv2.resize(img, (Wp, Hp))  # match prediction resolution
                colors_flat = img_r.reshape(-1, 3)  # (H*W, 3)
                valid_vi = all_valid[vi]
                all_colors_pred.append(colors_flat[valid_vi])
                all_colors_gt.append(colors_flat[valid_vi])
            colors_pred = np.vstack(all_colors_pred)  # (N, 3) uint8
            colors_gt = np.vstack(all_colors_gt)

            # ── Write colored PLY files ──
            def write_ply(path, points, colors):
                """Write a simple colored PLY file."""
                n_pts = len(points)
                with open(path, 'w') as f:
                    f.write("ply\nformat ascii 1.0\n")
                    f.write(f"element vertex {n_pts}\n")
                    f.write("property float x\nproperty float y\nproperty float z\n")
                    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
                    f.write("end_header\n")
                    for i in range(n_pts):
                        f.write(f"{points[i,0]:.4f} {points[i,1]:.4f} {points[i,2]:.4f} "
                                f"{colors[i,0]} {colors[i,1]} {colors[i,2]}\n")

            # Subsample for manageable file size (max 200k points)
            max_ply = 200000
            if len(aligned_combined) > max_ply:
                ply_idx = rng.choice(len(aligned_combined), max_ply, replace=False)
            else:
                ply_idx = np.arange(len(aligned_combined))
            write_ply(str(out_dir / "pred_colored.ply"),
                      aligned_combined[ply_idx], colors_pred[ply_idx])
            write_ply(str(out_dir / "gt_colored.ply"),
                      gt_combined[ply_idx], colors_gt[ply_idx])

            # ── Save depth map images (pred vs GT) per view ──
            offset_vis = 0
            for vi in range(n):
                nv = all_pred[vi].shape[0]
                img_name = Path(filelist[vi]).stem

                # Pred depth map (from depthmaps output of global_aligner)
                pred_depth_map = depthmaps[vi]  # (Hp, Wp)
                gt_depth_vi = gt_data[vi]['depth']
                gt_depth_r = cv2.resize(gt_depth_vi, (Wp, Hp), interpolation=cv2.INTER_NEAREST)

                # Scale-align pred depth to GT (median ratio) for visualization
                valid_depth = (gt_depth_r > 0.01) & (gt_depth_r < 100) & (pred_depth_map > 0.01) & np.isfinite(pred_depth_map)
                if valid_depth.sum() > 10:
                    scale_ratio = np.median(gt_depth_r[valid_depth]) / np.median(pred_depth_map[valid_depth])
                    pred_depth_aligned = pred_depth_map * scale_ratio
                else:
                    pred_depth_aligned = pred_depth_map

                # Use GT range for both
                gt_valid = gt_depth_r[gt_depth_r > 0.01]
                vmin = np.percentile(gt_valid, 2) if len(gt_valid) > 10 else 0
                vmax = np.percentile(gt_valid, 98) if len(gt_valid) > 10 else 10

                # Mask invalid GT zones on pred too
                gt_mask = (gt_depth_r > 0.01) & (gt_depth_r < 100)
                gt_depth_vis = np.where(gt_mask, gt_depth_r, np.nan)
                pred_depth_vis = np.where(gt_mask, pred_depth_aligned, np.nan)

                # Error map
                error_map = np.abs(pred_depth_aligned - gt_depth_r)
                error_map[~valid_depth] = np.nan

                fig, axes = plt.subplots(1, 4, figsize=(24, 5))
                # Input image
                img_rgb = cv2.cvtColor(cv2.imread(filelist[vi]), cv2.COLOR_BGR2RGB)
                img_rgb = cv2.resize(img_rgb, (Wp, Hp))
                axes[0].imshow(img_rgb)
                axes[0].set_title(f"View {vi}: {img_name}")
                axes[0].axis('off')
                # GT depth (masked)
                axes[1].imshow(gt_depth_vis, cmap='turbo', vmin=vmin, vmax=vmax)
                axes[1].set_title("GT Depth")
                axes[1].axis('off')
                # Pred depth (scale-aligned, same mask as GT)
                axes[2].imshow(pred_depth_vis, cmap='turbo', vmin=vmin, vmax=vmax)
                axes[2].set_title(f"Pred Depth (x{scale_ratio:.2f})" if valid_depth.sum() > 10 else "Pred Depth")
                axes[2].axis('off')
                # Error map
                emax = np.percentile(error_map[valid_depth], 95) if valid_depth.sum() > 10 else 1
                axes[3].imshow(error_map, cmap='hot', vmin=0, vmax=emax)
                axes[3].set_title("Abs Error")
                axes[3].axis('off')

                plt.tight_layout()
                plt.savefig(str(out_dir / f"depth_view{vi}_{img_name}.png"), dpi=120, bbox_inches='tight')
                plt.close(fig)

                # Also save input image alone
                cv2.imwrite(str(out_dir / f"input_view{vi}_{img_name}.jpg"),
                           cv2.imread(filelist[vi]))

                offset_vis += nv

            print(f"    Exported: {out_dir} (PLY + depth images)")
        except Exception as e:
            print(f"  Warning: export failed: {e}")

    return results


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-view eval on BlendedMVS test split")
    parser.add_argument("--n_views", type=int, default=5, help="Number of views per scene")
    parser.add_argument("--max_scenes", type=int, default=10, help="Max test scenes to evaluate")
    parser.add_argument("--niter", type=int, default=300, help="Global alignment iterations")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--students", nargs="*", default=None)
    parser.add_argument("--include_teacher", action="store_true")
    parser.add_argument("--export_ply", action="store_true")
    parser.add_argument("--max_ply_scenes", type=int, default=0,
                        help="Limit PLY export to N best scenes (by chamfer). 0=all")
    parser.add_argument("--ply_dir", default="ply_multiview")
    parser.add_argument("--output", default="eval_blendedmvs_multiview_results.json")
    parser.add_argument("--checkpoint_epoch", type=str, default=None,
                        help="Override checkpoint epoch for all students (e.g. '40', 'best', 'last')")
    args = parser.parse_args()

    # Override checkpoint paths if requested
    if args.checkpoint_epoch is not None:
        for name, cfg in STUDENT_CONFIGS.items():
            base_dir = os.path.dirname(cfg['ckpt'])
            cfg['ckpt'] = os.path.join(base_dir, f"checkpoint-{args.checkpoint_epoch}.pth")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Get test scenes
    scenes = get_test_scenes(BMVS_ROOT)
    scene_list = sorted(scenes.keys())
    print(f"Found {len(scene_list)} test scenes")

    # Limit
    rng = np.random.default_rng(42)
    if len(scene_list) > args.max_scenes:
        chosen = rng.choice(len(scene_list), args.max_scenes, replace=False)
        scene_list = [scene_list[i] for i in sorted(chosen)]
    print(f"Evaluating {len(scene_list)} scenes with {args.n_views} views each\n")

    all_results = {}

    def eval_model(model, model_name):
        scene_results = []
        for si, seq in enumerate(scene_list):
            seq_path = os.path.join(BMVS_ROOT, seq)
            img_indices = sorted(scenes[seq])
            print(f"  [{model_name}] Scene {si+1}/{len(scene_list)}: {seq} "
                  f"({len(img_indices)} available images)")
            # Only export PLY for first N scenes (or all if max_ply_scenes==0)
            do_ply = args.export_ply and (
                args.max_ply_scenes == 0 or si < args.max_ply_scenes)
            r = evaluate_scene_multiview(
                model, model_name, seq_path, img_indices, device,
                n_views=args.n_views, niter=args.niter,
                export_ply=do_ply, ply_dir=args.ply_dir)
            if r is not None:
                scene_results.append(r)
                print(f"    chamfer={r['chamfer']:.4f}  abs_rel={r['abs_rel']:.4f}  "
                      f"delta1={r['delta1']:.4f}  RRA={r['RRA_median']:.2f}°  "
                      f"RTA={r['RTA_median']:.2f}°")

        # Aggregate
        if not scene_results:
            return {}
        agg = {}
        for key in scene_results[0]:
            if key in ('scene', 'n_views', 'n_points'):
                continue
            vals = [r[key] for r in scene_results if not np.isnan(r.get(key, float('nan')))]
            if vals:
                agg[key] = float(np.mean(vals))
                agg[key + "_std"] = float(np.std(vals))
        agg['n_scenes'] = len(scene_results)
        agg['per_scene'] = scene_results
        return agg

    # Teacher
    if args.include_teacher:
        print("=" * 60)
        print("Evaluating Teacher")
        print("=" * 60)
        teacher = load_teacher(device)
        all_results["Teacher"] = eval_model(teacher, "Teacher")
        del teacher
        torch.cuda.empty_cache()

    # Students
    student_names = args.students or list(STUDENT_CONFIGS.keys())
    for name in student_names:
        if name not in STUDENT_CONFIGS:
            continue
        print(f"\n{'=' * 60}")
        print(f"Evaluating {name}")
        print("=" * 60)
        model = load_student(name, STUDENT_CONFIGS[name], device)
        if model is None:
            continue
        all_results[name] = eval_model(model, name)
        del model
        torch.cuda.empty_cache()

    # Save
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")

    # Summary table
    print(f"\n{'=' * 100}")
    print(f"{'Model':<20} {'chamfer':>8} {'accuracy':>9} {'complete':>9} "
          f"{'abs_rel':>8} {'delta1':>7} {'RMSE':>7} "
          f"{'RRA°':>7} {'RTA°':>7} {'scenes':>6}")
    print(f"{'-' * 100}")
    for name, r in all_results.items():
        if not r:
            continue
        print(f"{name:<20} {r.get('chamfer',0):>8.4f} {r.get('accuracy',0):>9.4f} "
              f"{r.get('completeness',0):>9.4f} {r.get('abs_rel',0):>8.4f} "
              f"{r.get('delta1',0):>7.4f} {r.get('rmse',0):>7.4f} "
              f"{r.get('RRA_median',0):>7.2f} {r.get('RTA_median',0):>7.2f} "
              f"{r.get('n_scenes',0):>6d}")


if __name__ == "__main__":
    main()
