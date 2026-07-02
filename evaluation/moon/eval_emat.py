#!/usr/bin/env python3
"""
Évaluation RRA / RTA via Essential Matrix pour Teacher + 5 Students.
Usage: python eval_emat.py [--max_pairs N] [--threshold T] [--output_dir DIR]
"""

import sys
import os
import argparse
import time
import json

import numpy as np
import torch
import cv2
import pandas as pd
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

import mast3r.utils.path_to_dust3r  # noqa
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.utils.image import load_images
from distillation_dual import (
    build_mobilenet_student,
    build_vit_student,
    build_vit_tiny_student,
)


# =============================================================================
# Métriques
# =============================================================================
def pose_error(R_est, R_gt, t_est, t_gt):
    """max(RRA, RTA) — standard pose error for AUC computation."""
    rra = rra_deg(R_est, R_gt)
    rta = rta_deg(t_est, t_gt)
    return max(rra, rta)


def compute_auc(errors, thresholds=(5, 10, 20)):
    """
    Compute AUC of the pose error recall curve at given thresholds (degrees).
    Standard protocol from Roma / DUSt3R / VGGT.
    errors: list of max(RRA, RTA) per pair (inf for failures).
    Returns dict like {'AUC@5': 0.45, 'AUC@10': 0.62, 'AUC@20': 0.78}
    """
    errors = np.array(errors, dtype=np.float64)
    result = {}
    for t in thresholds:
        # recall curve: for each angle x in [0, t], fraction of errors <= x
        n_bins = t * 10  # 0.1° resolution
        x = np.linspace(0, t, n_bins + 1)
        recalls = np.array([(errors <= xi).mean() for xi in x])
        auc = np.trapz(recalls, x) / t  # normalize by threshold
        result[f"AUC@{t}"] = round(auc * 100, 2)  # percentage
    return result


def rra_deg(R_est, R_gt):
    R_err = R_est @ R_gt.T
    return np.degrees(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1)))


def rta_deg(t_est, t_gt):
    te = t_est / (np.linalg.norm(t_est) + 1e-12)
    tg = t_gt / (np.linalg.norm(t_gt) + 1e-12)
    return np.degrees(np.arccos(np.clip(np.dot(te, tg), -1, 1)))


def compute_pose_essential(pts_im0, pts_im1, K, threshold=1.0):
    """
    Relative pose via Essential matrix (5-point algorithm + RANSAC).
    Returns R, t (unit vector), n_inliers.
    """
    pts1 = pts_im0.astype(np.float64)
    pts2 = pts_im1.astype(np.float64)

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


# =============================================================================
# Model loading
# =============================================================================
STUDENT_CONFIGS = {
    "S1_MobileNet": dict(
        builder=build_mobilenet_student,
        kwargs=dict(dec_embed_dim=512, dec_depth=6, dec_num_heads=4, mlp_ratio=1.0,
                    backbone_name="mobilenetv3_large_100"),
        ckpt="output/gt_s1_mobilenet/checkpoint-best.pth",
    ),
    "S2_ViT-Small": dict(
        builder=build_vit_student,
        kwargs=dict(dec_embed_dim=512, dec_depth=6, dec_num_heads=4, mlp_ratio=1.0,
                    prefer_dinov2=True, freeze_backbone=False),
        ckpt="output/gt_s2_vit_small/checkpoint-best.pth",
    ),
    "S3_ViT-Small-Frozen": dict(
        builder=build_vit_student,
        kwargs=dict(dec_embed_dim=512, dec_depth=6, dec_num_heads=4, mlp_ratio=1.0,
                    prefer_dinov2=True, freeze_backbone=False),
        ckpt="output/gt_s3_vit_small_frozen/checkpoint-best.pth",
    ),
    "S4_ViT-Small-Reduced": dict(
        builder=build_vit_student,
        kwargs=dict(dec_embed_dim=256, dec_depth=4, dec_num_heads=4, mlp_ratio=1.0,
                    prefer_dinov2=True, freeze_backbone=False),
        ckpt="output/gt_s4_vit_small_reduced/checkpoint-best.pth",
    ),
    "S5_ViT-Tiny": dict(
        builder=build_vit_tiny_student,
        kwargs=dict(dec_embed_dim=512, dec_depth=6, dec_num_heads=4, mlp_ratio=1.0,
                    model_name="vit_tiny_patch16_224", freeze_backbone=False, img_size=512),
        ckpt="output/gt_s5_vit_tiny/checkpoint-best.pth",
    ),
    # DUNE-backbone students (distill3r_cmp run)
    "S6_Distill3R": dict(
        builder=build_vit_student,
        kwargs=dict(dec_embed_dim=384, dec_depth=6, dec_num_heads=6, mlp_ratio=1.0,
                    backbone_type="dune"),
        ckpt="output/distill3r_cmp_s6_baseline/checkpoint-best.pth",
    ),
    "S7_Hybrid": dict(
        builder=build_vit_student,
        kwargs=dict(dec_embed_dim=384, dec_depth=6, dec_num_heads=6, mlp_ratio=1.0,
                    backbone_type="dune"),
        ckpt="output/distill3r_cmp_s7_hybrid/checkpoint-best.pth",
    ),
    "S10_DUNE-Full": dict(
        builder=build_vit_student,
        kwargs=dict(dec_embed_dim=512, dec_depth=6, dec_num_heads=4, mlp_ratio=1.0,
                    backbone_type="dune"),
        ckpt="output/distill3r_cmp_s10_dune_full/checkpoint-best.pth",
    ),
    # ── Moon RFD ablation (train_moon_rfd_ablation.sh) ──────────────────
    "moon_rfd_only": dict(
        builder=build_vit_student,
        kwargs=dict(dec_embed_dim=512, dec_depth=6, dec_num_heads=4, mlp_ratio=1.0,
                    prefer_dinov2=True, backbone_type="dinov2"),
        ckpt="output/moon_rfd_only/checkpoint-50.pth",
    ),
    "moon_feat_rfd": dict(
        builder=build_vit_student,
        kwargs=dict(dec_embed_dim=512, dec_depth=6, dec_num_heads=4, mlp_ratio=1.0,
                    prefer_dinov2=True, backbone_type="dinov2"),
        ckpt="output/moon_feat_rfd/checkpoint-50.pth",
    ),
    # ── Moon tiny/reduced (train_moon_tiny_reduced_feat_rfd.sh) ────────
    "moon_tiny_feat_rfd": dict(
        builder=build_vit_tiny_student,
        kwargs=dict(dec_embed_dim=512, dec_depth=6, dec_num_heads=4, mlp_ratio=1.0),
        ckpt="output/moon_tiny_feat_rfd/checkpoint-best.pth",
    ),
    "moon_reduced_feat_rfd": dict(
        builder=build_vit_student,
        kwargs=dict(dec_embed_dim=256, dec_depth=4, dec_num_heads=4, mlp_ratio=1.0,
                    prefer_dinov2=True, backbone_type="dinov2"),
        ckpt="output/moon_reduced_feat_rfd/checkpoint-best.pth",
    ),
}

# K_gt for 512x384 cropped images (center crop from 512x512)
K_GT = np.array([
    [618.0387,   0.,      256.],
    [  0.,     618.0387,  192.],
    [  0.,       0.,        1.]
], dtype=np.float64)

GT_FOLDERS = {
    "nadir":   Path("Datas/TESTS/test_image_clean"),
    "pitch":   Path("Datas/TESTS/test_image_clean_pitch"),
    "landing": Path("Datas/TESTS/test_image_clean_landing"),
}


def load_teacher(device):
    teacher_ckpt = "MOONSt3R.pth"
    print(f"Loading teacher from {teacher_ckpt}...")
    ckpt_data = torch.load(teacher_ckpt, map_location="cpu")
    args_str = ckpt_data["args"].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if "landscape_only" not in args_str:
        args_str = args_str[:-1] + ", landscape_only=False)"
    else:
        args_str = args_str.replace(" ", "").replace(
            "landscape_only=True", "landscape_only=False"
        )
    inf = float("inf")  # noqa: F841 — needed by eval()
    model = eval(args_str)
    model.load_state_dict(ckpt_data["model"], strict=False)
    model = model.to(device).eval()
    # Ensure any plain torch.Tensor attributes attached to modules are moved to device.
    def _move_unregistered_tensors(mod, device):
        moved = []
        for name, module in mod.named_modules():
            for attr, val in list(vars(module).items()):
                if isinstance(val, torch.Tensor):
                    if val.device != device:
                        setattr(module, attr, val.to(device))
                        moved.append(f"{name}.{attr}" if name else attr)
        if moved:
            print(f"  Moved {len(moved)} unregistered tensors to {device}: {moved[:10]}{'...' if len(moved)>10 else ''}")

    _move_unregistered_tensors(model, device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Teacher loaded: {n_params:.1f}M params")
    return model


def load_students(device):
    models = {}
    for sname, cfg in STUDENT_CONFIGS.items():
        print(f"Building {sname}...")
        student = cfg["builder"](device=str(device), **cfg["kwargs"])
        ckpt_path = cfg["ckpt"]
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt.get("model", ckpt)
            missing, unexpected = student.load_state_dict(state_dict, strict=False)
            print(f"  Loaded: {ckpt_path}")
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
        else:
            print(f"  WARNING: checkpoint not found at {ckpt_path}")
        student.eval()
        # also move any unregistered tensors on students
        def _move_unregistered_tensors_student(mod, device):
            for name, module in mod.named_modules():
                for attr, val in list(vars(module).items()):
                    if isinstance(val, torch.Tensor):
                        if val.device != device:
                            setattr(module, attr, val.to(device))
        _move_unregistered_tensors_student(student, torch.device(str(device)))
        models[sname] = student
        n_params = sum(p.numel() for p in student.parameters()) / 1e6
        print(f"  {sname}: {n_params:.1f}M params")
    return models


# =============================================================================
# Evaluation loop
# =============================================================================
def evaluate_model_on_folder(
    model, model_name, device, gt_folder, K_gt,
    max_pairs=None, n_matches=2000, border=3, threshold=1.0,
):
    gt_images = sorted(gt_folder.glob("*.jpg"))
    gt_pairs = list(zip(gt_images[0::2], gt_images[1::2]))

    if max_pairs is not None and max_pairs < len(gt_pairs):
        step = max(1, len(gt_pairs) // max_pairs)
        gt_pairs = gt_pairs[::step][:max_pairs]

    results = []
    for img0_path, img1_path in gt_pairs:
        name0, name1 = img0_path.stem, img1_path.stem

        # GT relative pose
        data0 = np.load(gt_folder / f"{name0}.npz")
        data1 = np.load(gt_folder / f"{name1}.npz")
        T_w_c0 = data0["cam2world"].astype(np.float64)
        T_w_c1 = data1["cam2world"].astype(np.float64)
        T_rel_gt = np.linalg.inv(T_w_c1) @ T_w_c0
        R_gt = T_rel_gt[:3, :3]
        t_gt = T_rel_gt[:3, 3]

        try:
            images = load_images(
                [str(img0_path), str(img1_path)], size=512, verbose=False
            )
            with torch.no_grad():
                output = inference(
                    [tuple(images)], model, device, batch_size=1, verbose=False
                )

            pred1, pred2 = output["pred1"], output["pred2"]
            view1, view2 = output["view1"], output["view2"]

            desc1 = pred1["desc"].squeeze(0).detach()
            desc2 = pred2["desc"].squeeze(0).detach()
            matches_im0, matches_im1 = fast_reciprocal_NNs(
                desc1, desc2,
                subsample_or_initxy1=8,
                device=device,
                dist="dot",
                block_size=2**13,
            )

            # Border filter
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

            # Top N by combined confidence
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
            n_keep = min(n_matches, len(matches_im0))
            top_idx = np.argsort(combined_conf)[::-1][:n_keep]
            m0 = matches_im0[top_idx]
            m1 = matches_im1[top_idx]

            # Essential matrix
            R_est, t_est, n_inliers = compute_pose_essential(
                m0.astype(np.float64), m1.astype(np.float64), K_gt, threshold=threshold
            )

            if R_est is not None:
                rra = rra_deg(R_est, R_gt)
                rta = rta_deg(t_est, t_gt)
                pe = max(rra, rta)
            else:
                rra, rta, pe = np.inf, np.inf, np.inf
                n_inliers = 0

            results.append({
                "Model": model_name,
                "Pair": f"{name0}_{name1}",
                "RRA": round(rra, 4),
                "RTA": round(rta, 4),
                "PoseError": round(pe, 4),
                "Inliers": n_inliers,
                "Total": len(m0),
            })

        except Exception as e:
            print(f"  {name0}_{name1}: EXCEPTION {e}")
            results.append({
                "Model": model_name,
                "Pair": f"{name0}_{name1}",
                "RRA": float("inf"),
                "RTA": float("inf"),
                "Inliers": 0,
                "Total": 0,
            })

    return results


def print_summary(df, title):
    df_valid = df[df["RRA"] < 180].copy()
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

    model_names = df_valid["Model"].unique()
    rows = []
    for mname in model_names:
        ms = df_valid[df_valid["Model"] == mname]

        # AUC computation: use ALL pairs (incl. failures = inf)
        ms_all = df[df["Model"] == mname]
        pose_errors = ms_all["PoseError"].values
        auc = compute_auc(pose_errors, thresholds=(5, 10, 20))

        row = {
            "Model": mname,
            "N": len(ms),
            "RRA_med": round(ms["RRA"].median(), 2),
            "RRA_mean": round(ms["RRA"].mean(), 2),
            "RTA_med": round(ms["RTA"].median(), 2),
            "RTA_mean": round(ms["RTA"].mean(), 2),
            "Inliers_med": int(ms["Inliers"].median()),
        }
        row.update(auc)
        for t in [5, 10, 15, 30]:
            row[f"RRA<{t}"] = f"{(ms['RRA'] < t).mean() * 100:.0f}%"
            row[f"RTA<{t}"] = f"{(ms['RTA'] < t).mean() * 100:.0f}%"
        rows.append(row)

    df_summary = pd.DataFrame(rows)
    print(df_summary.to_string(index=False))
    return df_summary


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="E-matrix RRA/RTA evaluation")
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Max pairs per GT folder (None = all)")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="RANSAC inlier threshold in pixels")
    parser.add_argument("--n_matches", type=int, default=2000,
                        help="Number of top matches to keep")
    parser.add_argument("--output_dir", type=str, default="eval_results",
                        help="Output directory for CSV results")
    parser.add_argument("--gt_folders", nargs="+", default=["nadir", "pitch", "landing"],
                        choices=["nadir", "pitch", "landing"],
                        help="GT folders to evaluate on")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset of model keys to evaluate (default: all). "
                             "Use 'teacher' for the teacher. "
                             f"Available students: {list(STUDENT_CONFIGS.keys())}")
    parser.add_argument("--no_teacher", action="store_true",
                        help="Skip the teacher model")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Resolve which models to run
    run_teacher = not args.no_teacher
    student_keys = list(STUDENT_CONFIGS.keys())
    if args.models is not None:
        if "teacher" in args.models:
            run_teacher = True
        else:
            run_teacher = False
        student_keys = [k for k in args.models if k != "teacher" and k in STUDENT_CONFIGS]
        unknown = [k for k in args.models if k != "teacher" and k not in STUDENT_CONFIGS]
        if unknown:
            print(f"WARNING: unknown model keys ignored: {unknown}")

    # Load models
    all_models = {}
    if run_teacher:
        all_models["Teacher"] = load_teacher(device)
    for sname in student_keys:
        cfg = STUDENT_CONFIGS[sname]
        print(f"Building {sname}...")
        student = cfg["builder"](device=str(device), **cfg["kwargs"])
        ckpt_path = cfg["ckpt"]
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt.get("model", ckpt)
            missing, unexpected = student.load_state_dict(state_dict, strict=False)
            print(f"  Loaded: {ckpt_path}")
            if missing:
                print(f"  Missing keys: {len(missing)}")
        else:
            print(f"  WARNING: checkpoint not found at {ckpt_path}")
        student.eval()
        all_models[sname] = student

    # Evaluate per GT folder
    all_results = []
    t_start = time.time()

    for folder_key in args.gt_folders:
        gt_folder = GT_FOLDERS[folder_key]
        if not gt_folder.exists():
            print(f"\nWARNING: {gt_folder} does not exist, skipping")
            continue

        n_pairs = len(list(gt_folder.glob("*.jpg"))) // 2
        print(f"\n{'#'*70}")
        print(f"  GT folder: {folder_key} ({gt_folder}) — {n_pairs} pairs total")
        print(f"{'#'*70}")

        for model_name, model in all_models.items():
            print(f"\n  --- {model_name} ---")
            t0 = time.time()
            results = evaluate_model_on_folder(
                model, model_name, device, gt_folder, K_GT,
                max_pairs=args.max_pairs,
                n_matches=args.n_matches,
                threshold=args.threshold,
            )
            dt = time.time() - t0

            for r in results:
                r["Folder"] = folder_key
            all_results.extend(results)

            n_valid = sum(1 for r in results if r["RRA"] < 180)
            rra_vals = [r["RRA"] for r in results if r["RRA"] < 180]
            rta_vals = [r["RTA"] for r in results if r["RRA"] < 180]
            if rra_vals:
                print(f"    {model_name}/{folder_key}: {n_valid} pairs, "
                      f"RRA_med={np.median(rra_vals):.2f}°, "
                      f"RTA_med={np.median(rta_vals):.2f}°  "
                      f"({dt:.1f}s)")

    total_time = time.time() - t_start
    print(f"\nTotal evaluation time: {total_time:.1f}s")

    # Build dataframe and save
    df_all = pd.DataFrame(all_results)
    csv_path = os.path.join(args.output_dir, "emat_results_all.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"\nAll results saved to {csv_path}")

    # Per-folder summaries
    for folder_key in args.gt_folders:
        df_folder = df_all[df_all["Folder"] == folder_key]
        if len(df_folder) == 0:
            continue
        summary = print_summary(df_folder, f"Essential Matrix — {folder_key}")
        summary_path = os.path.join(args.output_dir, f"emat_summary_{folder_key}.csv")
        summary.to_csv(summary_path, index=False)

    # Global summary
    summary_global = print_summary(df_all, "Essential Matrix — ALL FOLDERS")
    summary_global.to_csv(
        os.path.join(args.output_dir, "emat_summary_global.csv"), index=False
    )

    # Save config
    config = {
        "max_pairs": args.max_pairs,
        "threshold": args.threshold,
        "n_matches": args.n_matches,
        "gt_folders": args.gt_folders,
        "K_gt": K_GT.tolist(),
        "student_configs": {
            k: {"ckpt": v["ckpt"]} for k, v in STUDENT_CONFIGS.items()
        },
        "total_time_s": round(total_time, 1),
    }
    with open(os.path.join(args.output_dir, "eval_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
