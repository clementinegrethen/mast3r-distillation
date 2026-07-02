#!/usr/bin/env python3
"""
evalAllPairs_multi.py

Run the full MASt3REvaluator pipeline (Sim(3) alignment, 3D scene metrics,
2D depth map metrics, terrain analysis) on Teacher + 5 Students, mirroring
the model configs from eval_emat.py.

3D metrics (RMSE, MAE, Chamfer, Pearson) are computed on the merged scene
(view0 + view1). 2D metrics (SSIM, regression) are per-view then averaged.

Optionally loads eval_emat results to compare depth metrics with RRA/RTA.

Usage:
    python evalAllPairs_multi.py --gt_folders landing --max_pairs 2
    python evalAllPairs_multi.py --gt_folders nadir pitch landing
"""

import sys
import os
import argparse
import time
import json

import numpy as np
import torch
import pandas as pd
from pathlib import Path

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
from MASt3REvaluator import MASt3REvaluator


def batch_evaluate_model(
    model, model_name, device, gt_folder, folder_key,
    output_root, max_pairs=None, save_viz=True, use_sparse_ga=False,
    skip_stitch=False, disable_icp=False,
):
    """Run MASt3REvaluator on all pairs for a single model."""
    gt_folder = Path(gt_folder)
    images = sorted(gt_folder.glob("*.jpg"))
    pairs = list(zip(images[0::2], images[1::2]))

    if max_pairs is not None and max_pairs < len(pairs):
        step = max(1, len(pairs) // max_pairs)
        pairs = pairs[::step][:max_pairs]

    results = []

    for img1, img2 in pairs:
        name = img1.stem
        out_dir = Path(output_root) / model_name / folder_key / f"{name}_vs_{img2.stem}"

        evaluator = MASt3REvaluator(
            gt_folder=gt_folder,
            name=name,
            img1=img1,
            img2=img2,
            device=device,
            ckpt_path="unused",
            output_dir=out_dir,
            model=model,
            save_viz=save_viz,
            use_sparse_ga=use_sparse_ga,
            skip_stitch=skip_stitch,
            disable_icp=disable_icp,
        )

        print(f"\n  [{model_name}] {img1.name} vs {img2.name}")
        try:
            metrics, terrain_metrics, profile_metrics, error_stats = (
                evaluator.run_complete_evaluation()
            )

            row = {
                "Model": model_name,
                "Folder": folder_key,
                "Pair": f"{name}_{img2.stem}",
            }
            row.update(metrics)
            results.append(row)

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            results.append({
                "Model": model_name,
                "Folder": folder_key,
                "Pair": f"{name}_{img2.stem}",
                "rmse": float("nan"),
                "mae": float("nan"),
            })

    return results


def load_emat_results(emat_csv):
    """Load eval_emat results CSV for comparison."""
    path = Path(emat_csv)
    if not path.exists():
        print(f"eval_emat results not found at {path}, skipping comparison")
        return None
    df = pd.read_csv(path)
    return df


def compare_with_emat(df_full, df_emat):
    """Print side-by-side comparison with eval_emat RRA/RTA results."""
    if df_emat is None:
        print("\nNo eval_emat results to compare with.")
        return None

    print("\n" + "=" * 80)
    print("  COMPARISON: evalAllPairs depth metrics vs eval_emat RRA/RTA")
    print("=" * 80)

    models = df_full["Model"].unique()
    rows = []
    for mname in models:
        ms_full = df_full[df_full["Model"] == mname]
        rmse_vals = ms_full["rmse"].dropna()
        mae_vals = ms_full["mae"].dropna()

        ms_emat = df_emat[df_emat["Model"] == mname] if df_emat is not None else pd.DataFrame()
        rra_emat = ms_emat["RRA"].replace([np.inf], np.nan).dropna() if len(ms_emat) > 0 else pd.Series(dtype=float)
        rta_emat = ms_emat["RTA"].replace([np.inf], np.nan).dropna() if len(ms_emat) > 0 else pd.Series(dtype=float)

        row = {
            "Model": mname,
            "N_depth": len(rmse_vals),
            "RMSE_med": round(rmse_vals.median(), 3) if len(rmse_vals) > 0 else "N/A",
            "MAE_med": round(mae_vals.median(), 3) if len(mae_vals) > 0 else "N/A",
            "N_emat": len(rra_emat),
            "RRA_emat_med": round(rra_emat.median(), 2) if len(rra_emat) > 0 else "N/A",
            "RTA_emat_med": round(rta_emat.median(), 2) if len(rta_emat) > 0 else "N/A",
        }
        rows.append(row)

    df_cmp = pd.DataFrame(rows)
    print(df_cmp.to_string(index=False))
    return df_cmp


def print_full_summary(df, title):
    """Print summary stats per model: 3D scene metrics + averaged 2D metrics."""
    # Filter out failed pairs (NaN rmse)
    df_valid = df.dropna(subset=["rmse"]).copy()
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

    model_names = df_valid["Model"].unique()
    rows = []
    for mname in model_names:
        ms = df_valid[df_valid["Model"] == mname]

        row = {
            "Model": mname,
            "N": len(ms),
            # 3D scene metrics (merged view0+view1)
            "RMSE_med": round(ms["rmse"].median(), 3) if "rmse" in ms.columns else "N/A",
            "RMSE_mean": round(ms["rmse"].mean(), 3) if "rmse" in ms.columns else "N/A",
            "MAE_med": round(ms["mae"].median(), 3) if "mae" in ms.columns else "N/A",
            "Pearson_med": round(ms["pearson_r"].median(), 4) if "pearson_r" in ms.columns else "N/A",
            "Chamfer_med": round(ms["chamfer_distance"].median(), 3) if "chamfer_distance" in ms.columns else "N/A",
            # Averaged 2D metrics (view0+view1 avg)
            "SSIM_avg_med": round(ms["ssim_adaptive_avg"].median(), 4) if "ssim_adaptive_avg" in ms.columns else "N/A",
            "R2_avg_med": round(ms["regression_r2_avg"].median(), 4) if "regression_r2_avg" in ms.columns else "N/A",
            "Pearson_avg_med": round(ms["pearson_r_avg"].median(), 4) if "pearson_r_avg" in ms.columns else "N/A",
            # Scale
            "Scale_med": round(ms["scale"].median(), 4) if "scale" in ms.columns else "N/A",
            "ScaleErr%_med": round(ms["scale_error_pct"].median(), 2) if "scale_error_pct" in ms.columns else "N/A",
        }
        rows.append(row)

    df_summary = pd.DataFrame(rows)
    print(df_summary.to_string(index=False))
    return df_summary


def main():
    parser = argparse.ArgumentParser(description="Full evaluation (depth+pose) for Teacher + 5 Students")
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Max pairs per GT folder (None = all)")
    parser.add_argument("--output_dir", type=str, default="bonne_nuit",
                        help="Output directory")
    parser.add_argument("--gt_folders", nargs="+", default=["nadir", "pitch", "landing"],
                        choices=["nadir", "pitch", "landing"],
                        help="GT folders to evaluate on")
    parser.add_argument("--emat_csv", type=str, default="eval_results/emat_results_all.csv",
                        help="Path to eval_emat results CSV for comparison")
    parser.add_argument("--no_viz", action="store_true",
                        help="Disable saving visualizations (faster)")
    parser.add_argument("--sparse_ga", action="store_true",
                        help="Use sparse_global_alignment to reconstruct scene (depth+K_GT instead of direct pts3d)")
    parser.add_argument("--no_stitch", action="store_true",
                        help="Disable stitching of views using MASt3R matches")
    parser.add_argument("--no_icp", action="store_true",
                        help="Disable ICP point-to-plane refinement after stitching")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load all models (same as eval_emat.py)
    teacher = load_teacher(device)
    students = load_students(device)
    all_models = {"Teacher": teacher}
    all_models.update(students)

    # Run evaluation
    all_results = []
    t_start = time.time()

    for folder_key in args.gt_folders:
        gt_folder = GT_FOLDERS[folder_key]
        if not gt_folder.exists():
            print(f"\nWARNING: {gt_folder} does not exist, skipping")
            continue

        n_pairs = len(list(gt_folder.glob("*.jpg"))) // 2
        print(f"\n{'#' * 70}")
        print(f"  GT folder: {folder_key} ({gt_folder}) - {n_pairs} pairs total")
        print(f"{'#' * 70}")

        for model_name, model in all_models.items():
            print(f"\n  --- {model_name} ---")
            t0 = time.time()

            results = batch_evaluate_model(
                model, model_name, device, gt_folder, folder_key,
                output_root=args.output_dir,
                max_pairs=args.max_pairs,
                save_viz=not args.no_viz,
                use_sparse_ga=args.sparse_ga,
                skip_stitch=args.no_stitch,
                disable_icp=args.no_icp,
            )
            dt = time.time() - t0

            all_results.extend(results)
            n_valid = sum(1 for r in results if not np.isnan(r.get("rmse", float("nan"))))
            print(f"    {model_name}/{folder_key}: {n_valid}/{len(results)} valid pairs ({dt:.1f}s)")

    total_time = time.time() - t_start
    print(f"\nTotal evaluation time: {total_time:.1f}s")

    # Build DataFrame and save
    df_all = pd.DataFrame(all_results)
    csv_path = os.path.join(args.output_dir, "full_results_all.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"\nAll results saved to {csv_path}")

    # Per-folder summaries
    for folder_key in args.gt_folders:
        df_folder = df_all[df_all["Folder"] == folder_key]
        if len(df_folder) == 0:
            continue
        summary = print_full_summary(df_folder, f"Full Evaluation - {folder_key}")
        summary.to_csv(os.path.join(args.output_dir, f"full_summary_{folder_key}.csv"), index=False)

    # Global summary
    summary_global = print_full_summary(df_all, "Full Evaluation - ALL FOLDERS")
    summary_global.to_csv(os.path.join(args.output_dir, "full_summary_global.csv"), index=False)

    # Compare with eval_emat results
    df_emat = load_emat_results(args.emat_csv)
    df_cmp = compare_with_emat(df_all, df_emat)
    if df_cmp is not None:
        df_cmp.to_csv(os.path.join(args.output_dir, "depth_vs_emat_comparison.csv"), index=False)

    # Save config
    config = {
        "max_pairs": args.max_pairs,
        "gt_folders": args.gt_folders,
        "emat_csv": args.emat_csv,
        "student_configs": {k: {"ckpt": v["ckpt"]} for k, v in STUDENT_CONFIGS.items()},
        "total_time_s": round(total_time, 1),
    }
    with open(os.path.join(args.output_dir, "eval_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nDone. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
