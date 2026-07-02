#!/usr/bin/env python3
"""
eval_single_job.py — Evaluate a SINGLE model on all GT folders.

Designed to be called from eval_moon_rfd_ablation.sh as parallel SLURM jobs.
Each job evaluates one model (Teacher / moon_rfd_only / moon_feat_rfd) on all
3 GT folders, then saves per-model CSV + timing JSON.

Usage:
    python evaluation/moon/eval_single_job.py \
        --model Teacher \
        --output_root results/moon/rfd_ablation \
        --max_pairs 50

    python evaluation/moon/eval_single_job.py \
        --model moon_rfd_only \
        --output_root results/moon/rfd_ablation
"""

import sys
import os
import argparse
import time
import json
import warnings

warnings.simplefilter("ignore")

# ── Project root setup (this file lives at evaluation/moon/eval_single_job.py)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, _THIS_DIR)  # for eval_emat, MAST3RUtils, moon_eval

import mast3r.utils.path_to_dust3r  # noqa

import torch
import numpy as np
import pandas as pd
from pathlib import Path

import eval_emat as em
from moon_eval import MoonEvaluator
from moon_eval.reporter import save_results, print_summary_table, print_efficiency_summary


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate one model on lunar GT data (parallel SLURM job)."
    )
    p.add_argument("--model", required=True,
                   help="Model name: Teacher, moon_rfd_only, moon_feat_rfd, "
                        "S1_MobileNet, S2_ViT-Small, etc.")
    p.add_argument("--output_root", required=True,
                   help="Root output directory (per-model subdir created inside)")
    p.add_argument("--mode", default="global_aligner",
                   choices=["global_aligner", "sparse_ga", "sparse_ga_depth"],
                   help="Reconstruction mode (default: global_aligner)")
    p.add_argument("--gt_folders", nargs="+", default=["nadir", "pitch", "landing"],
                   help="GT folder keys to evaluate (default: nadir pitch landing)")
    p.add_argument("--max_pairs", type=int, default=None,
                   help="Max pairs per GT folder (None = all)")
    p.add_argument("--save_viz", action="store_true",
                   help="Save per-pair diagnostic visualizations")
    p.add_argument("--no_ply", action="store_true",
                   help="Skip saving PLY point clouds")
    p.add_argument("--force", action="store_true",
                   help="Force re-evaluation even if pair_report.txt already exists")
    p.add_argument("--device", default="cuda",
                   help="Torch device (default: cuda)")
    p.add_argument("--ckpt", default=None,
                   help="Override checkpoint path (ignores STUDENT_CONFIGS ckpt)")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n[{args.model}] Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}  "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load model ────────────────────────────────────────────────────────────
    if args.model == "Teacher":
        model = em.load_teacher(device)
    else:
        if args.model not in em.STUDENT_CONFIGS:
            raise ValueError(
                f"Unknown model '{args.model}'.\n"
                f"Available: Teacher, {', '.join(em.STUDENT_CONFIGS)}"
            )
        cfg = em.STUDENT_CONFIGS[args.model]
        print(f"Building {args.model}...")
        student = cfg["builder"](device=str(device), **cfg["kwargs"])
        ckpt_path = args.ckpt if args.ckpt else cfg["ckpt"]
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
        # Move any unregistered tensors to device
        for name, module in student.named_modules():
            for attr, val in list(vars(module).items()):
                if isinstance(val, torch.Tensor) and val.device != device:
                    setattr(module, attr, val.to(device))
        student.eval()
        model = student

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {args.model}: {n_params:.1f}M params")

    # ── Output dirs ───────────────────────────────────────────────────────────
    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Evaluate on each GT folder ────────────────────────────────────────────
    all_results = []
    t_start = time.time()

    for folder_key in args.gt_folders:
        gt_folder = em.GT_FOLDERS.get(folder_key)
        if gt_folder is None or not Path(gt_folder).exists():
            print(f"\n  [SKIP] GT folder '{folder_key}' not found")
            continue

        n_pairs = len(list(Path(gt_folder).glob("*.jpg"))) // 2
        print(f"\n{'#' * 70}")
        print(f"  {args.model} — {folder_key}  ({n_pairs} pairs total)")
        print(f"{'#' * 70}")

        evaluator = MoonEvaluator(
            model=model,
            model_name=args.model,
            device=device,
            K_GT=em.K_GT,
            output_root=str(out_dir),
            mode=args.mode,
            use_icp=True,
            save_ply=not args.no_ply,
            save_viz=args.save_viz,
            n_matches=2000,
            emat_threshold=1.0,
            verbose=False,
        )
        results = evaluator.evaluate_folder(
            gt_folder, folder_key,
            max_pairs=args.max_pairs,
            force=args.force,
        )
        dt = time.time() - t_start
        n_ok = sum(1 for r in results if "error" not in r)
        print(f"  {n_ok}/{len(results)} valid pairs ({dt:.1f}s)")
        all_results.extend(results)

    total_time = time.time() - t_start
    print(f"\n[{args.model}] Total: {total_time:.1f}s")

    if not all_results:
        safe_model = args.model.replace("/", "_").replace(" ", "_")
        slice_path = out_dir / f"results_{safe_model}.csv"
        if slice_path.exists():
            print(f"All pairs already done. Existing: {slice_path}")
        else:
            print("No results produced.")
        return

    df_new = pd.DataFrame(all_results)

    # ── Save per-model CSV slice (merge with previous if re-running) ──────────
    safe_model = args.model.replace("/", "_").replace(" ", "_")
    slice_path = out_dir / f"results_{safe_model}.csv"
    if slice_path.exists():
        df_prev = pd.read_csv(slice_path)
        new_pairs = set(df_new["pair"])
        df_kept = df_prev[~df_prev["pair"].isin(new_pairs)]
        df = pd.concat([df_kept, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(slice_path, index=False)
    print(f"\nSlice saved: {slice_path}  ({len(df_new)} new + {len(df) - len(df_new)} previous)")

    # ── Also update shared results_all.csv at output_root ────────────────────
    all_csv = Path(args.output_root) / "results_all.csv"
    if all_csv.exists():
        df_existing = pd.read_csv(all_csv)
        df_existing = df_existing[df_existing["Model"] != args.model]
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(all_csv, index=False)
    print(f"Combined: {all_csv}  ({len(df)} total rows)")

    # ── Quick summary tables ──────────────────────────────────────────────────
    df_this = df[df["Model"] == args.model]
    print_summary_table(df_this)
    print_efficiency_summary(df_this)

    # ── Timing JSON ───────────────────────────────────────────────────────────
    info_path = out_dir / f"timing_{safe_model}.json"
    with open(info_path, "w") as f:
        json.dump({
            "model": args.model,
            "mode": args.mode,
            "n_params_M": round(n_params, 1),
            "total_time_s": round(total_time, 1),
            "n_pairs": len(all_results),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }, f, indent=2)


if __name__ == "__main__":
    main()
