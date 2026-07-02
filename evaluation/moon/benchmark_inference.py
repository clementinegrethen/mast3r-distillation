#!/usr/bin/env python3
"""
Inference Latency & Peak GPU Memory Benchmark.

Distill3R-style sliding-window protocol:
  Given N input views, form N-1 consecutive pairs, run forward pass for all pairs,
  measure wall-clock time and peak GPU VRAM.

Usage:
  python benchmark_inference.py --n_views 12 32 64 96 128

Output:
  - CSV table with columns: Model, Params(M), N, Time(s), Mem(GB)
  - LaTeX table for paper
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.inference import inference, collate_with_cat, loss_of_one_batch
from dust3r.utils.image import load_images

from eval_emat import load_teacher, STUDENT_CONFIGS
from distillation_dual import (
    build_mobilenet_student,
    build_vit_student,
    build_vit_tiny_student,
    build_dinov3_student,
)


def count_params_M(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def make_dummy_pairs(n_views, resolution=512, device="cpu"):
    """Create N-1 consecutive dummy image pairs for sliding-window benchmark."""
    # Generate random images as if loaded by load_images
    pairs = []
    # Pre-generate all dummy views
    views = []
    for i in range(n_views):
        img = torch.randn(1, 3, resolution, resolution, device=device)
        view = {
            'img': img,
            'true_shape': torch.tensor([[resolution, resolution]]),
            'idx': i,
            'instance': str(i),
        }
        views.append(view)

    # Sliding window: consecutive pairs
    for i in range(n_views - 1):
        pairs.append((views[i], views[i + 1]))

    return pairs


@torch.no_grad()
def benchmark_model(model, n_views, resolution, warmup_iters, bench_iters, device):
    """
    Benchmark a model with N views (N-1 pairs) and return time/memory stats.
    Returns dict: {time_s, mem_gb, n_pairs}
    """
    model.eval()
    n_pairs = n_views - 1

    # Warmup: run a few forward passes to stabilize GPU clocks and caches
    dummy_pairs_warmup = make_dummy_pairs(min(n_views, 4), resolution, device="cpu")
    for _ in range(warmup_iters):
        for pair in dummy_pairs_warmup:
            batch = collate_with_cat([pair])
            _ = loss_of_one_batch(batch, model, None, device)
        torch.cuda.synchronize(device)

    # Benchmark: run full N-view sliding window
    times = []
    mems = []

    for trial in range(bench_iters):
        dummy_pairs = make_dummy_pairs(n_views, resolution, device="cpu")

        # Clear GPU cache and reset memory stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        t_start = time.perf_counter()

        # Process all N-1 pairs and accumulate predictions on GPU.
        # A real reconstruction pipeline keeps all pointmaps/confs in VRAM
        # for global alignment, so this reflects true memory cost.
        accumulated_gpu = []
        for pair in dummy_pairs:
            batch = collate_with_cat([pair])
            res = loss_of_one_batch(batch, model, None, device)
            # Extract and keep pred tensors on GPU (pts3d, conf, desc)
            for key in ('pred1', 'pred2'):
                if key in res:
                    for k, v in res[key].items():
                        if isinstance(v, torch.Tensor) and v.is_cuda:
                            accumulated_gpu.append(v)

        torch.cuda.synchronize(device)
        t_end = time.perf_counter()

        elapsed = t_end - t_start
        peak_mem = torch.cuda.max_memory_allocated(device) / 1e9  # GB

        # Free accumulated results
        del accumulated_gpu

        times.append(elapsed)
        mems.append(peak_mem)

    return {
        'time_s': round(np.median(times), 2),
        'mem_gb': round(max(mems), 2),  # peak across trials
        'n_pairs': n_pairs,
    }


def load_student(name, device):
    """Load a student model (no checkpoint needed for benchmark — random weights are fine)."""
    cfg = STUDENT_CONFIGS[name]
    student = cfg["builder"](device=str(device), **cfg["kwargs"])
    student = student.to(device).eval()
    return student


def main():
    parser = argparse.ArgumentParser(description="Inference benchmark (Distill3R-style)")
    parser.add_argument('--n_views', type=int, nargs='+', default=[2, 4 ,12, 32, 64, 96, 128],
                        help='Number of input views to benchmark')
    parser.add_argument('--resolution', type=int, default=512,
                        help='Input image resolution (square)')
    parser.add_argument('--warmup_iters', type=int, default=3,
                        help='Number of warmup iterations')
    parser.add_argument('--bench_iters', type=int, default=5,
                        help='Number of benchmark iterations (median is reported)')
    parser.add_argument('--output_csv', type=str, default='benchmark_results.csv')
    parser.add_argument('--output_latex', type=str, default='benchmark_results.tex')
    parser.add_argument('--models', type=str, nargs='*', default=None,
                        help='Models to benchmark (default: Teacher + all students)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"GPU: {gpu_name}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"N views: {args.n_views}")
    print(f"Warmup: {args.warmup_iters}, Bench: {args.bench_iters} iters")
    print()

    # Build model list
    models_to_bench = {}

    if args.models is None or 'Teacher' in args.models:
        print("Loading Teacher (MOONSt3R)...")
        teacher = load_teacher(device)
        models_to_bench['Teacher'] = teacher

    student_names = list(STUDENT_CONFIGS.keys())
    for sname in student_names:
        if args.models is not None and sname not in args.models:
            continue
        print(f"Building {sname} (random weights — no checkpoint needed for latency benchmark)...")
        models_to_bench[sname] = load_student(sname, device)

    # Run benchmarks
    rows = []
    for model_name, model in models_to_bench.items():
        n_params = count_params_M(model)
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_name} ({n_params:.1f}M params)")
        print(f"{'='*60}")

        for n_views in args.n_views:
            print(f"  N={n_views} views ({n_views-1} pairs)...", end=" ", flush=True)
            try:
                result = benchmark_model(
                    model, n_views, args.resolution,
                    args.warmup_iters, args.bench_iters, device,
                )
                print(f"Time: {result['time_s']:.2f}s, Mem: {result['mem_gb']:.2f} GB")
                rows.append({
                    'Model': model_name,
                    'Params(M)': round(n_params, 1),
                    'N': n_views,
                    'Time(s)': result['time_s'],
                    'Mem(GB)': result['mem_gb'],
                })
            except torch.cuda.OutOfMemoryError:
                print(f"OOM!")
                rows.append({
                    'Model': model_name,
                    'Params(M)': round(n_params, 1),
                    'N': n_views,
                    'Time(s)': float('nan'),
                    'Mem(GB)': float('nan'),
                })
                torch.cuda.empty_cache()

            # Free GPU memory between N-view runs
            torch.cuda.empty_cache()

        # Unload model between models to free memory
        del model
        torch.cuda.empty_cache()

    # Save results
    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")

    # Print formatted table
    print(f"\n{'='*70}")
    print("RESULTS TABLE")
    print(f"{'='*70}")
    print(f"GPU: {gpu_name}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print()

    # Pivot table: Model x N -> Time(s)
    pivot_time = df.pivot(index='Model', columns='N', values='Time(s)')
    pivot_mem = df.pivot(index='Model', columns='N', values='Mem(GB)')

    # Add params column
    params_map = df.drop_duplicates('Model').set_index('Model')['Params(M)']

    print("--- Inference Time (s) ---")
    print(f"{'Model':<25} {'Params':>8}", end="")
    for n in args.n_views:
        print(f" {'N='+str(n):>8}", end="")
    print()
    print("-" * (25 + 8 + len(args.n_views) * 9))
    for model_name in pivot_time.index:
        p = params_map.get(model_name, 0)
        print(f"{model_name:<25} {p:>7.1f}M", end="")
        for n in args.n_views:
            val = pivot_time.loc[model_name, n] if n in pivot_time.columns else float('nan')
            if pd.isna(val):
                print(f" {'OOM':>8}", end="")
            else:
                print(f" {val:>7.2f}s", end="")
        print()

    print()
    print("--- Peak GPU Memory (GB) ---")
    print(f"{'Model':<25} {'Params':>8}", end="")
    for n in args.n_views:
        print(f" {'N='+str(n):>8}", end="")
    print()
    print("-" * (25 + 8 + len(args.n_views) * 9))
    for model_name in pivot_mem.index:
        p = params_map.get(model_name, 0)
        print(f"{model_name:<25} {p:>7.1f}M", end="")
        for n in args.n_views:
            val = pivot_mem.loc[model_name, n] if n in pivot_mem.columns else float('nan')
            if pd.isna(val):
                print(f" {'OOM':>8}", end="")
            else:
                print(f" {val:>7.2f}", end="")
        print()

    # Generate LaTeX table
    with open(args.output_latex, 'w') as f:
        f.write("% Auto-generated inference benchmark table\n")
        f.write(f"% GPU: {gpu_name}, Resolution: {args.resolution}x{args.resolution}\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Inference latency (seconds) and peak GPU memory (GB) ")
        f.write(f"on {gpu_name}. ")
        f.write(f"Resolution: {args.resolution}$\\times${args.resolution}. ")
        f.write("Sliding-window protocol with N consecutive views.}\n")
        f.write("\\label{tab:inference_benchmark}\n")

        n_cols = len(args.n_views)
        col_spec = "l r " + " ".join(["r r"] * n_cols)
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\toprule\n")

        # Header row 1: N values spanning 2 columns each
        f.write("& ")
        for n in args.n_views:
            f.write(f"& \\multicolumn{{2}}{{c}}{{N={n}}} ")
        f.write("\\\\\n")

        # Header row 2: sub-columns
        cmidrule_start = 3
        for i in range(n_cols):
            f.write(f"\\cmidrule(lr){{{cmidrule_start}-{cmidrule_start+1}}} ")
            cmidrule_start += 2
        f.write("\n")

        f.write("Model & Params ")
        for _ in args.n_views:
            f.write("& Time & Mem ")
        f.write("\\\\\n")
        f.write("\\midrule\n")

        # Data rows
        for model_name in pivot_time.index:
            p = params_map.get(model_name, 0)
            f.write(f"{model_name} & {p:.1f}M ")
            for n in args.n_views:
                t_val = pivot_time.loc[model_name, n] if n in pivot_time.columns else float('nan')
                m_val = pivot_mem.loc[model_name, n] if n in pivot_mem.columns else float('nan')
                if pd.isna(t_val):
                    f.write("& OOM & OOM ")
                else:
                    f.write(f"& {t_val:.2f} & {m_val:.2f} ")
            f.write("\\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"\nLaTeX table saved to {args.output_latex}")

    # Save metadata
    meta = {
        'gpu': gpu_name,
        'resolution': args.resolution,
        'n_views': args.n_views,
        'warmup_iters': args.warmup_iters,
        'bench_iters': args.bench_iters,
    }
    with open(args.output_csv.replace('.csv', '_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()
