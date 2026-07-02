#!/usr/bin/env python3
"""
Compute GFLOPs for teacher and all student models.

Uses fvcore.nn.FlopCountAnalysis on a single forward pass (1 pair, 512x384).
Reports: Params (M), GFLOPs, compression ratio vs teacher.

Usage:
  python evaluation/moon/compute_flops.py
  python evaluation/moon/compute_flops.py --resolution 512 384
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, '.')
sys.path.insert(0, 'mast3r')
sys.path.insert(0, 'dust3r')

import mast3r.utils.path_to_dust3r  # noqa
from mast3r.model import AsymmetricMASt3R

from distillation_dual import build_mobilenet_student, build_vit_student
from evaluation.dtu.eval_compare_all_dtu import STUDENT_CONFIGS, build_vit_student as _bvs


# ── model loading ──────────────────────────────────────────────────────────────

def load_teacher(device):
    ckpt = "MOONSt3R.pth"
    model = AsymmetricMASt3R.from_pretrained(ckpt).to(device).eval()
    return model


def load_student_model(key, device):
    cfg = STUDENT_CONFIGS[key]
    model = cfg["builder"](device=str(device), **cfg["kwargs"])
    return model.to(device).eval()


# ── dummy input ────────────────────────────────────────────────────────────────

def make_dummy_batch(H, W, device):
    """Single image pair, batch=1."""
    view1 = {
        'img':        torch.randn(1, 3, H, W, device=device),
        'true_shape': torch.tensor([[H, W]], device=device),
        'idx':        0,
        'instance':   '0',
    }
    view2 = {
        'img':        torch.randn(1, 3, H, W, device=device),
        'true_shape': torch.tensor([[H, W]], device=device),
        'idx':        1,
        'instance':   '1',
    }
    return view1, view2


# ── flop counting ──────────────────────────────────────────────────────────────

def count_flops(model, H, W, device):
    """
    Count GFLOPs via ptflops (get_model_complexity_info).
    Uses a custom input constructor to feed dict-based view pairs.
    """
    from ptflops import get_model_complexity_info

    def input_constructor(input_res):
        # input_res is ignored; we always build a fixed H x W pair
        v1, v2 = make_dummy_batch(H, W, device)
        # ptflops calls model(*args) where args = input_constructor output
        return {"view1": v1, "view2": v2}

    with torch.no_grad():
        try:
            macs, _ = get_model_complexity_info(
                model,
                (3, H, W),           # shape hint (unused, overridden by constructor)
                input_constructor=input_constructor,
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False,
            )
            return macs / 1e9  # GFLOPs (MACs ≈ FLOPs/2, but reported as GFLOPs here)
        except Exception as e:
            print(f"  [ptflops failed: {e}]")
            return float('nan')


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int, nargs=2, default=[384, 512],
                        help='H W (default: 384 512, i.e. 512x384 after DUSt3R crop)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    H, W = args.resolution
    device = torch.device(args.device)
    print(f"Resolution: {W}x{H}  |  Device: {device}\n")

    results = []

    # Teacher
    print("Loading Teacher (MOONSt3R)...")
    teacher = load_teacher(device)
    p_t = count_params(teacher)
    f_t = count_flops(teacher, H, W, device)
    results.append(('Teacher', p_t, f_t, 1.0, 1.0))
    print(f"  Params: {p_t:.1f}M   GFLOPs: {f_t:.1f}\n")
    del teacher; torch.cuda.empty_cache()

    # Students
    for key in STUDENT_CONFIGS:
        print(f"Loading {key}...")
        try:
            model = load_student_model(key, device)
            p = count_params(model)
            f = count_flops(model, H, W, device)
            param_ratio = p_t / p if p > 0 else float('nan')
            flop_ratio  = f_t / f if f > 0 else float('nan')
            results.append((key, p, f, param_ratio, flop_ratio))
            print(f"  Params: {p:.1f}M ({param_ratio:.1f}x↓)   GFLOPs: {f:.1f} ({flop_ratio:.1f}x↓)\n")
        except Exception as e:
            print(f"  ERROR: {e}\n")
            results.append((key, float('nan'), float('nan'), float('nan'), float('nan')))
        finally:
            torch.cuda.empty_cache()

    # Print table
    print("\n" + "="*75)
    print(f"{'Model':<30} {'Params(M)':>10} {'GFLOPs':>10} {'Param↓':>8} {'FLOP↓':>8}")
    print("="*75)
    for name, p, f, pr, fr in results:
        tag = " ← teacher" if name == "Teacher" else ""
        p_s  = f"{p:.1f}"  if not (isinstance(p,  float) and p  != p) else "n/a"
        f_s  = f"{f:.1f}"  if not (isinstance(f,  float) and f  != f) else "n/a"
        pr_s = f"{pr:.1f}x" if not (isinstance(pr, float) and pr != pr) else "n/a"
        fr_s = f"{fr:.1f}x" if not (isinstance(fr, float) and fr != fr) else "n/a"
        print(f"{name:<30} {p_s:>10} {f_s:>10} {pr_s:>8} {fr_s:>8}{tag}")
    print("="*75)

    # Save as JSON
    out = {
        'resolution': f'{W}x{H}',
        'device': str(device),
        'models': [
            {'name': n, 'params_M': round(p, 2), 'gflops': round(f, 2),
             'param_compression': round(pr, 2), 'flop_compression': round(fr, 2)}
            for n, p, f, pr, fr in results
        ]
    }
    import json
    outpath = 'results/flops_table.json'
    os.makedirs('results', exist_ok=True)
    with open(outpath, 'w') as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == '__main__':
    main()
