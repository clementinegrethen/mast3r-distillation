#!/usr/bin/env python3
"""
Compare pair_report.txt metrics across all students, averaged per test folder.

Usage:
  python compare_pair_reports.py [--root eval_benchmark/with_init]

Output: formatted tables (clean, landing, pitch, ALL) for key paper metrics.
"""

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


# Metrics to extract from pair_report.txt
# Format: (display_name, regex_pattern_avg, regex_pattern_absrel, direction)
# direction: 'lower' = lower is better, 'higher' = higher is better
# For metrics with AbsRel column: avg is col3, absrel is col4
# Pose metrics have no AbsRel (single value after colon)
METRICS = [
    # 3D metrics — GLOBAL SCENE (v0+v1 combined): 2 columns (abs, AbsRel)
    ("Chamfer (m)",      r"GLOBAL SCENE.*?Chamfer \(m\)\s+([\d.]+)",
                         r"GLOBAL SCENE.*?Chamfer \(m\)\s+[\d.]+\s+([\d.]+)",            "lower"),
    ("RMSE (m)",         r"RMSE \(m\)\s+[\d.]+\s+[\d.]+\s+([\d.]+)",
                         r"RMSE \(m\)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)",               "lower"),
    ("Accuracy (m)",     r"GLOBAL SCENE.*?Accuracy \(m\)\s+([\d.]+)",
                         r"GLOBAL SCENE.*?Accuracy \(m\)\s+[\d.]+\s+([\d.]+)",              "lower"),
    ("Completeness (m)", r"GLOBAL SCENE.*?Completeness \(m\)\s+([\d.]+)",
                         r"GLOBAL SCENE.*?Completeness \(m\)\s+[\d.]+\s+([\d.]+)",          "lower"),
    # Depth metrics
    ("Depth MAE (m)",    r"Depth MAE \(m\)\s+[\d.]+\s+[\d.]+\s+([\d.]+)",
                         r"Depth MAE \(m\)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)",          "lower"),
    ("Depth RMSE (m)",   r"Depth RMSE \(m\)\s+[\d.]+\s+[\d.]+\s+([\d.]+)",
                         r"Depth RMSE \(m\)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)",         "lower"),
    ("Depth Pearson",    r"Depth Pearson\s+[\d.]+\s+[\d.]+\s+([\d.]+)",
                         None,                                                                "higher"),
    ("Depth SSIM",       r"Depth SSIM\s+[\d.]+\s+[\d.]+\s+([\d.]+)",
                         None,                                                                "higher"),
    # Pose (single values, no AbsRel)
    ("RRA (deg)",        r"RRA emat \(°\)\s*:\s*([\d.]+)",        None, "lower"),
    ("RTA (deg)",        r"RTA emat \(°\)\s*:\s*([\d.]+)",        None, "lower"),
    ("Pose err (deg)",   r"Pose error emat \(°\)\s*:\s*([\d.]+)", None, "lower"),
    ("VCRE med (px)",    r"VCRE median \(px\)\s*:\s*([\d.]+)",    None, "lower"),
    ("VCRE (% diag)",   r"VCRE \(% diag\)\s*:\s*([\d.]+)",       None, "lower"),
    # Depth profile (central profile along v0/v1)
    ("Profile corr",     r"Profile corr\s+[\d.]+\s+[\d.]+\s+([\d.]+)",     None, "higher"),
    ("Profile MAE (m)",  r"Profile MAE \(m\)\s+[\d.]+\s+[\d.]+\s+([\d.]+)", None, "lower"),
    # Slope (no AbsRel column in report)
    ("Slope MAE (deg)",  r"Slope MAE \(°\)\s+[\d.]+\s+[\d.]+\s+([\d.]+)",  None, "lower"),
    ("Slope corr",       r"Slope corr\s+[\d.]+\s+[\d.]+\s+([\d.]+)",       None, "higher"),
]


def parse_report(filepath):
    """Parse a pair_report.txt and return a dict of metric_name -> value.
    Also extracts AbsRel values where available (key: metric_name + ' AbsRel')."""
    with open(filepath, 'r') as f:
        text = f.read()

    values = {}
    for name, pattern_avg, pattern_absrel, _ in METRICS:
        m = re.search(pattern_avg, text, re.DOTALL)
        if m:
            try:
                values[name] = float(m.group(1))
            except (ValueError, IndexError):
                pass
        if pattern_absrel:
            m2 = re.search(pattern_absrel, text, re.DOTALL)
            if m2:
                try:
                    values[name + ' AbsRel'] = float(m2.group(1))
                except (ValueError, IndexError):
                    pass
    return values


def collect_all(root):
    """
    Walk root dir and collect metrics.
    Returns: dict[student][folder] -> list of metric dicts (one per pair)
    """
    data = defaultdict(lambda: defaultdict(list))

    for student_dir in sorted(Path(root).iterdir()):
        if not student_dir.is_dir():
            continue
        student = student_dir.name

        for folder_dir in sorted(student_dir.iterdir()):
            if not folder_dir.is_dir():
                continue
            folder = folder_dir.name

            for pair_dir in sorted(folder_dir.iterdir()):
                report = pair_dir / "pair_report.txt"
                if report.exists():
                    values = parse_report(report)
                    if values:
                        data[student][folder].append(values)

    return data


def compute_means(records):
    """Given a list of metric dicts, compute mean per metric."""
    if not records:
        return {}
    all_keys = set()
    for r in records:
        all_keys.update(r.keys())
    means = {}
    for k in all_keys:
        vals = [r[k] for r in records if k in r]
        if vals:
            means[k] = np.mean(vals)
    return means


def print_table(title, students, means_per_student, metrics_to_show=None):
    """Print a formatted comparison table."""
    if metrics_to_show is None:
        metrics_to_show = [m[0] for m in METRICS]

    # Find best values per metric (for bold marking) — use AbsRel when available
    best = {}
    for mname, _, _, direction in METRICS:
        if mname not in metrics_to_show:
            continue
        vals = []
        for s in students:
            v = means_per_student.get(s, {}).get(mname)
            if v is not None:
                vals.append((v, s))
        if vals:
            if direction == "lower":
                best[mname] = min(vals, key=lambda x: x[0])[1]
            else:
                best[mname] = max(vals, key=lambda x: x[0])[1]

    # Header
    col_w = 22
    metric_w = 20
    total_w = metric_w + col_w * len(students) + 4
    print()
    print(f"{'=' * total_w}")
    print(f"  {title}")
    print(f"{'=' * total_w}")

    # Column headers
    header = f"{'Metric':<{metric_w}}"
    for s in students:
        short = s.replace("_", " ").replace("ViT-Small", "ViT-S").replace("ViT-Tiny", "ViT-T")
        header += f"{short:>{col_w}}"
    print(header)
    print("-" * len(header))

    # Rows: show "abs_value (AbsRel)" when AbsRel available, else just value
    for mname, _, _, direction in METRICS:
        if mname not in metrics_to_show:
            continue
        row = f"{mname:<{metric_w}}"
        for s in students:
            v = means_per_student.get(s, {}).get(mname)
            ar = means_per_student.get(s, {}).get(mname + ' AbsRel')
            if v is None:
                row += f"{'n/a':>{col_w}}"
            else:
                is_best = (best.get(mname) == s)
                fmt_v = f"{v:.4f}" if abs(v) < 1 else f"{v:.2f}"
                if is_best:
                    fmt_v = f"*{fmt_v}*"
                if ar is not None:
                    fmt_ar = f"{ar:.4f}"
                    combined = f"{fmt_v} ({fmt_ar})"
                else:
                    combined = fmt_v
                row += f"{combined:>{col_w}}"
        print(row)

    print("-" * len(header))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='eval_benchmark/with_init',
                        help='Root directory containing student subdirs')
    args = parser.parse_args()

    data = collect_all(args.root)
    if not data:
        print(f"No pair_report.txt found in {args.root}")
        return

    students = sorted(data.keys())
    folders = sorted(set(f for s in data for f in data[s]))

    # Map folder names to short labels
    folder_labels = {
        'test_image_clean': 'Nadir',
        'test_image_clean_landing': 'Landing',
        'test_image_clean_pitch': 'Pitch',
    }

    # Per-folder tables
    for folder in folders:
        label = folder_labels.get(folder, folder)
        means = {}
        for s in students:
            records = data[s].get(folder, [])
            means[s] = compute_means(records)
            means[s]['_n_pairs'] = len(records)

        # Count pairs
        pair_info = ", ".join(f"{s}: {means[s]['_n_pairs']}p" for s in students)
        print_table(f"{label}  ({pair_info})", students, means)

    # ALL folders combined
    means_all = {}
    for s in students:
        all_records = []
        for folder in folders:
            all_records.extend(data[s].get(folder, []))
        means_all[s] = compute_means(all_records)
        means_all[s]['_n_pairs'] = len(all_records)

    pair_info = ", ".join(f"{s}: {means_all[s]['_n_pairs']}p" for s in students)
    print_table(f"ALL (combined)  ({pair_info})", students, means_all)

    # ── VCRE AUC (MASt3R-style, à partir des VCRE médians par paire) ──────────
    def vcre_auc(vcre_list, t):
        """AUC de la courbe recall VCRE dans [0, t] pixels, normalisée (0-100)."""
        arr = np.array(vcre_list, dtype=np.float64)
        n_bins = max(int(t * 2), 100)
        x = np.linspace(0, t, n_bins + 1)
        recalls = np.array([(arr <= xi).mean() for xi in x])
        return float(np.trapz(recalls, x) / t * 100)

    auc_thresholds = [5, 10, 50, 100, 200]
    col_w = 22
    metric_w = 20
    total_w = metric_w + col_w * len(students) + 4
    print()
    print("=" * total_w)
    print("  VCRE AUC & Precision (MASt3R-style) — ALL folders")
    print("=" * total_w)
    header = f"{'Metric':<{metric_w}}"
    for s in students:
        short = s.replace("_", " ").replace("ViT-Small", "ViT-S").replace("ViT-Tiny", "ViT-T")
        header += f"{short:>{col_w}}"
    print(header)
    print("-" * len(header))

    # Collect per-student VCRE lists (ALL folders)
    vcre_per_student = {}
    for s in students:
        vals = []
        for folder in folders:
            for rec in data[s].get(folder, []):
                v = rec.get('VCRE med (px)')
                if v is not None:
                    vals.append(v)
        vcre_per_student[s] = vals

    # Collect per-student VCRE (% diag) lists (ALL folders)
    vcre_pct_per_student = {}
    for s in students:
        vals = []
        for folder in folders:
            for rec in data[s].get(folder, []):
                v = rec.get('VCRE (% diag)')
                if v is not None:
                    vals.append(v)
        vcre_pct_per_student[s] = vals

    # Prec@5% diag and Prec@10% diag (MapFree-style)
    for thr_pct in [5, 10]:
        row = f"{f'Prec@{thr_pct}%diag (%)':<{metric_w}}"
        best_val, best_s = -1, None
        pcts = {}
        for s in students:
            vals = vcre_pct_per_student[s]
            if vals:
                pct = 100.0 * sum(1 for v in vals if v < thr_pct) / len(vals)
                pcts[s] = pct
                if pct > best_val:
                    best_val, best_s = pct, s
        for s in students:
            if s not in pcts:
                row += f"{'n/a':>{col_w}}"
            else:
                fmt = f"{pcts[s]:.1f}%"
                if s == best_s:
                    fmt = f"*{fmt}*"
                row += f"{fmt:>{col_w}}"
        print(row)

    # Precision@90px (= % < 90px, MASt3R paper threshold)
    row = f"{'Prec@90px (%)':<{metric_w}}"
    best_val, best_s = -1, None
    pcts = {}
    for s in students:
        vals = vcre_per_student[s]
        if vals:
            pct = 100.0 * sum(1 for v in vals if v < 90) / len(vals)
            pcts[s] = pct
            if pct > best_val:
                best_val, best_s = pct, s
    for s in students:
        if s not in pcts:
            row += f"{'n/a':>{col_w}}"
        else:
            fmt = f"{pcts[s]:.1f}%"
            if s == best_s:
                fmt = f"*{fmt}*"
            row += f"{fmt:>{col_w}}"
    print(row)

    # Reproj median (px)
    row = f"{'Reproj med (px)':<{metric_w}}"
    best_val, best_s = float('inf'), None
    meds = {}
    for s in students:
        vals = vcre_per_student[s]
        if vals:
            med = float(np.median(vals))
            meds[s] = med
            if med < best_val:
                best_val, best_s = med, s
    for s in students:
        if s not in meds:
            row += f"{'n/a':>{col_w}}"
        else:
            fmt = f"{meds[s]:.2f}"
            if s == best_s:
                fmt = f"*{fmt}*"
            row += f"{fmt:>{col_w}}"
    print(row)

    # AUC at each threshold
    for t in auc_thresholds:
        row = f"{'VCRE AUC@' + str(t) + 'px':<{metric_w}}"
        best_val, best_s = -1, None
        aucs = {}
        for s in students:
            vals = vcre_per_student[s]
            if vals:
                a = vcre_auc(vals, t)
                aucs[s] = a
                if a > best_val:
                    best_val, best_s = a, s
        for s in students:
            if s not in aucs:
                row += f"{'n/a':>{col_w}}"
            else:
                fmt = f"{aucs[s]:.1f}%"
                if s == best_s:
                    fmt = f"*{fmt}*"
                row += f"{fmt:>{col_w}}"
        print(row)

    print("-" * len(header))
    print()

    # VCRE threshold table
    vcre_thresholds = [5, 10, 20, 50, 90, 100]
    col_w = 22
    metric_w = 20
    total_w = metric_w + col_w * len(students) + 4
    print()
    print("=" * total_w)
    print("  VCRE recall @ thresholds (% pairs below N px) — ALL folders")
    print("=" * total_w)
    header = f"{'Threshold':<{metric_w}}"
    for s in students:
        short = s.replace("_", " ").replace("ViT-Small", "ViT-S").replace("ViT-Tiny", "ViT-T")
        header += f"{short:>{col_w}}"
    print(header)
    print("-" * len(header))
    for thr in vcre_thresholds:
        row = f"VCRE < {thr}px{'':<{metric_w - len(f'VCRE < {thr}px')}}"
        best_val = -1
        best_s = None
        pcts = {}
        for s in students:
            all_vcre = []
            for folder in folders:
                for rec in data[s].get(folder, []):
                    v = rec.get('VCRE med (px)')
                    if v is not None:
                        all_vcre.append(v)
            if all_vcre:
                pct = 100.0 * sum(1 for v in all_vcre if v < thr) / len(all_vcre)
                pcts[s] = pct
                if pct > best_val:
                    best_val = pct
                    best_s = s
        for s in students:
            if s not in pcts:
                row += f"{'n/a':>{col_w}}"
            else:
                fmt = f"{pcts[s]:.1f}%"
                if s == best_s:
                    fmt = f"*{fmt}*"
                row += f"{fmt:>{col_w}}"
        print(row)
    print("-" * len(header))
    print()

    # Also save CSV
    csv_path = os.path.join(args.root, "comparison_table.csv")
    with open(csv_path, 'w') as f:
        # Header: per student, value + AbsRel columns
        student_cols = []
        for s in students:
            student_cols.append(s)
            student_cols.append(s + " AbsRel")
        f.write("Folder,Metric," + ",".join(student_cols) + "\n")
        for folder in folders + ["ALL"]:
            label = folder_labels.get(folder, folder) if folder != "ALL" else "ALL"
            if folder == "ALL":
                means = means_all
            else:
                means = {}
                for s in students:
                    means[s] = compute_means(data[s].get(folder, []))
            for mname, _, _, _ in METRICS:
                vals = []
                for s in students:
                    v = means.get(s, {}).get(mname)
                    ar = means.get(s, {}).get(mname + ' AbsRel')
                    vals.append(f"{v:.4f}" if v is not None else "")
                    vals.append(f"{ar:.4f}" if ar is not None else "")
                f.write(f"{label},{mname}," + ",".join(vals) + "\n")

    print(f"CSV saved to {csv_path}")


if __name__ == '__main__':
    main()
