"""
moon_eval/reporter.py — Summary printing and results persistence.

Functions:
  print_summary_table    — Classic 3D metrics per model
  print_terrain_summary  — Relief / terrain feature metrics
  print_camera_summary   — RRA / RTA / AUC camera pose metrics
  print_hda_summary      — HDA slope classification table
  save_results           — CSV + JSON persistence
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _med(df: pd.DataFrame, col: str, fmt: str = ".4f") -> str:
    """Median of a column, formatted.  Returns 'n/a' if column missing/empty."""
    if col not in df.columns:
        return "n/a"
    vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) == 0:
        return "n/a"
    return format(float(vals.median()), fmt)


def _pct(df: pd.DataFrame, col: str) -> str:
    """Median of a percentage column, formatted as '%.1f%%'."""
    if col not in df.columns:
        return "n/a"
    vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) == 0:
        return "n/a"
    return f"{float(vals.median()):.1f}%"


# ─────────────────────────────────────────────────────────────────────────────
# Summary tables
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Print classic 3D reconstruction metrics per model.

    Columns: Model, N, chamfer, accuracy, completeness, rmse, depth_pearson,
             slope_corr, slope_mae, profile_corr, slope_agree_10deg
    """
    print("\n" + "=" * 110)
    print("  CLASSIC 3D RECONSTRUCTION METRICS  (median across pairs)")
    print("=" * 110)

    rows = []
    for mname in sorted(df["Model"].unique()):
        ms = df[df["Model"] == mname]
        row = {
            "Model": mname,
            "N": len(ms),
            # Absolute distances (m)
            "chamfer_m": _med(ms, "avg_chamfer", ".2f"),
            "rmse_m": _med(ms, "avg_rmse", ".2f"),
            # AbsRel = error / gt_median_depth  (unitless, ∈ [0,1])
            "chamfer_absrel": _med(ms, "avg_chamfer_absrel", ".4f"),
            "rmse_absrel": _med(ms, "avg_rmse_absrel", ".4f"),
            "scene_ch_absrel": _med(ms, "scene_chamfer_absrel", ".4f"),
            # GSD-relative = error / GSD  (pixel-equivalent, altitude-independent)
            "chamfer_gsd": _med(ms, "avg_chamfer_gsd", ".2f"),
            "rmse_gsd": _med(ms, "avg_rmse_gsd", ".2f"),
            # Correlation / quality
            "depth_pearson": _med(ms, "avg_depth_pearson", ".4f"),
            "depth_ssim": _med(ms, "avg_depth_ssim", ".4f"),
            "slope_corr": _med(ms, "avg_slope_corr", ".4f"),
            "slope_mae°": _med(ms, "avg_slope_mae", ".3f"),
            "profile_corr": _med(ms, "avg_profile_corr", ".4f"),
        }
        rows.append(row)

    df_sum = pd.DataFrame(rows)
    print(df_sum.to_string(index=False))
    return df_sum


def print_terrain_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Print terrain and relief quality metrics per model."""
    print("\n" + "=" * 110)
    print("  TERRAIN RELIEF METRICS  (median across pairs)")
    print("=" * 110)

    rows = []
    for mname in sorted(df["Model"].unique()):
        ms = df[df["Model"] == mname]
        row = {
            "Model": mname,
            "N": len(ms),
            "slope_ssim": _med(ms, "avg_slope_ssim", ".4f"),
            "relief_corr": _med(ms, "avg_relief_corr", ".4f"),
            "roughness_corr": _med(ms, "avg_roughness_corr", ".4f"),
            "curvature_corr": _med(ms, "avg_curvature_corr", ".4f"),
            "crater_IoU": _med(ms, "avg_crater_combined_iou", ".4f"),
            "crater_rim_IoU": _med(ms, "avg_crater_rim_iou", ".4f"),
            "crater_int_IoU": _med(ms, "avg_crater_interior_iou", ".4f"),
            "peak_recall%": _pct(ms, "avg_peak_location_recall"),
            "valley_recall%": _pct(ms, "avg_valley_location_recall"),
            "aspect_hist_corr": _med(ms, "avg_aspect_hist_corr", ".4f"),
        }
        rows.append(row)

    df_sum = pd.DataFrame(rows)
    print(df_sum.to_string(index=False))
    return df_sum


def print_camera_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Print camera pose metrics (RRA / RTA / AUC) per model.

    For MASt3R-based models: uses rra_emat / rta_emat (Essential matrix from
    dense matches) and re-computes AUC per model from all pose_error_emat values.

    For COLMAP-SIFT baseline: uses rra_colmap / rta_colmap (SIFT + E-matrix).
    """
    from .metrics.camera import compute_auc

    print("\n" + "=" * 110)
    print("  CAMERA POSE METRICS  (median RRA/RTA, global AUC across all pairs)")
    print("=" * 110)

    rows = []
    for mname in sorted(df["Model"].unique()):
        ms = df[df["Model"] == mname]

        # Detect baseline type and select the right pose columns
        if mname == "COLMAP-SIFT":
            pe_col  = "pose_error_colmap"
            rra_col = "rra_colmap"
            rta_col = "rta_colmap"
            inlier_col = "n_inliers_colmap"
        elif mname == "COLMAP-MVS":
            pe_col  = "pose_error_colmap_mvs"
            rra_col = "rra_colmap_mvs"
            rta_col = "rta_colmap_mvs"
            inlier_col = "n_inliers_colmap_mvs"
        else:
            pe_col  = "pose_error_emat"
            rra_col = "rra_emat"
            rta_col = "rta_emat"
            inlier_col = None   # deep models show n_inliers_emat

        is_classical = mname in ("COLMAP-SIFT", "COLMAP-MVS")

        pose_errors = (
            ms[pe_col].replace([np.nan], np.inf).values.tolist()
            if pe_col in ms.columns else []
        )
        auc = compute_auc(pose_errors) if pose_errors else {}

        # Valid pairs only (finite RRA)
        if rra_col in ms.columns:
            ms_valid = ms[ms[rra_col].replace([np.inf, -np.inf], np.nan).notna()]
        else:
            ms_valid = ms.iloc[0:0]   # empty

        row: Dict = {
            "Model": mname,
            "N": len(ms),
            "N_valid_pose": len(ms_valid),
            "RRA_med(°)": _med(ms_valid, rra_col, ".2f"),
            "RTA_med(°)": _med(ms_valid, rta_col, ".2f"),
            "AUC@5": f"{auc.get('AUC@5', float('nan')):.1f}",
            "AUC@10": f"{auc.get('AUC@10', float('nan')):.1f}",
            "AUC@20": f"{auc.get('AUC@20', float('nan')):.1f}",
        }
        if not is_classical:
            row["RRA_algn_med"] = _med(ms, "rra_aligner", ".2f")
            row["RTA_algn_med"] = _med(ms, "rta_aligner", ".2f")
            row["VCRE_med(px)"] = _med(ms, "vcre_median_px", ".1f")
            row["VCRE(%diag)"] = _med(ms, "vcre_pct", ".3f")
            # VCRE Precision & AUC (MASt3R-style)
            vcre_vals = (
                ms["vcre_median_px"].replace([np.inf, -np.inf], np.nan)
                .dropna().values.tolist()
                if "vcre_median_px" in ms.columns else []
            )
            if vcre_vals:
                from .metrics.camera import compute_vcre_auc
                vcre_auc = compute_vcre_auc(vcre_vals)
                row["VCRE_Prec@90"] = f"{vcre_auc.get('VCRE_Prec@100', 0):.1f}"
                # Use 90px like MASt3R paper for precision
                prec90 = float((np.array(vcre_vals) < 90).mean() * 100)
                row["VCRE_Prec@90"] = f"{prec90:.1f}"
                row["VCRE_AUC@100"] = f"{vcre_auc.get('VCRE_AUC@100', 0):.1f}"
            row["n_inliers_emat"] = _med(ms, "n_inliers_emat", ".0f")
        else:
            row["n_inliers"] = _med(ms, inlier_col, ".0f")
        rows.append(row)

    df_sum = pd.DataFrame(rows)
    print(df_sum.to_string(index=False))
    return df_sum


def print_hda_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Print HDA slope detection table per model per threshold."""
    print("\n" + "=" * 110)
    print("  HDA SLOPE DETECTION  (median across pairs)")
    print("=" * 110)

    rows = []
    for mname in sorted(df["Model"].unique()):
        ms = df[df["Model"] == mname]
        row: Dict = {"Model": mname}
        for t in [5, 10, 15, 20]:
            row[f"agree_{t}°"] = _pct(ms, f"avg_slope_agree_{t}deg")
            row[f"miss_{t}°"] = _pct(ms, f"avg_slope_miss_{t}deg")
            row[f"FA_{t}°"] = _pct(ms, f"avg_slope_false_alarm_{t}deg")
        rows.append(row)

    df_hda = pd.DataFrame(rows)
    print(df_hda.to_string(index=False))
    return df_hda


def print_efficiency_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Print inference time and GPU memory usage per model."""
    print("\n" + "=" * 110)
    print("  INFERENCE EFFICIENCY  (median across pairs)")
    print("=" * 110)

    rows = []
    for mname in sorted(df["Model"].unique()):
        ms = df[df["Model"] == mname]
        row = {
            "Model": mname,
            "N": len(ms),
            "infer_time_s": _med(ms, "inference_time_s", ".2f"),
            "peak_gpu_MB": _med(ms, "peak_gpu_mem_mb", ".0f"),
        }
        # Compute throughput (pairs/min)
        if "inference_time_s" in ms.columns:
            vals = ms["inference_time_s"].dropna()
            if len(vals) > 0:
                med_t = float(vals.median())
                row["pairs_per_min"] = f"{60.0 / med_t:.1f}" if med_t > 0 else "n/a"
            else:
                row["pairs_per_min"] = "n/a"
        else:
            row["pairs_per_min"] = "n/a"
        rows.append(row)

    df_sum = pd.DataFrame(rows)
    print(df_sum.to_string(index=False))
    return df_sum


def print_all(df: pd.DataFrame) -> None:
    """Print all summary tables."""
    print_summary_table(df)
    print_terrain_summary(df)
    print_camera_summary(df)
    print_hda_summary(df)
    print_efficiency_summary(df)


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_pair_report_txt(row: dict, out_dir) -> Path:
    """Write a human-readable .txt report for one evaluated pair.

    Saved to:  {out_dir}/pair_report.txt

    Sections:
      - Identity (pair, folder, model)
      - Alignment
      - Classic 3D metrics  (per-view + average)
      - Depth metrics        (per-view + average)
      - Camera pose
      - Slope / HDA
      - Terrain / relief
    """
    from datetime import datetime

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pair_report.txt"

    def _f(key, fmt=".4f", unit=""):
        v = row.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "n/a"
        try:
            return f"{float(v):{fmt}}{unit}"
        except (TypeError, ValueError):
            return str(v)

    def _pf(key, unit="%"):
        """Percentage field."""
        return _f(key, fmt=".1f", unit=unit)

    sep = "─" * 62

    lines = [
        "=" * 62,
        "  PAIR EVALUATION REPORT",
        "=" * 62,
        f"  Date    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Model   : {row.get('Model', 'n/a')}",
        f"  Folder  : {row.get('Folder', 'n/a')}",
        f"  Pair    : {row.get('pair', 'n/a')}",
        "",
        sep,
        "  ALIGNMENT",
        sep,
        f"  Scale              : {_f('scene_scale', '.6f')}",
        f"  Scale error        : {_f('scene_scale_err_pct', '.2f')}%",
        f"  Method             : {row.get('alignment_method', 'n/a')}",
        f"  Valid points       : {row.get('scene_n_valid', 'n/a')}",
        f"  GT focal (px)      : {_f('gt_focal', '.1f')}",
        f"  Optim focal v0/v1  : {_f('optimized_focal_v0', '.1f')} / {_f('optimized_focal_v1', '.1f')}",
        f"  GT median depth    : {_f('gt_median_depth', '.1f')} m  (AbsRel denominator)",
        f"  GT terrain span    : {_f('gt_terrain_span', '.1f')} m  (depth p5→p95)",
        f"  GT GSD (m/px)      : {_f('gt_gsd', '.4f')}  (GSD-rel denominator)",
        f"  Inference time (s) : {_f('inference_time_s', '.3f')}",
        f"  Peak GPU mem (MB)  : {_f('peak_gpu_mem_mb', '.0f')}",
        "",
    ]

    # ── Per-view + avg: classic 3D
    hdr = f"  {'Metric':<26} {'v0':>10} {'v1':>10} {'avg':>10} {'AbsRel':>10}"
    lines += [sep, "  CLASSIC 3D METRICS — PER VIEW", sep, hdr]
    classic_keys = [
        ("RMSE (m)",            "rmse",           "rmse_absrel"),
        ("MAE 3D (m)",          "mae_3d",         "mae_3d_absrel"),
        ("Accuracy (m)",        "accuracy",       "accuracy_absrel"),
        ("Completeness (m)",    "completeness",   "completeness_absrel"),
        ("Chamfer (m)",         "chamfer",        "chamfer_absrel"),
        ("Acc median (m)",      "acc_median",     "acc_median_absrel"),
        ("Compl median (m)",    "compl_median",   "compl_median_absrel"),
        ("Pearson Z",           "pearson_z",      None),
    ]
    for label, key, rkey in classic_keys:
        v0 = _f(f"v0_{key}")
        v1 = _f(f"v1_{key}")
        av = _f(f"avg_{key}")
        rv = _f(f"avg_{rkey}", ".4f") if rkey else "      n/a"
        lines.append(f"  {label:<26} {v0:>10} {v1:>10} {av:>10} {rv:>10}")

    # ── Global / combined-cloud metrics
    hdr1 = f"  {'Metric':<30} {'abs (m)':>10} {'AbsRel':>10}"
    lines += ["", sep, "  CLASSIC 3D METRICS — GLOBAL SCENE (v0+v1 combined)", sep, hdr1]
    scene_keys = [
        ("Accuracy (m)",         "scene_accuracy"),
        ("Completeness (m)",     "scene_completeness"),
        ("Chamfer (m)",          "scene_chamfer"),
        ("Acc median (m)",       "scene_acc_median"),
        ("Compl median (m)",     "scene_compl_median"),
        ("Hausdorff p95 (m)",    "scene_hausdorff_p95"),
        ("Hausdorff max (m)",    "scene_hausdorff_max"),
    ]
    for label, key in scene_keys:
        abs_v = _f(key)
        rel_v = _f(f"{key}_absrel", ".4f")
        lines.append(f"  {label:<30} {abs_v:>10} {rel_v:>10}")
    for t in [0.5, 1.0, 2.0]:
        lp = f"Acc <{t}m (%)"
        lc = f"Compl <{t}m (%)"
        lines.append(f"  {lp:<30} {_pf('scene_acc_pct_under_'+str(t)):>10}")
        lines.append(f"  {lc:<30} {_pf('scene_compl_pct_under_'+str(t)):>10}")

    # ── Per-view + avg: depth map
    hdr3 = f"  {'Metric':<26} {'v0':>10} {'v1':>10} {'avg':>10}"
    lines += ["", sep, "  DEPTH MAP METRICS", sep, hdr]
    depth_keys_absrel = [
        ("Depth MAE (m)",        "depth_mae",      "depth_mae_absrel"),
        ("Depth RMSE (m)",       "depth_rmse",     "depth_rmse_absrel"),
    ]
    depth_keys_plain = [
        ("Depth Pearson",        "depth_pearson"),
        ("Depth SSIM",           "depth_ssim"),
        ("SILog",                "silog"),
        ("delta1 (%)",           "delta1"),
        ("delta2 (%)",           "delta2"),
        ("delta3 (%)",           "delta3"),
        ("Profile corr",         "profile_corr"),
        ("Profile MAE (m)",      "profile_mae"),
    ]
    for label, key, rkey in depth_keys_absrel:
        v0 = _f(f"v0_{key}")
        v1 = _f(f"v1_{key}")
        av = _f(f"avg_{key}")
        rv = _f(f"avg_{rkey}", ".4f")
        lines.append(f"  {label:<26} {v0:>10} {v1:>10} {av:>10} {rv:>10}")
    for label, key in depth_keys_plain:
        v0 = _f(f"v0_{key}")
        v1 = _f(f"v1_{key}")
        av = _f(f"avg_{key}")
        lines.append(f"  {label:<26} {v0:>10} {v1:>10} {av:>10}")

    # ── Camera pose
    lines += ["", sep, "  CAMERA POSE", sep]
    lines += [
        f"  RRA emat (°)       : {_f('rra_emat', '.2f')}",
        f"  RTA emat (°)       : {_f('rta_emat', '.2f')}",
        f"  Pose error emat (°): {_f('pose_error_emat', '.2f')}",
        f"  N inliers emat     : {row.get('n_inliers_emat', 'n/a')}",
        f"  RRA aligner (°)    : {_f('rra_aligner', '.2f')}",
        f"  RTA aligner (°)    : {_f('rta_aligner', '.2f')}",
        f"  VCRE median (px)   : {_f('vcre_median_px', '.2f')}",
        f"  VCRE mean (px)     : {_f('vcre_mean_px', '.2f')}",
        f"  VCRE (% diag)      : {_f('vcre_pct', '.3f')}",
    ]

    # ── Slope / HDA
    lines += ["", sep, "  SLOPE & HDA", sep, hdr3]
    slope_keys = [
        ("Slope corr",          "slope_corr"),
        ("Slope MAE (°)",       "slope_mae"),
        ("Slope RMSE (°)",      "slope_rmse"),
    ]
    for label, key in slope_keys:
        v0 = _f(f"v0_{key}")
        v1 = _f(f"v1_{key}")
        av = _f(f"avg_{key}")
        lines.append(f"  {label:<22} {v0:>10} {v1:>10} {av:>10}")

    lines.append("")
    lines.append(f"  {'HDA threshold':<22} {'agree':>8} {'miss':>8} {'false alm':>9}")
    for t in [5, 10, 15, 20]:
        ag = _pf(f"avg_slope_agree_{t}deg")
        mi = _pf(f"avg_slope_miss_{t}deg")
        fa = _pf(f"avg_slope_false_alarm_{t}deg")
        lines.append(f"  {t}°{'':<20} {ag:>8} {mi:>8} {fa:>9}")

    # ── Terrain / relief
    lines += ["", sep, "  TERRAIN / RELIEF", sep, hdr3]
    terrain_keys = [
        ("Slope SSIM",          "slope_ssim"),
        ("Relief corr",         "relief_corr"),
        ("Roughness corr",      "roughness_corr"),
        ("Curvature corr",      "curvature_corr"),
        ("Crater IoU",          "crater_combined_iou"),
        ("Crater rim IoU",      "crater_rim_iou"),
        ("Crater int IoU",      "crater_interior_iou"),
        ("Aspect hist corr",    "aspect_hist_corr"),
        ("Peak recall (%)",     "peak_location_recall"),
        ("Valley recall (%)",   "valley_location_recall"),
    ]
    for label, key in terrain_keys:
        v0 = _f(f"v0_{key}")
        v1 = _f(f"v1_{key}")
        av = _f(f"avg_{key}")
        lines.append(f"  {label:<22} {v0:>10} {v1:>10} {av:>10}")

    lines += ["", "=" * 62, ""]

    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return path


def save_results(
    df: pd.DataFrame,
    output_dir,
    prefix: str = "results",
) -> None:
    """Save full results CSV and per-model summary JSON.

    Creates:
        {output_dir}/{prefix}_full.csv    — all per-pair metrics
        {output_dir}/{prefix}_summary.json — per-model {metric: {median, mean, std}}

    Args:
        df         : full results DataFrame
        output_dir : output directory (will be created if needed)
        prefix     : file name prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{prefix}_full.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFull results saved to {csv_path}")

    # Per-model summary JSON
    summary: Dict = {}
    for mname in df["Model"].unique():
        ms = df[df["Model"] == mname]
        summary[mname] = {}
        for col in df.columns:
            if col in ("Model", "Folder", "pair", "error", "alignment_method"):
                continue
            vals = ms[col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(vals) == 0:
                continue
            try:
                summary[mname][col] = {
                    "median": round(float(vals.median()), 6),
                    "mean": round(float(vals.mean()), 6),
                    "std": round(float(vals.std()), 6),
                    "n": int(len(vals)),
                }
            except (TypeError, ValueError):
                pass

    json_path = output_dir / f"{prefix}_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON saved to {json_path}")
