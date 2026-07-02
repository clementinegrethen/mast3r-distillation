"""
moon_eval/visualizer.py — Visualization module for MASt3R Moon reconstruction evaluation.

Generates per-pair diagnostic figures:
  01_depth_comparison_basic    — depth maps + difference
  02_depth_comparison_hillshade— hillshading, terrain colormap, absolute error
  03_central_profile           — GT vs Pred depth along central row
  04_error_chamfer_analysis    — error histogram, scatter, Chamfer bar, coverage
  05_slope_aspect_maps         — slope/aspect/roughness maps (adaptive scale)
  06_terrain_classification    — 5-class terrain map (flat/gentle/steep/ridge/valley)

Figures are saved to:
  {output_root}/{model_name}/{gt_folder_name}/{pair_key}/{fig_name}.png
"""

import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LightSource


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, out_dir: Path, name: str, dpi: int = 150) -> Path:
    """Save figure to {out_dir}/{name}.png at `dpi` DPI."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _robust_vlim(data: np.ndarray, pct_lo: float = 2, pct_hi: float = 98):
    """Percentile-based vmin/vmax, ignoring non-finite values."""
    finite = data[np.isfinite(data)]
    if len(finite) == 0:
        return 0.0, 1.0
    return float(np.percentile(finite, pct_lo)), float(np.percentile(finite, pct_hi))


def _sym_vlim(data: np.ndarray, pct: float = 98) -> float:
    """Symmetric vmax from percentile of |data|."""
    finite = data[np.isfinite(data)]
    if len(finite) == 0:
        return 1.0
    return float(np.percentile(np.abs(finite), pct))


def _terrain_classification(slope_deg: np.ndarray, curvature: np.ndarray) -> np.ndarray:
    """
    5-class terrain map:
      0 — flat         (slope < 5°)
      1 — gentle slope (5–15°)
      2 — steep slope  (≥ 15°)
      3 — ridge        (steep + curvature < −0.01)
      4 — valley       (steep + curvature > +0.01)
    """
    cls = np.zeros_like(slope_deg, dtype=np.int32)
    gentle = (slope_deg >= 5) & (slope_deg < 15)
    steep = slope_deg >= 15
    cls[gentle] = 1
    cls[steep] = 2
    cls[steep & (curvature < -0.01)] = 3
    cls[steep & (curvature > 0.01)] = 4
    return cls


# ─────────────────────────────────────────────────────────────────────────────
# Individual figure generators
# ─────────────────────────────────────────────────────────────────────────────

def fig_depth_basic(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    out_dir: Path,
    prefix: str = "",
    view_label: str = "",
) -> Path:
    """Fig 01 — Side-by-side depth maps + signed difference."""
    diff = depth_pred - depth_gt
    vmin_d, vmax_d = _robust_vlim(np.stack([depth_pred, depth_gt]))
    vmax_diff = _sym_vlim(diff)

    title_sfx = f" [{view_label}]" if view_label else ""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    fig.suptitle(f"Depth Comparison{title_sfx}", fontsize=13)

    im0 = axes[0].imshow(depth_pred, cmap="viridis", vmin=vmin_d, vmax=vmax_d)
    axes[0].set_title("Depth — Pred")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(depth_gt, cmap="viridis", vmin=vmin_d, vmax=vmax_d)
    axes[1].set_title("Depth — GT")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-vmax_diff, vmax=vmax_diff)
    axes[2].set_title("Difference (Pred − GT)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    name = f"{prefix}01_depth_comparison_basic" if prefix else "01_depth_comparison_basic"
    return _save_fig(fig, out_dir, name)


def fig_depth_hillshade(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    out_dir: Path,
    prefix: str = "",
    view_label: str = "",
) -> Path:
    """Fig 02 — Hillshading (LightSource), terrain colormap, absolute error."""
    diff = depth_pred - depth_gt
    vmax_diff = _sym_vlim(diff)

    ls = LightSource(azdeg=315, altdeg=45)
    hs_gt = ls.hillshade(depth_gt, vert_exag=2)
    hs_pred = ls.hillshade(depth_pred, vert_exag=2)

    vmin_d, vmax_d = _robust_vlim(np.stack([depth_pred, depth_gt]))
    abs_err = np.abs(diff)

    title_sfx = f" [{view_label}]" if view_label else ""
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    fig.suptitle(f"Hillshade & Terrain{title_sfx}", fontsize=13)

    im0 = axes[0, 0].imshow(depth_gt, cmap="terrain", vmin=vmin_d, vmax=vmax_d)
    axes[0, 0].set_title("Depth GT")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(depth_pred, cmap="terrain", vmin=vmin_d, vmax=vmax_d)
    axes[0, 1].set_title("Depth Pred")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(diff, cmap="RdBu_r", vmin=-vmax_diff, vmax=vmax_diff)
    axes[0, 2].set_title("Difference (Pred − GT)")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    axes[1, 0].imshow(hs_gt, cmap="gray")
    axes[1, 0].set_title("Hillshade GT")

    axes[1, 1].imshow(hs_pred, cmap="gray")
    axes[1, 1].set_title("Hillshade Pred")

    vmax_abs = _sym_vlim(abs_err, pct=98)
    im5 = axes[1, 2].imshow(abs_err, cmap="Reds", vmin=0, vmax=vmax_abs)
    axes[1, 2].set_title("Absolute Error")
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)

    name = f"{prefix}02_depth_comparison_hillshade" if prefix else "02_depth_comparison_hillshade"
    return _save_fig(fig, out_dir, name)


def fig_central_profile(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    out_dir: Path,
    prefix: str = "",
    view_label: str = "",
) -> Path:
    """Fig 03 — Depth profile along central row."""
    H = depth_pred.shape[0]
    row = H // 2
    gt_row = depth_gt[row]
    pred_row = depth_pred[row]

    title_sfx = f" [{view_label}]" if view_label else ""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(gt_row, label="GT", linewidth=2, color="steelblue")
    ax.plot(pred_row, label="Pred", linewidth=2, color="tomato", linestyle="--")
    ax.set_title(f"Central Depth Profile (row {row}){title_sfx}")
    ax.set_xlabel("Pixel column")
    ax.set_ylabel("Depth (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    name = f"{prefix}03_central_profile" if prefix else "03_central_profile"
    return _save_fig(fig, out_dir, name)


def fig_error_chamfer(
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    mask: np.ndarray,
    metrics: dict,
    out_dir: Path,
    prefix: str = "",
    view_label: str = "",
) -> Path:
    """Fig 04 — Error histogram, GT vs Pred scatter, Chamfer bar chart, coverage."""
    title_sfx = f" [{view_label}]" if view_label else ""

    diff_flat = (depth_pred - depth_gt)[mask]
    gt_flat = depth_gt[mask]
    pred_flat = depth_pred[mask]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle(f"Error & Distance Analysis{title_sfx}", fontsize=13)

    # --- Error histogram ---
    ax = axes[0, 0]
    ax.hist(diff_flat, bins=100, alpha=0.75, edgecolor="black", linewidth=0.3, color="cornflowerblue")
    ax.axvline(0, color="red", linestyle="--", alpha=0.7, label="zero error")
    ax.axvline(float(np.mean(diff_flat)), color="green", linestyle="-", alpha=0.8,
               label=f"mean={np.mean(diff_flat):.3f}")
    ax.set_title("Error Histogram (Pred − GT)")
    ax.set_xlabel("Error (m)")
    ax.set_ylabel("Pixel count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- GT vs Pred scatter ---
    ax = axes[0, 1]
    n = min(5000, len(gt_flat))
    idx = np.random.choice(len(gt_flat), n, replace=False) if len(gt_flat) > n else np.arange(len(gt_flat))
    ax.scatter(gt_flat[idx], pred_flat[idx], alpha=0.4, s=1, color="navy")
    lo = min(gt_flat.min(), pred_flat.min())
    hi = max(gt_flat.max(), pred_flat.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="ideal")
    r_val = metrics.get("depth_pearson", metrics.get("pearson_z", float("nan")))
    ax.set_title(f"GT vs Pred  (r={r_val:.3f})" if np.isfinite(r_val) else "GT vs Pred")
    ax.set_xlabel("Depth GT (m)")
    ax.set_ylabel("Depth Pred (m)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Chamfer / distance bar chart ---
    ax = axes[1, 0]
    bar_labels = ["Chamfer", "Acc\n(pred→GT)", "Compl\n(GT→pred)"]
    bar_vals = [
        metrics.get("chamfer", metrics.get("avg_chamfer", float("nan"))),
        metrics.get("accuracy", metrics.get("avg_accuracy", float("nan"))),
        metrics.get("completeness", metrics.get("avg_completeness", float("nan"))),
    ]
    colors = ["steelblue", "lightcoral", "lightgreen"]
    finite_vals = [v if np.isfinite(v) else 0.0 for v in bar_vals]
    bars = ax.bar(bar_labels, finite_vals, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, bar_vals):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("3D Distance Metrics (m)")
    ax.set_ylabel("Distance (m)")
    ax.grid(True, alpha=0.3, axis="y")

    # --- Coverage by threshold ---
    ax = axes[1, 1]
    thresholds = [10.0, 20.0, 30.0, 40.0]
    cov_pred = [metrics.get(f"acc_pct_under_{t}", metrics.get(f"coverage_pred_to_gt_{t}m", float("nan")))
                for t in thresholds]
    cov_gt = [metrics.get(f"compl_pct_under_{t}", metrics.get(f"coverage_gt_to_pred_{t}m", float("nan")))
              for t in thresholds]
    x = np.arange(len(thresholds))
    w = 0.35
    b1 = ax.bar(x - w / 2, [v if np.isfinite(v) else 0.0 for v in cov_pred], w,
                label="Pred→GT (accuracy)", alpha=0.8, color="lightcoral")
    b2 = ax.bar(x + w / 2, [v if np.isfinite(v) else 0.0 for v in cov_gt], w,
                label="GT→Pred (completeness)", alpha=0.8, color="lightgreen")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.0f}m" for t in thresholds])
    ax.set_title("Coverage by Distance Threshold (%)")
    ax.set_ylabel("Coverage (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    name = f"{prefix}04_error_chamfer_analysis" if prefix else "04_error_chamfer_analysis"
    return _save_fig(fig, out_dir, name)


def fig_slope_aspect(
    slope_pred: np.ndarray,
    slope_gt: np.ndarray,
    aspect_pred: np.ndarray,
    aspect_gt: np.ndarray,
    roughness_pred: np.ndarray,
    roughness_gt: np.ndarray,
    mask: np.ndarray,
    out_dir: Path,
    prefix: str = "",
    view_label: str = "",
) -> Path:
    """Fig 05 — Slope / aspect / roughness maps with adaptive colour scales."""
    title_sfx = f" [{view_label}]" if view_label else ""

    # Adaptive scale: common range for GT+pred slopes
    valid_slopes = np.concatenate([slope_gt[mask], slope_pred[mask]])
    valid_slopes = valid_slopes[np.isfinite(valid_slopes)]
    if len(valid_slopes) > 0:
        vmin_s = float(np.percentile(valid_slopes, 2))
        vmax_s = float(np.percentile(valid_slopes, 98))
    else:
        vmin_s, vmax_s = 0.0, 30.0

    slope_diff = slope_pred - slope_gt
    vmax_sd = _sym_vlim(slope_diff[mask & np.isfinite(slope_diff)])

    roughness_diff = roughness_pred - roughness_gt
    vmax_rd = _sym_vlim(roughness_diff[mask & np.isfinite(roughness_diff)])

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
    fig.suptitle(f"Slope / Aspect / Roughness{title_sfx}", fontsize=13)

    im0 = axes[0, 0].imshow(slope_gt, cmap="YlOrRd", vmin=vmin_s, vmax=vmax_s)
    axes[0, 0].set_title("Slope GT (°)")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(slope_pred, cmap="YlOrRd", vmin=vmin_s, vmax=vmax_s)
    axes[0, 1].set_title("Slope Pred (°)")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(slope_diff, cmap="RdBu_r", vmin=-vmax_sd, vmax=vmax_sd)
    axes[0, 2].set_title("Slope Difference (°)")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    im3 = axes[1, 0].imshow(aspect_gt, cmap="hsv", vmin=0, vmax=360)
    axes[1, 0].set_title("Aspect GT (°)")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    im4 = axes[1, 1].imshow(aspect_pred, cmap="hsv", vmin=0, vmax=360)
    axes[1, 1].set_title("Aspect Pred (°)")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    im5 = axes[1, 2].imshow(roughness_diff, cmap="RdBu_r", vmin=-vmax_rd, vmax=vmax_rd)
    axes[1, 2].set_title("Roughness Difference")
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)

    name = f"{prefix}05_slope_aspect_maps" if prefix else "05_slope_aspect_maps"
    return _save_fig(fig, out_dir, name)


def fig_terrain_classification(
    slope_pred: np.ndarray,
    slope_gt: np.ndarray,
    curvature_pred: np.ndarray,
    curvature_gt: np.ndarray,
    out_dir: Path,
    prefix: str = "",
    view_label: str = "",
) -> Path:
    """Fig 06 — Terrain classification (5 classes)."""
    cls_gt = _terrain_classification(slope_gt, curvature_gt)
    cls_pred = _terrain_classification(slope_pred, curvature_pred)

    title_sfx = f" [{view_label}]" if view_label else ""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle(f"Terrain Classification{title_sfx}", fontsize=13)

    for ax, cls, label in zip(axes, [cls_gt, cls_pred], ["GT", "Pred"]):
        ax.imshow(cls, cmap="tab10", vmin=0, vmax=9)
        ax.set_title(f"Terrain Classification — {label}")

    # Shared legend
    class_colors = plt.get_cmap("tab10")(np.linspace(0, 0.4, 5))
    patches = [
        mpatches.Patch(color=class_colors[i], label=lbl)
        for i, lbl in enumerate(["flat (<5°)", "gentle (5–15°)", "steep (≥15°)", "ridge", "valley"])
    ]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=9, frameon=True)

    name = f"{prefix}06_terrain_classification" if prefix else "06_terrain_classification"
    return _save_fig(fig, out_dir, name)


def fig_hda_detection(
    slope_pred: np.ndarray,
    slope_gt: np.ndarray,
    mask: np.ndarray,
    out_dir: Path,
    thresholds: tuple = (5, 10, 15, 20),
    prefix: str = "",
    view_label: str = "",
) -> Path:
    """Fig 07 — HDA (Hazard Detection & Avoidance) slope classification.

    For each threshold t, shows a 2×2 confusion map:
      - True Negative  (GT safe,   Pred safe)   → green
      - True Positive  (GT unsafe, Pred unsafe)  → red
      - False Alarm    (GT safe,   Pred unsafe)  → orange  (mission abort)
      - Miss           (GT unsafe, Pred safe)    → purple  (dangerous!)

    One subplot per threshold, arranged in a single row.
    Also shows a bar chart of agree / false-alarm / miss rates.
    """
    title_sfx = f" [{view_label}]" if view_label else ""
    n = len(thresholds)

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 9), constrained_layout=True)
    fig.suptitle(f"HDA Slope Detection{title_sfx}", fontsize=13)

    # Colour map: 0=TN(green), 1=TP(red), 2=FA(orange), 3=miss(purple)
    cmap = matplotlib.colors.ListedColormap(["#2ca02c", "#d62728", "#ff7f0e", "#9467bd"])
    norm = matplotlib.colors.BoundaryNorm([0, 1, 2, 3, 4], cmap.N)

    agree_list, fa_list, miss_list = [], [], []

    for col, t in enumerate(thresholds):
        gt_safe   = slope_gt < t
        pred_safe = slope_pred < t

        conf_map = np.full(slope_gt.shape, -1, dtype=np.int32)
        conf_map[mask & gt_safe   & pred_safe]  = 0   # TN
        conf_map[mask & ~gt_safe  & ~pred_safe] = 1   # TP
        conf_map[mask & gt_safe   & ~pred_safe] = 2   # False Alarm
        conf_map[mask & ~gt_safe  & pred_safe]  = 3   # Miss

        # Map image
        ax_map = axes[0, col]
        disp = conf_map.copy().astype(float)
        disp[~mask] = np.nan
        ax_map.imshow(disp, cmap=cmap, norm=norm, interpolation="nearest")
        ax_map.set_title(f"threshold = {t}°", fontsize=10)
        ax_map.axis("off")

        # Stats
        n_valid = mask.sum()
        if n_valid > 0:
            ag = float((conf_map[mask] == 0).sum() + (conf_map[mask] == 1).sum()) / n_valid * 100
        else:
            ag = float("nan")
        n_safe_gt = (mask & gt_safe).sum()
        n_unsafe_gt = (mask & ~gt_safe).sum()
        fa = float((conf_map[mask] == 2).sum() / n_safe_gt * 100) if n_safe_gt > 0 else float("nan")
        mi = float((conf_map[mask] == 3).sum() / n_unsafe_gt * 100) if n_unsafe_gt > 0 else float("nan")
        agree_list.append(ag)
        fa_list.append(fa)
        miss_list.append(mi)

    # Bar chart row
    x = np.arange(n)
    w = 0.25
    ax_bar = axes[1, :]
    # merge bottom row into one axis
    for ax in ax_bar[1:]:
        ax.set_visible(False)
    ax_bar[0].set_visible(False)
    # redraw with a single spanning axis
    ax_b = fig.add_subplot(2, 1, 2)
    ax_b.bar(x - w, [v if np.isfinite(v) else 0 for v in agree_list], w,
             label="Agreement (%)", color="#2ca02c", alpha=0.85)
    ax_b.bar(x,     [v if np.isfinite(v) else 0 for v in fa_list], w,
             label="False Alarm (%)", color="#ff7f0e", alpha=0.85)
    ax_b.bar(x + w, [v if np.isfinite(v) else 0 for v in miss_list], w,
             label="Miss — DANGER (%)", color="#9467bd", alpha=0.85)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f"{t}°" for t in thresholds])
    ax_b.set_ylabel("Rate (%)")
    ax_b.set_title("HDA Rates by Threshold")
    ax_b.legend(fontsize=9)
    ax_b.grid(True, alpha=0.3, axis="y")
    ax_b.set_ylim(0, 110)

    # Shared legend for the maps
    patches = [
        mpatches.Patch(color="#2ca02c", label="True Negative  (GT safe,   Pred safe)"),
        mpatches.Patch(color="#d62728", label="True Positive  (GT unsafe, Pred unsafe)"),
        mpatches.Patch(color="#ff7f0e", label="False Alarm    (GT safe,   Pred unsafe)"),
        mpatches.Patch(color="#9467bd", label="Miss — DANGER  (GT unsafe, Pred safe)"),
    ]
    fig.legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.9)

    name = f"{prefix}07_hda_detection" if prefix else "07_hda_detection"
    return _save_fig(fig, out_dir, name)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def visualize_pair(
    depth_pred_v0: np.ndarray,
    depth_gt_v0: np.ndarray,
    depth_pred_v1: np.ndarray,
    depth_gt_v1: np.ndarray,
    mask_v0: np.ndarray,
    mask_v1: np.ndarray,
    slope_pred_v0: np.ndarray,
    slope_gt_v0: np.ndarray,
    aspect_pred_v0: np.ndarray,
    aspect_gt_v0: np.ndarray,
    roughness_pred_v0: np.ndarray,
    roughness_gt_v0: np.ndarray,
    curvature_pred_v0: np.ndarray,
    curvature_gt_v0: np.ndarray,
    metrics_v0: dict,
    slope_pred_v1: np.ndarray,
    slope_gt_v1: np.ndarray,
    aspect_pred_v1: np.ndarray,
    aspect_gt_v1: np.ndarray,
    roughness_pred_v1: np.ndarray,
    roughness_gt_v1: np.ndarray,
    curvature_pred_v1: np.ndarray,
    curvature_gt_v1: np.ndarray,
    metrics_v1: dict,
    out_dir: Path,
    dpi: int = 150,
) -> list:
    """
    Generate all 6 diagnostic figures for both views of a stereo pair.

    Returns a list of saved file paths.
    """
    out_dir = Path(out_dir)
    saved = []

    for vi, (dp, dg, mk, sp, sg, ap, ag, rp, rg, cp, cg, mv) in enumerate([
        (depth_pred_v0, depth_gt_v0, mask_v0,
         slope_pred_v0, slope_gt_v0, aspect_pred_v0, aspect_gt_v0,
         roughness_pred_v0, roughness_gt_v0, curvature_pred_v0, curvature_gt_v0, metrics_v0),
        (depth_pred_v1, depth_gt_v1, mask_v1,
         slope_pred_v1, slope_gt_v1, aspect_pred_v1, aspect_gt_v1,
         roughness_pred_v1, roughness_gt_v1, curvature_pred_v1, curvature_gt_v1, metrics_v1),
    ]):
        pfx = f"v{vi}_"
        vl = f"view {vi}"

        try:
            saved.append(fig_depth_basic(dp, dg, out_dir, prefix=pfx, view_label=vl))
        except Exception as e:
            print(f"    viz [{vl}] fig01 failed: {e}")

        try:
            saved.append(fig_depth_hillshade(dp, dg, out_dir, prefix=pfx, view_label=vl))
        except Exception as e:
            print(f"    viz [{vl}] fig02 failed: {e}")

        try:
            saved.append(fig_central_profile(dp, dg, out_dir, prefix=pfx, view_label=vl))
        except Exception as e:
            print(f"    viz [{vl}] fig03 failed: {e}")

        try:
            saved.append(fig_error_chamfer(dp, dg, mk, mv, out_dir, prefix=pfx, view_label=vl))
        except Exception as e:
            print(f"    viz [{vl}] fig04 failed: {e}")

        try:
            saved.append(fig_slope_aspect(sp, sg, ap, ag, rp, rg, mk, out_dir, prefix=pfx, view_label=vl))
        except Exception as e:
            print(f"    viz [{vl}] fig05 failed: {e}")

        try:
            saved.append(fig_terrain_classification(sp, sg, cp, cg, out_dir, prefix=pfx, view_label=vl))
        except Exception as e:
            print(f"    viz [{vl}] fig06 failed: {e}")

        try:
            saved.append(fig_hda_detection(sp, sg, mk, out_dir, prefix=pfx, view_label=vl))
        except Exception as e:
            print(f"    viz [{vl}] fig07 failed: {e}")

    return saved
