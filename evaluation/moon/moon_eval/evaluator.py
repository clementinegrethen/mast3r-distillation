"""
moon_eval/evaluator.py — MoonEvaluator: main orchestrator class.

Runs the full evaluation pipeline for one or many image pairs:
  1. Load GT (depth, intrinsics, cam2world)
  2. Run reconstruction (global_aligner or sparse_ga)
  3. Scene-level Sim(3) alignment + ICP refinement (one transform for both views)
  4. Per-view classic metrics (accuracy/completeness/chamfer, depth, 3D)
  5. Per-view terrain metrics (slope, HDA, curvature, craters, relief)
  6. Camera pose metrics (Essential matrix + aligner)
  7. Overlap consistency
  8. Average all per-view metrics
  9. Save PLY point clouds (optional)
"""

import numpy as np
import torch
import open3d as o3d
import time
from pathlib import Path
from typing import Dict, List, Optional

from .alignment import improved_gt_alignment, apply_sim3
from .gt_loader import load_gt_view
from .reconstruction import get_reconstruction
from .reporter import save_pair_report_txt
from .metrics.classic import (
    compute_accuracy_completeness,
    compute_depth_metrics,
    compute_3d_metrics,
    compute_profile_metrics,
    compute_overlap_consistency,
)
from .metrics.camera import compute_camera_metrics_for_pair, compute_pose_from_aligner
from .metrics.terrain import (
    compute_slope_map,
    compute_slope_metrics,
    compute_hda_metrics,
    compute_curvature_maps,
    compute_roughness_map,
    compute_relief_metrics,
)

# Canonical crop dimensions (DUSt3R centre-crop of 512×512 input)
Hc, Wc = 384, 512


def _prefix_dict(d: dict, prefix: str) -> dict:
    """Return a copy of dict d with every key prefixed by prefix + '_'."""
    return {f"{prefix}_{k}": v for k, v in d.items()}


class MoonEvaluator:
    """Full evaluation pipeline for a MASt3R model on lunar stereo pairs.

    Parameters
    ----------
    model        : Loaded MASt3R / student model (eval mode).
    model_name   : Human-readable name used in output paths and CSV columns.
    device       : torch device string or object.
    K_GT         : (3, 3) known camera intrinsics.
    output_root  : Root directory for PLY / transform file outputs.
    mode         : Reconstruction backend — 'global_aligner' | 'sparse_ga' | 'sparse_ga_depth'.
    use_icp      : Whether to apply ICP refinement after Umeyama alignment.
    save_ply     : Whether to save aligned PLY point clouds per pair.
    n_matches    : Number of top matches for Essential-matrix camera metrics.
    emat_threshold : RANSAC pixel threshold for Essential-matrix estimation.
    min_valid_pts  : Minimum number of valid correspondences to proceed.
    verbose      : Print alignment info per pair.
    reconstruction_kwargs : Extra kwargs forwarded to get_reconstruction().
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_name: str,
        device,
        K_GT: np.ndarray,
        output_root,
        mode: str = "global_aligner",
        use_icp: bool = True,
        save_ply: bool = True,
        save_viz: bool = False,
        n_matches: int = 2000,
        emat_threshold: float = 1.0,
        min_valid_pts: int = 100,
        verbose: bool = False,
        **reconstruction_kwargs,
    ):
        self.model = model
        self.model_name = model_name
        self.device = device
        self.K_GT = K_GT
        self.output_root = Path(output_root)
        self.mode = mode
        self.use_icp = use_icp
        self.save_ply = save_ply
        self.save_viz = save_viz
        self.n_matches = n_matches
        self.emat_threshold = emat_threshold
        self.min_valid_pts = min_valid_pts
        self.verbose = verbose
        self.reconstruction_kwargs = reconstruction_kwargs

    # ─────────────────────────────────────────────────────────────────────────
    # Single-pair evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate_pair(
        self,
        img1_path,
        img2_path,
        gt_folder,
    ) -> Dict:
        """Run the full evaluation pipeline for one image pair.

        Returns a flat dict of all metrics, ready to be stored as a DataFrame row.
        """
        gt_folder = Path(gt_folder)
        name0 = Path(img1_path).stem
        name1 = Path(img2_path).stem
        pair_key = f"{name0}_{name1}"

        result: Dict = {
            "pair": pair_key,
            "gt_focal": float(self.K_GT[0, 0]),
        }

        # ── Step 1: Load GT ──────────────────────────────────────────────────
        gt_pts0, depth_gt0, Kc0, T_wc0, mask_g0 = load_gt_view(gt_folder, name0, Hc, Wc)
        gt_pts1, depth_gt1, Kc1, T_wc1, mask_g1 = load_gt_view(gt_folder, name1, Hc, Wc)

        # GT scene statistics — used for scale-normalised metrics.
        #
        # gt_median_depth : median depth of the combined GT scene.
        #   Used to compute AbsRel = error / gt_median_depth  (unitless, ∈ [0,1]).
        #   This is the standard "Absolute Relative Error" from depth estimation
        #   literature (DUSt3R, VGGT, NerfingMVS…). It makes results comparable
        #   across scenes at different altitudes without any unit assumptions.
        #
        # gt_terrain_span : p95 − p5 depth range — kept for reference and the
        #   coverage-by-threshold display in the text report.
        gt_z_all = np.concatenate([
            depth_gt0[mask_g0.reshape(Hc, Wc)],
            depth_gt1[mask_g1.reshape(Hc, Wc)],
        ])
        gt_z_all = gt_z_all[np.isfinite(gt_z_all)]
        if len(gt_z_all) > 10:
            gt_median_depth  = float(np.median(gt_z_all))
            gt_terrain_span  = float(
                np.percentile(gt_z_all, 95) - np.percentile(gt_z_all, 5)
            )
        else:
            gt_median_depth = 1.0
            gt_terrain_span = 1.0
        result["gt_median_depth"]  = gt_median_depth
        result["gt_terrain_span"]  = gt_terrain_span

        # GSD = ground sampling distance (m/px) at median depth.
        # GSD = depth / focal — the pixel footprint on the terrain.
        # Normalising 3D errors by GSD gives a "pixel-equivalent error"
        # that is altitude-independent and directly comparable across scenes.
        gt_focal_px = float(self.K_GT[0, 0])
        gt_gsd = gt_median_depth / gt_focal_px if gt_focal_px > 0 else 1.0
        result["gt_gsd"] = gt_gsd

        # ── Step 2: Reconstruction (with inference profiling) ────────────────
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)

        t_infer_start = time.perf_counter()

        pts3d_list, depthmaps, confs, poses, focals = get_reconstruction(
            self.model,
            self.device,
            img1_path,
            img2_path,
            self.K_GT,
            mode=self.mode,
            verbose=self.verbose,
            **self.reconstruction_kwargs,
        )

        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t_infer_end = time.perf_counter()

        result["inference_time_s"] = round(t_infer_end - t_infer_start, 4)
        if torch.cuda.is_available() and self.device.type == "cuda":
            result["peak_gpu_mem_mb"] = round(
                torch.cuda.max_memory_allocated(self.device) / 1e6, 1
            )

        pts3d_v0 = np.array(pts3d_list[0]).reshape(-1, 3)
        pts3d_v1 = np.array(pts3d_list[1]).reshape(-1, 3)
        conf_v0 = np.array(confs[0]).reshape(-1).astype(float)
        conf_v1 = np.array(confs[1]).reshape(-1).astype(float)

        result["optimized_focal_v0"] = float(focals[0]) if focals.ndim > 0 else float(focals)
        result["optimized_focal_v1"] = float(focals[1]) if len(focals) > 1 else float(focals)

        # ── Step 3: Build combined correspondences for scene-level alignment ─
        mask_ok0 = (conf_v0 >= 1.0) & mask_g0 & np.isfinite(pts3d_v0).all(axis=1)
        mask_ok1 = (conf_v1 >= 1.0) & mask_g1 & np.isfinite(pts3d_v1).all(axis=1)

        n_valid_total = mask_ok0.sum() + mask_ok1.sum()
        if n_valid_total < self.min_valid_pts:
            result["error"] = f"Too few valid pts ({n_valid_total})"
            return result

        pred_combined = np.vstack([pts3d_v0[mask_ok0], pts3d_v1[mask_ok1]])
        gt_combined = np.vstack([gt_pts0[mask_ok0], gt_pts1[mask_ok1]])

        # ── Step 4: Scene-level Sim(3) + ICP alignment ──────────────────────
        # Pass confidence scores for confidence-weighted alignment fallback
        conf_combined = np.concatenate([conf_v0[mask_ok0], conf_v1[mask_ok1]])

        if self.use_icp:
            T_sim3, scale, align_method = improved_gt_alignment(
                pred_combined, gt_combined,
                conf=conf_combined,
                verbose=self.verbose,
            )
        else:
            from .alignment import align_sim3_ransac
            T_sim3, scale, n_inl = align_sim3_ransac(pred_combined, gt_combined, Nsub=8000)
            align_method = "ransac_only"

        result["scene_scale"] = float(scale)
        result["scene_scale_err_pct"] = float(abs(scale - 1.0) * 100)
        result["alignment_method"] = align_method
        result["scene_n_valid"] = int(n_valid_total)

        # ── Step 5: Apply the SAME transform to both views ───────────────────
        aligned_v0 = apply_sim3(pts3d_v0, T_sim3)
        aligned_v1 = apply_sim3(pts3d_v1, T_sim3)

        # ── Step 5b: Global scene metrics on combined cloud ───────────────────
        # Compute on the full combined pred / GT cloud (v0 + v1), no per-view split.
        # This gives Chamfer / Hausdorff on the whole scene rather than per view.
        try:
            pred_all = np.vstack([aligned_v0[mask_ok0], aligned_v1[mask_ok1]])
            gt_all   = np.vstack([gt_pts0[mask_ok0], gt_pts1[mask_ok1]])
            if len(pred_all) > 0 and len(gt_all) > 0:
                scene_ac = compute_accuracy_completeness(
                    pred_all, gt_all, max_pts=40000, thresholds=(0.5, 1.0, 2.0)
                )
                for k, v in scene_ac.items():
                    result[f"scene_{k}"] = v
                # Hausdorff (p95, not max, to be outlier-robust)
                from scipy.spatial import cKDTree as _KD
                rng_h = np.random.RandomState(0)
                p_sub = pred_all[rng_h.choice(len(pred_all), min(len(pred_all), 40000), replace=False)]
                g_sub = gt_all[rng_h.choice(len(gt_all), min(len(gt_all), 40000), replace=False)]
                d_p2g, _ = _KD(g_sub).query(p_sub, k=1)
                d_g2p, _ = _KD(p_sub).query(g_sub, k=1)
                result["scene_hausdorff_p95"] = float(max(
                    np.percentile(d_p2g, 95), np.percentile(d_g2p, 95)
                ))
                result["scene_hausdorff_max"] = float(max(d_p2g.max(), d_g2p.max()))
                # AbsRel for scene metrics: error / gt_median_depth  (unitless, ∈ [0,1])
                # Standard depth-estimation normalisation (DUSt3R / VGGT papers).
                if gt_median_depth > 0:
                    for k in ("accuracy", "completeness", "chamfer",
                              "acc_median", "compl_median",
                              "hausdorff_p95", "hausdorff_max"):
                        sk = f"scene_{k}"
                        if sk in result:
                            result[f"{sk}_absrel"] = result[sk] / gt_median_depth
        except Exception as e:
            if self.verbose:
                print(f"  scene global metrics failed: {e}")

        # ── Step 6 + 7: Per-view metrics ─────────────────────────────────────
        view_pairs = [
            (0, aligned_v0, gt_pts0, depth_gt0, mask_ok0),
            (1, aligned_v1, gt_pts1, depth_gt1, mask_ok1),
        ]

        # Store terrain maps per view for optional visualization
        _viz_data: List[dict] = []

        for view_idx, (vi, aligned, gt_pts, depth_gt, mask_ok) in enumerate(view_pairs):
            prefix = f"v{vi}"
            result[f"{prefix}_n_pts"] = int(mask_ok.sum())

            # Reconstruct aligned depth map from Z component
            depth_pred_map = aligned.reshape(Hc, Wc, 3)[..., 2]
            depth_gt_map = gt_pts.reshape(Hc, Wc, 3)[..., 2]
            mask_2d = mask_ok.reshape(Hc, Wc)

            # Eroded mask for gradient-based metrics (slope, curvature, roughness).
            # Sobel/gradient operators are unreliable at mask boundaries — a 4-pixel
            # binary erosion removes those spurious edge values without losing much
            # of the interior region.
            from scipy.ndimage import binary_erosion
            _struct = np.ones((9, 9), bool)   # erode 4px on each side
            mask_eroded = binary_erosion(mask_2d, structure=_struct, border_value=0)
            if mask_eroded.sum() < 50:
                mask_eroded = mask_2d   # fallback: too small, keep original

            # 3D metrics  (full mask — not gradient-based, no border issue)
            m3d = compute_3d_metrics(aligned, gt_pts, mask_ok)
            result.update(_prefix_dict(m3d, prefix))
            # AbsRel versions: error / gt_median_depth  (unitless)
            if gt_median_depth > 0:
                for k in ("rmse", "mae_3d"):
                    if f"{prefix}_{k}" in result:
                        result[f"{prefix}_{k}_absrel"] = result[f"{prefix}_{k}"] / gt_median_depth

            # Accuracy / completeness / chamfer (full mask)
            if mask_ok.sum() >= 10:
                ac = compute_accuracy_completeness(
                    aligned[mask_ok], gt_pts[mask_ok], max_pts=20000
                )
                result.update(_prefix_dict(ac, prefix))
                if gt_median_depth > 0:
                    for k in ("accuracy", "completeness", "chamfer", "acc_median", "compl_median"):
                        if f"{prefix}_{k}" in result:
                            result[f"{prefix}_{k}_absrel"] = result[f"{prefix}_{k}"] / gt_median_depth

            # Depth map metrics (full mask — MAE/RMSE are point-to-point, no gradient)
            dm = compute_depth_metrics(depth_pred_map, depth_gt_map, mask_2d)
            result.update(_prefix_dict(dm, prefix))
            if gt_median_depth > 0:
                for k in ("depth_mae", "depth_rmse"):
                    if f"{prefix}_{k}" in result:
                        result[f"{prefix}_{k}_absrel"] = result[f"{prefix}_{k}"] / gt_median_depth

            # Profile metrics (full mask)
            pm = compute_profile_metrics(depth_pred_map, depth_gt_map, mask_2d, n_rows=1)
            result.update(_prefix_dict(pm, prefix))

            # Slope metrics — use eroded mask to avoid border artefacts
            slope_gt_map, aspect_gt_map = compute_slope_map(depth_gt_map)
            slope_pred_map, aspect_pred_map = compute_slope_map(depth_pred_map)

            sm = compute_slope_metrics(slope_pred_map, slope_gt_map, mask_eroded)
            result.update(_prefix_dict(sm, prefix))

            hda = compute_hda_metrics(slope_pred_map, slope_gt_map, mask_eroded)
            result.update(_prefix_dict(hda, prefix))

            # Curvature + roughness — use eroded mask
            try:
                curv_pred_map, _ = compute_curvature_maps(depth_pred_map)
                curv_gt_map, _ = compute_curvature_maps(depth_gt_map)
            except Exception:
                curv_pred_map = np.zeros_like(depth_pred_map)
                curv_gt_map = np.zeros_like(depth_gt_map)

            try:
                rough_pred_map = compute_roughness_map(depth_pred_map)
                rough_gt_map = compute_roughness_map(depth_gt_map)
            except Exception:
                rough_pred_map = np.zeros_like(depth_pred_map)
                rough_gt_map = np.zeros_like(depth_gt_map)

            # Relief / terrain metrics — eroded mask
            try:
                rm = compute_relief_metrics(depth_pred_map, depth_gt_map, mask_eroded)
                result.update(_prefix_dict(rm, prefix))
            except Exception as e:
                if self.verbose:
                    print(f"  [{prefix}] relief_metrics failed: {e}")

            # Collect data for visualization
            _viz_data.append({
                "depth_pred": depth_pred_map,
                "depth_gt": depth_gt_map,
                "mask": mask_2d,
                "slope_pred": slope_pred_map,
                "slope_gt": slope_gt_map,
                "aspect_pred": aspect_pred_map,
                "aspect_gt": aspect_gt_map,
                "roughness_pred": rough_pred_map,
                "roughness_gt": rough_gt_map,
                "curvature_pred": curv_pred_map,
                "curvature_gt": curv_gt_map,
                # per-view metric subset for viz labels
                "metrics": {
                    k.replace(f"{prefix}_", ""): result[k]
                    for k in result
                    if k.startswith(f"{prefix}_")
                },
            })

        # ── Step 8: Camera pose metrics ──────────────────────────────────────
        try:
            cam_metrics = compute_camera_metrics_for_pair(
                self.model,
                self.device,
                img1_path,
                img2_path,
                gt_folder,
                self.K_GT,
                n_matches=self.n_matches,
                threshold=self.emat_threshold,
                poses_from_aligner=poses if poses is not None else None,
            )
            result.update(cam_metrics)
        except Exception as e:
            if self.verbose:
                print(f"  camera metrics failed: {e}")

        # ── Step 9: Overlap consistency ──────────────────────────────────────
        try:
            pred_v0_valid = aligned_v0[mask_ok0]
            pred_v1_valid = aligned_v1[mask_ok1]
            terrain_span = float(np.linalg.norm(
                np.ptp(
                    np.vstack([gt_pts0[mask_g0], gt_pts1[mask_g1]]),
                    axis=0,
                )
            ))
            ov = compute_overlap_consistency(pred_v0_valid, pred_v1_valid, terrain_span)
            result.update(ov)
        except Exception as e:
            if self.verbose:
                print(f"  overlap_consistency failed: {e}")

        # ── Step 10: Average per-view metrics → avg_* ────────────────────────
        _metric_avg_keys = [
            "rmse", "mae_3d", "pearson_z",
            "rmse_absrel", "mae_3d_absrel",
            "accuracy", "completeness", "chamfer", "acc_median", "compl_median",
            "accuracy_absrel", "completeness_absrel", "chamfer_absrel",
            "acc_median_absrel", "compl_median_absrel",
            "depth_mae", "depth_rmse", "depth_pearson", "depth_ssim", "silog",
            "depth_mae_absrel", "depth_rmse_absrel",
            "delta1", "delta2", "delta3",
            "profile_mae", "profile_corr",
            "slope_corr", "slope_mae", "slope_rmse",
            "slope_agree_5deg", "slope_miss_5deg", "slope_false_alarm_5deg",
            "slope_agree_10deg", "slope_miss_10deg", "slope_false_alarm_10deg",
            "slope_agree_15deg", "slope_miss_15deg", "slope_false_alarm_15deg",
            "slope_agree_20deg", "slope_miss_20deg", "slope_false_alarm_20deg",
            "roughness_corr", "roughness_mae",
            "curvature_corr", "curvature_mae",
            "crater_rim_iou", "crater_interior_iou", "crater_combined_iou",
            "relief_corr", "slope_ssim", "aspect_hist_corr",
            "peak_location_recall", "valley_location_recall",
        ]
        for k in _metric_avg_keys:
            v0 = result.get(f"v0_{k}")
            v1 = result.get(f"v1_{k}")
            if v0 is not None and v1 is not None:
                try:
                    result[f"avg_{k}"] = float((float(v0) + float(v1)) / 2)
                except (TypeError, ValueError):
                    pass

        # ── Step 10b: GSD-relative metrics (error / GSD = pixel-equivalent) ──
        if gt_gsd > 0:
            for k in ("avg_rmse", "avg_mae_3d", "avg_chamfer", "avg_accuracy",
                       "avg_completeness", "avg_depth_mae", "avg_depth_rmse",
                       "scene_chamfer", "scene_accuracy", "scene_completeness",
                       "scene_hausdorff_p95"):
                if k in result:
                    result[f"{k}_gsd"] = result[k] / gt_gsd

        # ── Step 11: Save PLY ────────────────────────────────────────────────
        if self.save_ply:
            self._save_ply(
                aligned_v0[mask_ok0], aligned_v1[mask_ok1],
                gt_pts0[mask_ok0], gt_pts1[mask_ok1],
                T_sim3, gt_folder, pair_key,
            )

        # ── Step 12: Per-pair text report ────────────────────────────────────
        try:
            report_dir = self.output_root / self.model_name / gt_folder.name / pair_key
            save_pair_report_txt(result, report_dir)
        except Exception as e:
            if self.verbose:
                print(f"  pair report failed: {e}")

        # ── Step 13: Visualizations ───────────────────────────────────────────
        if self.save_viz and len(_viz_data) >= 2:
            try:
                from .visualizer import visualize_pair
                viz_out = self.output_root / self.model_name / gt_folder.name / pair_key / "viz"
                vd0, vd1 = _viz_data[0], _viz_data[1]
                visualize_pair(
                    depth_pred_v0=vd0["depth_pred"],
                    depth_gt_v0=vd0["depth_gt"],
                    depth_pred_v1=vd1["depth_pred"],
                    depth_gt_v1=vd1["depth_gt"],
                    mask_v0=vd0["mask"],
                    mask_v1=vd1["mask"],
                    slope_pred_v0=vd0["slope_pred"],
                    slope_gt_v0=vd0["slope_gt"],
                    aspect_pred_v0=vd0["aspect_pred"],
                    aspect_gt_v0=vd0["aspect_gt"],
                    roughness_pred_v0=vd0["roughness_pred"],
                    roughness_gt_v0=vd0["roughness_gt"],
                    curvature_pred_v0=vd0["curvature_pred"],
                    curvature_gt_v0=vd0["curvature_gt"],
                    metrics_v0=vd0["metrics"],
                    slope_pred_v1=vd1["slope_pred"],
                    slope_gt_v1=vd1["slope_gt"],
                    aspect_pred_v1=vd1["aspect_pred"],
                    aspect_gt_v1=vd1["aspect_gt"],
                    roughness_pred_v1=vd1["roughness_pred"],
                    roughness_gt_v1=vd1["roughness_gt"],
                    curvature_pred_v1=vd1["curvature_pred"],
                    curvature_gt_v1=vd1["curvature_gt"],
                    metrics_v1=vd1["metrics"],
                    out_dir=viz_out,
                )
                if self.verbose:
                    print(f"  Visualizations saved to {viz_out}")
            except Exception as e:
                if self.verbose:
                    print(f"  Visualization failed: {e}")

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Batch evaluation over a folder
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate_folder(
        self,
        gt_folder_path,
        folder_key: str,
        max_pairs: Optional[int] = None,
        force: bool = False,
    ) -> List[Dict]:
        """Evaluate all consecutive image pairs in a GT folder.

        Pairs: sorted images matched as (even, odd) indices — same as
        eval_per_view.py / evalAllPairs_multi.py.

        Args:
            gt_folder_path : path to folder containing .jpg + .npz + .exr files
            folder_key     : short name for this folder (used in CSV)
            max_pairs      : if set, sample up to this many pairs uniformly

        Returns list of result dicts, each annotated with 'Model' and 'Folder'.
        """
        gt_folder = Path(gt_folder_path)
        images = sorted(gt_folder.glob("*.jpg"))
        pairs = list(zip(images[0::2], images[1::2]))

        if max_pairs is not None and max_pairs < len(pairs):
            step = max(1, len(pairs) // max_pairs)
            pairs = pairs[::step][:max_pairs]

        n_skipped = 0
        results = []
        for img1, img2 in pairs:
            pair_key = f"{img1.stem}_{img2.stem}"
            report_path = self.output_root / self.model_name / gt_folder.name / pair_key / "pair_report.txt"
            if report_path.exists() and not force:
                n_skipped += 1
                continue
            print(f"  [{self.model_name}] {img1.name} vs {img2.name}")
            try:
                row = self.evaluate_pair(img1, img2, gt_folder)
                row["Model"] = self.model_name
                row["Folder"] = folder_key
                results.append(row)
            except Exception as e:
                import traceback
                print(f"    FAILED: {e}")
                if self.verbose:
                    traceback.print_exc()
                results.append({
                    "Model": self.model_name,
                    "Folder": folder_key,
                    "pair": pair_key,
                    "error": str(e),
                })

        if n_skipped:
            print(f"  [{self.model_name}] Skipped {n_skipped}/{len(pairs)} pairs (already done)")

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _save_ply(self, aligned_v0, aligned_v1, gt_pts0, gt_pts1, T_sim3,
                   gt_folder, pair_key):
        """Save aligned prediction and GT point clouds as PLY files."""
        try:
            out_dir = self.output_root / self.model_name / gt_folder.name / pair_key
            out_dir.mkdir(parents=True, exist_ok=True)

            combined_pred = np.vstack([aligned_v0, aligned_v1])
            combined_gt = np.vstack([gt_pts0, gt_pts1])

            # Remove non-finite points before saving
            pred_ok = np.isfinite(combined_pred).all(axis=1)
            gt_ok = np.isfinite(combined_gt).all(axis=1)

            pcd_pred = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(combined_pred[pred_ok])
            )
            pcd_gt = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(combined_gt[gt_ok])
            )

            o3d.io.write_point_cloud(str(out_dir / "aligned_pred.ply"), pcd_pred)
            o3d.io.write_point_cloud(str(out_dir / "aligned_gt.ply"), pcd_gt)
            np.savetxt(str(out_dir / "transform_sim3.txt"), T_sim3)
        except Exception as e:
            if self.verbose:
                print(f"    Warning: failed to save PLYs: {e}")
