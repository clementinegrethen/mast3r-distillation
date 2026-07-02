"""
moon_eval/baselines/colmap_mvs.py — Classical SIFT-SfM + PatchMatch MVS baseline.

Pipeline for a single stereo pair:
  1. Copy the two images to a temporary directory.
  2. Extract SIFT keypoints + descriptors  (pycolmap.extract_features).
  3. Exhaustive matching                   (pycolmap.match_exhaustive).
  4. Estimate two-view geometry → recover R / t.
  5. Triangulate sparse 3D points.
  6. PatchMatch Stereo dense depth estimation (pycolmap.patch_match_stereo).
  7. Read per-image depth maps, back-project to world-frame 3D points.
  8. Sim(3) alignment (Umeyama + optional ICP) to GT point cloud.
  9. Compute the same metrics as MoonEvaluator:
       classic 3D, depth map, slope/HDA, terrain/relief, camera pose.
 10. Save visualisations identical to those produced by MoonEvaluator.

Requirements:
  pip install pycolmap   (already in the mast3r conda env)
  pycolmap must be built with CUDA for PatchMatch (CPU fallback is very slow).

Limitations / notes:
  - patch_match_stereo requires the reconstruction to be initialised —
    we use the two-view Essential-matrix geometry as the initial model.
  - If PatchMatch fails (e.g. GPU not available, too few keypoints), the
    function falls back to sparse-only results (pose metrics only, depth = NaN).
  - Dense depth is computed in the *camera* frame; we lift it to world frame
    using the estimated camera pose (not GT), so errors in pose propagate to
    the depth map. This is the correct evaluation setting for a full SfM pipeline.
"""

import shutil
import tempfile
import struct
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pycolmap

# ─────────────────────────────────────────────────────────────────────────────
# Helpers — pose metrics
# ─────────────────────────────────────────────────────────────────────────────

def _rra_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    R_err = R_est @ R_gt.T
    trace = float(np.trace(R_err))
    cos_a = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _rta_deg(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    n_est = t_est / (np.linalg.norm(t_est) + 1e-15)
    n_gt  = t_gt  / (np.linalg.norm(t_gt)  + 1e-15)
    cos_a = np.clip(float(np.dot(n_est, n_gt)), -1.0, 1.0)
    return float(np.degrees(np.arccos(np.abs(cos_a))))


def _gt_relative_pose(gt_folder: Path, stem0: str, stem1: str):
    d0 = np.load(gt_folder / f"{stem0}.npz")
    d1 = np.load(gt_folder / f"{stem1}.npz")
    T_wc0 = d0["cam2world"].astype(np.float64)
    T_wc1 = d1["cam2world"].astype(np.float64)
    T_rel = np.linalg.inv(T_wc1) @ T_wc0
    return T_rel[:3, :3], T_rel[:3, 3]


def _pycolmap_device():
    """Return pycolmap.Device.auto — lets pycolmap decide CUDA vs CPU.

    We previously used Device.cuda when torch.cuda.is_available(), but pycolmap's
    SIFT GPU support is compiled separately and may not be available even when
    PyTorch CUDA works. Using 'auto' avoids the hard crash.
    """
    return pycolmap.Device.auto


# ─────────────────────────────────────────────────────────────────────────────
# Read COLMAP binary depth map
# ─────────────────────────────────────────────────────────────────────────────

def _read_colmap_depthmap(path: Path) -> Optional[np.ndarray]:
    """Read a COLMAP binary .photometric.bin or .geometric.bin depth map.

    Format:
        width (int32) height (int32) channels (int32)
        data[height * width * channels] (float32)

    Returns (H, W) float32 array or None on failure.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            raw = f.read(12)
            if len(raw) < 12:
                return None
            w, h, c = struct.unpack("iii", raw)
            n = h * w * c
            data = np.frombuffer(f.read(n * 4), dtype=np.float32)
        if data.size != n:
            return None
        depth = data.reshape(h, w, c)[..., 0]   # take first channel
        return depth.astype(np.float32)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Core: single-pair SIFT-SfM + PatchMatch MVS
# ─────────────────────────────────────────────────────────────────────────────

def run_colmap_mvs_pair(
    img0_path,
    img1_path,
    gt_folder,
    K_GT: np.ndarray,
    sift_options: Optional[dict] = None,
    matching_options: Optional[dict] = None,
    min_inliers: int = 15,
    use_icp: bool = True,
    save_viz: bool = False,
    viz_out_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Dict:
    """Run SIFT + PatchMatch MVS + full metric evaluation for one pair.

    Parameters
    ----------
    img0_path, img1_path : paths to the two JPEG images.
    gt_folder            : path to folder with .npz + .exr GT files.
    K_GT                 : (3, 3) known camera intrinsics (shared).
    sift_options         : override dict for pycolmap.SiftExtractionOptions.
    matching_options     : override dict for pycolmap.SiftMatchingOptions.
    min_inliers          : minimum inlier count for pose to be valid.
    use_icp              : whether to refine Sim(3) alignment with ICP.
    save_viz             : whether to generate and save visualisations.
    viz_out_dir          : directory for visualisation output (required if save_viz).
    verbose              : print intermediate info.

    Returns
    -------
    Flat dict with all available metrics.  Pose metrics use *_colmap_mvs keys;
    depth / 3D metrics use the same avg_* namespace as MoonEvaluator.
    """
    img0_path = Path(img0_path)
    img1_path = Path(img1_path)
    gt_folder = Path(gt_folder)
    stem0 = img0_path.stem
    stem1 = img1_path.stem
    pair_key = f"{stem0}_{stem1}"

    result: Dict = {
        "pair": pair_key,
        "Model": "COLMAP-MVS",
        "Folder": gt_folder.name,
        "gt_focal": float(K_GT[0, 0]),
        # Pose
        "rra_colmap_mvs":        float("nan"),
        "rta_colmap_mvs":        float("nan"),
        "pose_error_colmap_mvs": float("nan"),
        "n_inliers_colmap_mvs":  0,
        # Depth coverage
        "depth_coverage_v0": float("nan"),
        "depth_coverage_v1": float("nan"),
    }

    # ── Canonical crop dimensions (must match MoonEvaluator) ─────────────────
    Hc, Wc = 384, 512

    # ── Load GT ───────────────────────────────────────────────────────────────
    try:
        from ..gt_loader import load_gt_view
        gt_pts0, depth_gt0, Kc0, T_wc0_gt, mask_g0 = load_gt_view(gt_folder, stem0, Hc, Wc)
        gt_pts1, depth_gt1, Kc1, T_wc1_gt, mask_g1 = load_gt_view(gt_folder, stem1, Hc, Wc)
    except Exception as e:
        result["error"] = f"GT load failed: {e}"
        return result

    gt_z_all = np.concatenate([
        depth_gt0[mask_g0.reshape(Hc, Wc)],
        depth_gt1[mask_g1.reshape(Hc, Wc)],
    ])
    gt_z_all = gt_z_all[np.isfinite(gt_z_all)]
    if len(gt_z_all) > 10:
        gt_median_depth = float(np.median(gt_z_all))
        gt_terrain_span = float(np.percentile(gt_z_all, 95) - np.percentile(gt_z_all, 5))
    else:
        gt_median_depth = 1.0
        gt_terrain_span = 1.0
    result["gt_median_depth"] = gt_median_depth
    result["gt_terrain_span"] = gt_terrain_span

    with tempfile.TemporaryDirectory(prefix="moon_colmap_mvs_") as tmpdir:
        tmpdir = Path(tmpdir)
        img_dir   = tmpdir / "images"
        sparse_dir = tmpdir / "sparse" / "0"
        dense_dir  = tmpdir / "dense"
        img_dir.mkdir(parents=True)
        sparse_dir.mkdir(parents=True)
        dense_dir.mkdir(parents=True)

        db_path = str(tmpdir / "colmap.db")

        shutil.copy2(img0_path, img_dir / img0_path.name)
        shutil.copy2(img1_path, img_dir / img1_path.name)

        # Silence pycolmap / glog output
        pycolmap.logging.minloglevel = 2
        pycolmap.logging.stderrthreshold = 2

        fx = float(K_GT[0, 0])
        cx = float(K_GT[0, 2])
        cy = float(K_GT[1, 2])
        camera_model = "SIMPLE_PINHOLE"

        # ── SIFT extraction ───────────────────────────────────────────────────
        sift_ext = pycolmap.SiftExtractionOptions()
        sift_ext.max_num_features = 8192
        sift_ext.num_threads = 4
        if sift_options:
            for k, v in sift_options.items():
                setattr(sift_ext, k, v)

        reader_opts = pycolmap.ImageReaderOptions()
        reader_opts.camera_model = camera_model
        reader_opts.camera_params = f"{fx},{cx},{cy}"

        try:
            pycolmap.extract_features(
                database_path=db_path,
                image_path=str(img_dir),
                image_names=[img0_path.name, img1_path.name],
                camera_mode=pycolmap.CameraMode.SINGLE,
                camera_model=camera_model,
                reader_options=reader_opts,
                sift_options=sift_ext,
                device=_pycolmap_device(),
            )
        except Exception as e:
            result["error"] = f"extract_features failed: {e}"
            return result

        # ── Exhaustive matching ───────────────────────────────────────────────
        sift_match = pycolmap.SiftMatchingOptions()
        sift_match.num_threads = 4
        if matching_options:
            for k, v in matching_options.items():
                setattr(sift_match, k, v)

        verif_opts = pycolmap.TwoViewGeometryOptions()
        verif_opts.min_num_inliers = min_inliers

        try:
            pycolmap.match_exhaustive(
                database_path=db_path,
                sift_options=sift_match,
                verification_options=verif_opts,
                device=_pycolmap_device(),
            )
        except Exception as e:
            result["error"] = f"match_exhaustive failed: {e}"
            return result

        # ── Read DB: keypoints, inlier matches, camera, pose ─────────────────
        try:
            db = pycolmap.Database(db_path)
            images_db = db.read_all_images()
            img_list = images_db.values() if hasattr(images_db, "values") else images_db
            name_to_id = {img.name: img.image_id for img in img_list}
            img_map    = {img.name: img for img in img_list}

            id0 = name_to_id.get(img0_path.name)
            id1 = name_to_id.get(img1_path.name)
            if id0 is None or id1 is None:
                result["error"] = "Image IDs not found in DB"
                return result

            kp0 = db.read_keypoints(id0)
            kp1 = db.read_keypoints(id1)
            result["n_keypoints_colmap_mvs"] = int(kp0.shape[0]) + int(kp1.shape[0])

            tvg = db.read_two_view_geometry(id0, id1)
            if len(tvg.inlier_matches) < min_inliers:
                result["error"] = f"Too few inliers ({len(tvg.inlier_matches)})"
                return result
            result["n_inliers_colmap_mvs"] = int(len(tvg.inlier_matches))

            cams_db = db.read_all_cameras()
            cam_list = cams_db.values() if hasattr(cams_db, "values") else cams_db
            cam_map  = {c.camera_id: c for c in cam_list}
            cam_img0 = cam_map[img_map[img0_path.name].camera_id]
            cam_img1 = cam_map[img_map[img1_path.name].camera_id]

            inlier_idx = np.array(tvg.inlier_matches)
            pts0_inl = kp0[inlier_idx[:, 0], :2].astype(np.float64)
            pts1_inl = kp1[inlier_idx[:, 1], :2].astype(np.float64)

            ok = pycolmap.estimate_two_view_geometry_pose(
                cam_img0, pts0_inl, cam_img1, pts1_inl, tvg
            )
            if not ok:
                result["error"] = "estimate_two_view_geometry_pose failed"
                return result

            cam2_from_cam1 = tvg.cam2_from_cam1
            R_est = np.array(cam2_from_cam1.rotation.matrix())
            t_est = np.array(cam2_from_cam1.translation)

        except Exception as e:
            result["error"] = f"DB read / pose extraction failed: {e}"
            return result

        # ── Camera pose metrics ───────────────────────────────────────────────
        try:
            R_gt, t_gt = _gt_relative_pose(gt_folder, stem0, stem1)
            rra = _rra_deg(R_est, R_gt)
            rta = _rta_deg(t_est, t_gt)
            result["rra_colmap_mvs"]        = float(rra)
            result["rta_colmap_mvs"]        = float(rta)
            result["pose_error_colmap_mvs"] = float(max(rra, rta))
        except Exception as e:
            if verbose:
                print(f"  [COLMAP-MVS] GT pose comparison failed: {e}")

        # ── Incremental reconstruction (triangulation) ────────────────────────
        # We write the two-view geometry as an initial sparse model so that
        # COLMAP's dense step knows where the cameras are.
        try:
            recon = _build_two_view_reconstruction(
                tmpdir, img0_path, img1_path, cam_img0, R_est, t_est,
                kp0, kp1, inlier_idx, sparse_dir, verbose=verbose
            )
        except Exception as e:
            if verbose:
                print(f"  [COLMAP-MVS] triangulation failed: {e}")
            # Return early with pose metrics only
            return result

        # ── Dense depth: PatchMatch CUDA (preferred) → SGBM → sparse interpolation ──
        dense_depth0, dense_depth1 = None, None

        # 1. Try PatchMatch first (requires CUDA pycolmap)
        try:
            dense_depth0, dense_depth1 = _run_patch_match(
                tmpdir, sparse_dir, dense_dir, img_dir,
                img0_path, img1_path, verbose=verbose
            )
            if dense_depth0 is not None:
                result["depth_backend"] = "patchmatch"
        except Exception as e:
            if verbose:
                print(f"  [COLMAP-MVS] PatchMatch failed: {e}")

        # 2. Fallback: sparse 3D triangulation → depth by interpolation
        #    This always works (no GPU required) and gives ~2000+ pts per pair.
        if dense_depth0 is None or dense_depth1 is None:
            if verbose:
                print(f"  [COLMAP-MVS] Using sparse-SfM depth (triangulated keypoints)...")
            try:
                dense_depth0, dense_depth1 = _sparse_to_depth(
                    recon, K_GT, R_est, t_est, Hc, Wc, verbose=verbose
                )
                result["depth_backend"] = "sparse_sfm"
            except Exception as e:
                if verbose:
                    print(f"  [COLMAP-MVS] sparse depth failed: {e}")

        if dense_depth0 is not None and dense_depth1 is not None:
            from scipy.interpolate import griddata

            # Interpolate missing depth values for dense_depth0
            grid_x, grid_y = np.meshgrid(np.arange(Wc), np.arange(Hc))
            valid_points = np.isfinite(dense_depth0)
            points = np.column_stack(np.where(valid_points))
            values = dense_depth0[valid_points]
            dense_depth0 = griddata(points, values, (grid_y, grid_x), method='linear', fill_value=np.nan)

            # Interpolate missing depth values for dense_depth1
            valid_points = np.isfinite(dense_depth1)
            points = np.column_stack(np.where(valid_points))
            values = dense_depth1[valid_points]
            dense_depth1 = griddata(points, values, (grid_y, grid_x), method='linear', fill_value=np.nan)

            if verbose:
                print("  [COLMAP-MVS] Applied interpolation to sparse depth maps.")

        if dense_depth0 is None or dense_depth1 is None:
            result["depth_backend"] = "none"
            return result

        # Coverage of dense map (fraction of pixels with valid depth)
        result["depth_coverage_v0"] = float(np.isfinite(dense_depth0).mean())
        result["depth_coverage_v1"] = float(np.isfinite(dense_depth1).mean())

        if verbose:
            print(f"  [COLMAP-MVS] coverage v0={result['depth_coverage_v0']:.2%} "
                  f"v1={result['depth_coverage_v1']:.2%}")

        # ── Lift dense depth to world-frame 3D (using *estimated* pose) ───────
        # Camera 0 is the reference frame; camera 1 is cam0 @ R_est/t_est.
        # Build 4×4 poses: T_wc0 = I (reference), T_wc1 = inv(cam2_from_cam1)
        T_wc0_est = np.eye(4)
        T_c1_from_c0 = np.eye(4)
        T_c1_from_c0[:3, :3] = R_est
        T_c1_from_c0[:3, 3]  = t_est
        T_wc1_est = np.linalg.inv(T_c1_from_c0)

        pts3d_v0_est, mask_v0_est = _backproject(dense_depth0, K_GT, T_wc0_est, Hc, Wc)
        pts3d_v1_est, mask_v1_est = _backproject(dense_depth1, K_GT, T_wc1_est, Hc, Wc)

        # ── Align predicted cloud to GT with Sim(3) + optional ICP ───────────
        mask_ok0 = mask_v0_est & mask_g0
        mask_ok1 = mask_v1_est & mask_g1

        n_valid = mask_ok0.sum() + mask_ok1.sum()
        if n_valid < 100:
            result["error"] = f"Too few valid overlapping pts ({n_valid})"
            return result

        pred_combined = np.vstack([pts3d_v0_est[mask_ok0], pts3d_v1_est[mask_ok1]])
        gt_combined   = np.vstack([gt_pts0[mask_ok0], gt_pts1[mask_ok1]])

        try:
            if use_icp:
                from ..alignment import improved_gt_alignment, apply_sim3
                T_sim3, scale, align_method = improved_gt_alignment(
                    pred_combined, gt_combined, verbose=verbose
                )
            else:
                from ..alignment import align_sim3_ransac, apply_sim3
                T_sim3, scale, _ = align_sim3_ransac(pred_combined, gt_combined, Nsub=8000)
                align_method = "ransac_only"
        except Exception as e:
            result["error"] = f"Sim(3) alignment failed: {e}"
            return result

        result["scene_scale"]         = float(scale)
        result["scene_scale_err_pct"] = float(abs(scale - 1.0) * 100)
        result["alignment_method"]    = align_method
        result["scene_n_valid"]       = int(n_valid)

        from ..alignment import apply_sim3
        aligned_v0 = apply_sim3(pts3d_v0_est, T_sim3)
        aligned_v1 = apply_sim3(pts3d_v1_est, T_sim3)

        # ── Save aligned PLY point clouds ─────────────────────────────────────
        if save_viz and viz_out_dir is not None:
            try:
                import open3d as o3d
                ply_dir = Path(viz_out_dir)
                ply_dir.mkdir(parents=True, exist_ok=True)

                combined_pred = np.vstack([aligned_v0, aligned_v1])
                combined_gt   = np.vstack([gt_pts0, gt_pts1])

                pred_ok = np.isfinite(combined_pred).all(axis=1)
                gt_ok   = np.isfinite(combined_gt).all(axis=1)

                pcd_pred = o3d.geometry.PointCloud(
                    o3d.utility.Vector3dVector(combined_pred[pred_ok])
                )
                pcd_gt = o3d.geometry.PointCloud(
                    o3d.utility.Vector3dVector(combined_gt[gt_ok])
                )
                o3d.io.write_point_cloud(str(ply_dir / "aligned_pred.ply"), pcd_pred)
                o3d.io.write_point_cloud(str(ply_dir / "aligned_gt.ply"),   pcd_gt)
                np.savetxt(str(ply_dir / "transform_sim3.txt"), T_sim3)
                if verbose:
                    print(f"  [COLMAP-MVS] PLYs saved to {ply_dir}")
            except Exception as e:
                if verbose:
                    print(f"  [COLMAP-MVS] PLY save failed: {e}")

        # ── Per-view metrics (same logic as MoonEvaluator) ────────────────────
        from ..metrics.classic import (
            compute_accuracy_completeness, compute_depth_metrics,
            compute_3d_metrics, compute_profile_metrics,
        )
        from ..metrics.terrain import (
            compute_slope_map, compute_slope_metrics, compute_hda_metrics,
            compute_curvature_maps, compute_roughness_map, compute_relief_metrics,
        )

        _viz_data: List[dict] = []

        view_pairs = [
            (0, aligned_v0, gt_pts0, depth_gt0, mask_ok0),
            (1, aligned_v1, gt_pts1, depth_gt1, mask_ok1),
        ]

        for vi, aligned, gt_pts, depth_gt, mask_ok in view_pairs:
            prefix = f"v{vi}"
            result[f"{prefix}_n_pts"] = int(mask_ok.sum())

            depth_pred_map = aligned.reshape(Hc, Wc, 3)[..., 2]
            depth_gt_map   = gt_pts.reshape(Hc, Wc, 3)[..., 2]
            mask_2d        = mask_ok.reshape(Hc, Wc)

            from scipy.ndimage import binary_erosion
            _struct = np.ones((9, 9), bool)
            mask_eroded = binary_erosion(mask_2d, structure=_struct, border_value=0)
            if mask_eroded.sum() < 50:
                mask_eroded = mask_2d

            # 3D metrics
            m3d = compute_3d_metrics(aligned, gt_pts, mask_ok)
            result.update({f"{prefix}_{k}": v for k, v in m3d.items()})
            if gt_median_depth > 0:
                for k in ("rmse", "mae_3d"):
                    if f"{prefix}_{k}" in result:
                        result[f"{prefix}_{k}_absrel"] = result[f"{prefix}_{k}"] / gt_median_depth

            # Accuracy / completeness
            if mask_ok.sum() >= 10:
                ac = compute_accuracy_completeness(
                    aligned[mask_ok], gt_pts[mask_ok], max_pts=20000
                )
                result.update({f"{prefix}_{k}": v for k, v in ac.items()})
                if gt_median_depth > 0:
                    for k in ("accuracy", "completeness", "chamfer", "acc_median", "compl_median"):
                        if f"{prefix}_{k}" in result:
                            result[f"{prefix}_{k}_absrel"] = result[f"{prefix}_{k}"] / gt_median_depth

            # Depth map metrics
            dm = compute_depth_metrics(depth_pred_map, depth_gt_map, mask_2d)
            result.update({f"{prefix}_{k}": v for k, v in dm.items()})
            if gt_median_depth > 0:
                for k in ("depth_mae", "depth_rmse"):
                    if f"{prefix}_{k}" in result:
                        result[f"{prefix}_{k}_absrel"] = result[f"{prefix}_{k}"] / gt_median_depth

            # Profile
            pm = compute_profile_metrics(depth_pred_map, depth_gt_map, mask_2d, n_rows=1)
            result.update({f"{prefix}_{k}": v for k, v in pm.items()})

            # Slope / HDA
            slope_gt_map, aspect_gt_map   = compute_slope_map(depth_gt_map)
            slope_pred_map, aspect_pred_map = compute_slope_map(depth_pred_map)
            result.update({f"{prefix}_{k}": v
                            for k, v in compute_slope_metrics(slope_pred_map, slope_gt_map, mask_eroded).items()})
            result.update({f"{prefix}_{k}": v
                            for k, v in compute_hda_metrics(slope_pred_map, slope_gt_map, mask_eroded).items()})

            # Curvature / roughness / relief
            try:
                curv_pred_map, _ = compute_curvature_maps(depth_pred_map)
                curv_gt_map, _   = compute_curvature_maps(depth_gt_map)
            except Exception:
                curv_pred_map = np.zeros_like(depth_pred_map)
                curv_gt_map   = np.zeros_like(depth_gt_map)

            try:
                rough_pred_map = compute_roughness_map(depth_pred_map)
                rough_gt_map   = compute_roughness_map(depth_gt_map)
            except Exception:
                rough_pred_map = np.zeros_like(depth_pred_map)
                rough_gt_map   = np.zeros_like(depth_gt_map)

            try:
                rm = compute_relief_metrics(depth_pred_map, depth_gt_map, mask_eroded)
                result.update({f"{prefix}_{k}": v for k, v in rm.items()})
            except Exception as e:
                if verbose:
                    print(f"  [COLMAP-MVS] [{prefix}] relief_metrics failed: {e}")

            _viz_data.append({
                "depth_pred": depth_pred_map,
                "depth_gt":   depth_gt_map,
                "mask":       mask_2d,
                "slope_pred": slope_pred_map,
                "slope_gt":   slope_gt_map,
                "aspect_pred": aspect_pred_map,
                "aspect_gt":   aspect_gt_map,
                "roughness_pred": rough_pred_map,
                "roughness_gt":   rough_gt_map,
                "curvature_pred": curv_pred_map,
                "curvature_gt":   curv_gt_map,
                "metrics": {
                    k.replace(f"{prefix}_", ""): result[k]
                    for k in result if k.startswith(f"{prefix}_")
                },
            })

        # ── Average per-view metrics → avg_* ──────────────────────────────────
        _avg_keys = [
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
            "slope_agree_5deg",  "slope_miss_5deg",  "slope_false_alarm_5deg",
            "slope_agree_10deg", "slope_miss_10deg", "slope_false_alarm_10deg",
            "slope_agree_15deg", "slope_miss_15deg", "slope_false_alarm_15deg",
            "slope_agree_20deg", "slope_miss_20deg", "slope_false_alarm_20deg",
            "roughness_corr", "roughness_mae",
            "curvature_corr", "curvature_mae",
            "crater_rim_iou", "crater_interior_iou", "crater_combined_iou",
            "relief_corr", "slope_ssim", "aspect_hist_corr",
            "peak_location_recall", "valley_location_recall",
        ]
        for k in _avg_keys:
            v0 = result.get(f"v0_{k}")
            v1 = result.get(f"v1_{k}")
            if v0 is not None and v1 is not None:
                try:
                    result[f"avg_{k}"] = float((float(v0) + float(v1)) / 2)
                except (TypeError, ValueError):
                    pass

        # ── Scene-level combined cloud metrics ────────────────────────────────
        try:
            from ..metrics.classic import compute_accuracy_completeness as _cac
            from scipy.spatial import cKDTree as _KD
            pred_all = np.vstack([aligned_v0[mask_ok0], aligned_v1[mask_ok1]])
            gt_all   = np.vstack([gt_pts0[mask_ok0], gt_pts1[mask_ok1]])
            if len(pred_all) > 0:
                scene_ac = _cac(pred_all, gt_all, max_pts=40000)
                for k, v in scene_ac.items():
                    result[f"scene_{k}"] = v
                rng = np.random.RandomState(0)
                p_sub = pred_all[rng.choice(len(pred_all), min(len(pred_all), 40000), replace=False)]
                g_sub = gt_all[rng.choice(len(gt_all), min(len(gt_all), 40000), replace=False)]
                d_p2g, _ = _KD(g_sub).query(p_sub)
                d_g2p, _ = _KD(p_sub).query(g_sub)
                result["scene_hausdorff_p95"] = float(max(
                    np.percentile(d_p2g, 95), np.percentile(d_g2p, 95)
                ))
                result["scene_hausdorff_max"] = float(max(d_p2g.max(), d_g2p.max()))
                if gt_median_depth > 0:
                    for k in ("accuracy", "completeness", "chamfer",
                              "acc_median", "compl_median",
                              "hausdorff_p95", "hausdorff_max"):
                        sk = f"scene_{k}"
                        if sk in result:
                            result[f"{sk}_absrel"] = result[sk] / gt_median_depth
        except Exception as e:
            if verbose:
                print(f"  [COLMAP-MVS] scene metrics failed: {e}")

        # ── Visualisations ────────────────────────────────────────────────────
        if save_viz and len(_viz_data) >= 2 and viz_out_dir is not None:
            try:
                from ..visualizer import visualize_pair
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
                    out_dir=Path(viz_out_dir),
                )
                if verbose:
                    print(f"  [COLMAP-MVS] Visualisations saved to {viz_out_dir}")
            except Exception as e:
                if verbose:
                    print(f"  [COLMAP-MVS] Visualisation failed: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Internal: write a two-view pycolmap reconstruction
# ─────────────────────────────────────────────────────────────────────────────

def _build_two_view_reconstruction(
    tmpdir: Path,
    img0_path: Path,
    img1_path: Path,
    cam: "pycolmap.Camera",
    R_est: np.ndarray,
    t_est: np.ndarray,
    kp0: np.ndarray,
    kp1: np.ndarray,
    inlier_idx: np.ndarray,
    sparse_dir: Path,
    verbose: bool = False,
) -> "pycolmap.Reconstruction":
    """Build a pycolmap Reconstruction with two registered images.

    Camera 0 is fixed at the origin; camera 1 is placed at R_est / t_est.
    Inlier keypoint pairs are triangulated to populate the sparse map.

    pycolmap >= 3.10 architecture:
        Rig  ← cameras that form a rigid group
        Frame ← a captured instant; holds one Rig and its pose
        Image ← one sensor's data attached to a Frame

    Returns the pycolmap.Reconstruction object.
    """
    recon = pycolmap.Reconstruction()

    # ── Camera (SIMPLE_PINHOLE, shared by both images) ────────────────────────
    recon.add_camera(cam)
    cam_id = cam.camera_id

    # ── Sensor descriptor for the camera ─────────────────────────────────────
    sensor = pycolmap.sensor_t()
    sensor.type = pycolmap.SensorType.CAMERA
    sensor.id = cam_id

    # ── Rig (single camera) ───────────────────────────────────────────────────
    rig = pycolmap.Rig()
    rig.rig_id = 1
    rig.add_ref_sensor(sensor)
    recon.add_rig(rig)

    # ── Helper: add a frame + image with a given pose ─────────────────────────
    def _add_registered_image(image_id, frame_id, name, pose_rigid3d):
        """Add Frame + Image pair, register image."""
        data = pycolmap.data_t()
        data.sensor_id = sensor
        data.id = image_id

        frame = pycolmap.Frame()
        frame.frame_id = frame_id
        frame.rig_id = 1
        frame.add_data_id(data)
        recon.add_frame(frame)
        # rig_ptr is wired after add_frame — set pose now
        recon.frame(frame_id).set_cam_from_world(cam_id, pose_rigid3d)

        img = pycolmap.Image()
        img.image_id = image_id
        img.name = name
        img.camera_id = cam_id
        img.frame_id = frame_id
        recon.add_image(img)
        recon.register_image(image_id)

    _add_registered_image(1, 1, img0_path.name, pycolmap.Rigid3d())
    _add_registered_image(
        2, 2, img1_path.name,
        pycolmap.Rigid3d(rotation=pycolmap.Rotation3d(R_est), translation=t_est),
    )

    if verbose:
        print(f"  [COLMAP-MVS] registered {recon.num_reg_images()} images")

    # ── Triangulate inlier correspondences ────────────────────────────────────
    pts0 = kp0[inlier_idx[:, 0], :2].astype(np.float64)
    pts1 = kp1[inlier_idx[:, 1], :2].astype(np.float64)

    pose0 = recon.frame(1).sensor_from_world(sensor)
    pose1 = recon.frame(2).sensor_from_world(sensor)

    # pycolmap.triangulate_point takes (3,4) projection matrices + (2,1) image points.
    # Keypoints are in pixel coordinates, so we must use K @ [R|t] projection matrices.
    K_arr = np.array(cam.calibration_matrix())  # (3,3) from pycolmap camera
    P0 = K_arr @ pose0.matrix()   # (3,4) full projection: pixel coords
    P1 = K_arr @ pose1.matrix()

    n_ok = 0
    for i in range(len(pts0)):
        try:
            p3d = pycolmap.triangulate_point(
                cam1_from_world=P0,
                cam2_from_world=P1,
                cam_point1=pts0[i].reshape(2, 1),
                cam_point2=pts1[i].reshape(2, 1),
            )
            if p3d is None:
                continue
            recon.add_point3D(
                xyz=p3d.reshape(3),
                track=pycolmap.Track(),
                color=np.array([128, 128, 128], dtype=np.uint8),
            )
            n_ok += 1
        except Exception:
            continue

    if verbose:
        print(f"  [COLMAP-MVS] triangulated {n_ok}/{len(pts0)} points")

    # Write to disk so PatchMatch can load it
    recon.write(str(sparse_dir))
    return recon


# ─────────────────────────────────────────────────────────────────────────────
# Internal: sparse SfM triangulation → interpolated depth maps
# ─────────────────────────────────────────────────────────────────────────────

def _sparse_to_depth(
    recon: "pycolmap.Reconstruction",
    K: np.ndarray,
    R_est: np.ndarray,
    t_est: np.ndarray,
    Hc: int,
    Wc: int,
    verbose: bool = False,
) -> tuple:
    """Convert triangulated sparse 3D points to per-view depth maps.

    Projects the sparse point cloud onto each camera, then fills the depth map
    by Delaunay-interpolation (linear triangulation of visible points).

    Camera 0 is the reference frame (identity pose).
    Camera 1 pose: cam1_from_world = [R_est | t_est] (unit baseline from E-matrix).

    Returns (depth0, depth1) as (Hc, Wc) float32 arrays, or (None, None) if
    fewer than 50 points are available.
    """
    # Collect all 3D points from the reconstruction (in cam0 frame)
    pts3d = []
    for p3d in recon.points3D.values():
        pts3d.append(p3d.xyz)
    if len(pts3d) < 50:
        return None, None
    pts3d = np.array(pts3d, dtype=np.float64)   # (N, 3), in cam0 frame

    def _project_and_interpolate(pts_cam, K_arr):
        """Project N points in camera frame → interpolated (Hc, Wc) depth map."""
        z = pts_cam[:, 2]
        keep = z > 0
        pts_f = pts_cam[keep]
        z_f   = z[keep]
        if len(z_f) < 10:
            return None

        u = pts_f[:, 0] / z_f * K_arr[0, 0] + K_arr[0, 2]
        v = pts_f[:, 1] / z_f * K_arr[1, 1] + K_arr[1, 2]

        # Keep only points that project inside the image (with margin)
        margin = 5
        inside = (u >= margin) & (u < Wc - margin) & (v >= margin) & (v < Hc - margin)
        u = u[inside]; v = v[inside]; z_f = z_f[inside]
        if len(z_f) < 10:
            return None

        # Scatter sparse depth values directly onto the pixel grid — no
        # interpolation. Each keypoint writes to its nearest pixel; all other
        # pixels stay NaN. This avoids Delaunay triangle artefacts.
        depth = np.full((Hc, Wc), np.nan, dtype=np.float32)
        ui = np.clip(np.round(u).astype(int), 0, Wc - 1)
        vi = np.clip(np.round(v).astype(int), 0, Hc - 1)
        depth[vi, ui] = z_f.astype(np.float32)
        return depth

    # ── Camera 0: points already in cam0 frame ────────────────────────────────
    depth0 = _project_and_interpolate(pts3d, K)

    # ── Camera 1: rotate+translate pts3d into cam1 frame ─────────────────────
    # cam1_from_world: [R_est | t_est]  (unit baseline from E-matrix)
    pts_cam1 = (R_est @ pts3d.T).T + t_est[np.newaxis, :]
    depth1 = _project_and_interpolate(pts_cam1, K)

    if verbose:
        for vi, d in enumerate([depth0, depth1]):
            if d is not None:
                v = np.isfinite(d) & (d > 0)
                print(f"  [COLMAP-MVS/sparse] depth{vi} valid={v.mean():.1%} "
                      f"range=[{d[v].min():.2f},{d[v].max():.2f}]")
            else:
                print(f"  [COLMAP-MVS/sparse] depth{vi}: too few points")

    if depth0 is None or depth1 is None:
        return None, None

    # Require at least 10% coverage
    v0 = np.isfinite(depth0) & (depth0 > 0)
    v1 = np.isfinite(depth1) & (depth1 > 0)
    # With sparse (non-interpolated) points, coverage is naturally low (~0.5–2%).
    # Accept as long as we have at least 50 valid pixels per view.
    if v0.sum() < 50 or v1.sum() < 50:
        if verbose:
            print(f"  [COLMAP-MVS/sparse] too few valid pixels: "
                  f"v0={v0.sum()} v1={v1.sum()}")
        return None, None

    return depth0, depth1


# ─────────────────────────────────────────────────────────────────────────────
# Internal: OpenCV StereoSGBM fallback (CPU, no CUDA required)
# ─────────────────────────────────────────────────────────────────────────────

def _run_sgbm(
    img0_path: Path,
    img1_path: Path,
    K: np.ndarray,
    R_est: np.ndarray,
    t_est: np.ndarray,
    Hc: int = 384,
    Wc: int = 512,
    verbose: bool = False,
) -> tuple:
    """Dense depth via OpenCV StereoSGBM + rectification.

    Rectifies the two images using R/t, runs StereoSGBM to get disparity maps,
    then converts disparity → 3D points and re-projects onto the original
    (unrectified) camera grids.

    Returns (depth0, depth1) as (Hc, Wc) float32 arrays in the *original*
    (unrectified) camera frame, so they can be used with the GT intrinsics K.
    Depth scale is proportional to ||t_est|| (corrected by Sim(3) downstream).
    """
    import cv2

    # ── Load and centre-crop images ───────────────────────────────────────────
    def _load_crop(p):
        img = cv2.imread(str(p))
        H0, W0 = img.shape[:2]
        y0 = (H0 - Hc) // 2
        x0 = (W0 - Wc) // 2
        return img[y0:y0 + Hc, x0:x0 + Wc]

    img0 = _load_crop(img0_path)
    img1 = _load_crop(img1_path)

    # ── Stereo rectification ──────────────────────────────────────────────────
    # Normalize t so baseline = 1; scale recovered by Sim(3) alignment.
    t_norm = t_est / (np.linalg.norm(t_est) + 1e-15)

    dist = np.zeros(5, np.float64)
    R1, R2, P1, P2, Q, roi0, roi1 = cv2.stereoRectify(
        K, dist, K, dist,
        (Wc, Hc),
        R_est, t_norm.reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )

    map0x, map0y = cv2.initUndistortRectifyMap(K, dist, R1, P1, (Wc, Hc), cv2.CV_32FC1)
    map1x, map1y = cv2.initUndistortRectifyMap(K, dist, R2, P2, (Wc, Hc), cv2.CV_32FC1)

    rect0 = cv2.remap(img0, map0x, map0y, cv2.INTER_LINEAR)
    rect1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)

    if verbose:
        nz0 = float((rect0.sum(-1) > 0).mean())
        nz1 = float((rect1.sum(-1) > 0).mean())
        print(f"  [COLMAP-MVS/SGBM] rect coverage: r0={nz0:.1%} r1={nz1:.1%}")

    # ── Determine disparity sign based on baseline direction ──────────────────
    # P2[0,3] = -f * baseline  (camera 1 to the right → P2[0,3] < 0 → positive disp)
    # P2[0,3] = +f * baseline  (camera 1 to the left  → P2[0,3] > 0 → negative disp)
    f_rect   = float(P1[0, 0])
    cx_rect  = float(P1[0, 2])
    cy_rect  = float(P1[1, 2])
    p2_tx    = float(P2[0, 3])
    baseline = abs(p2_tx) / (f_rect + 1e-15)  # always positive

    ndisp = 256  # must be divisible by 16
    win   = 5
    if p2_tx > 0:
        # Camera 1 is to the LEFT: negative disparities in range [-ndisp, 0]
        min_d = -ndisp
    else:
        # Camera 1 is to the RIGHT: positive disparities in range [0, ndisp]
        min_d = 0

    if verbose:
        print(f"  [COLMAP-MVS/SGBM] f_rect={f_rect:.1f} baseline={baseline:.4f} "
              f"P2_tx={p2_tx:.1f} min_disp={min_d}")

    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_d,
        numDisparities=ndisp,
        blockSize=win,
        P1=8  * 3 * win ** 2,
        P2=32 * 3 * win ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=5,
        speckleWindowSize=50,
        speckleRange=16,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp0_raw = sgbm.compute(rect0, rect1).astype(np.float32) / 16.0

    # Valid disparity: exclude the invalid sentinel (minDisp - 1) and the far end
    # With minDisp=0:   invalid = -1,       valid > 0
    # With minDisp=-N:  invalid = -(N+1),   valid in (-N, 0)
    if min_d == 0:
        valid0 = disp0_raw > 1.0
    else:  # min_d = -ndisp
        valid0 = (disp0_raw > (min_d + 1)) & (disp0_raw < -1.0)

    if verbose:
        print(f"  [COLMAP-MVS/SGBM] disp0 valid={valid0.mean():.1%} "
              f"range=[{disp0_raw.min():.1f},{disp0_raw.max():.1f}]")

    # ── Also run the reversed pair for camera 1 ───────────────────────────────
    # For camera 1: swap the images so camera 1 is on the left, get positive disp
    if p2_tx > 0:
        # Camera 1 was to the left. For camera 1 as "left" camera, camera 0 is to the right.
        # Use the same stereo setup but swap images, min_d=0 for positive disparity
        sgbm_r = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=ndisp, blockSize=win,
            P1=8*3*win**2, P2=32*3*win**2, disp12MaxDiff=1, uniquenessRatio=5,
            speckleWindowSize=50, speckleRange=16, preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        disp1_rev = sgbm_r.compute(rect1, rect0).astype(np.float32) / 16.0
        valid1 = disp1_rev > 1.0
        # Depth for camera 1 in the reversed stereo: depth = f_rect * baseline / disp
        disp_for_depth1 = disp1_rev
    else:
        # Camera 1 to the right: use right-matcher
        sgbm_r2 = cv2.ximgproc.createRightMatcher(sgbm) if hasattr(cv2, 'ximgproc') else None
        if sgbm_r2 is not None:
            disp1_rev_raw = sgbm_r2.compute(rect1, rect0).astype(np.float32) / 16.0
            disp_for_depth1 = -disp1_rev_raw  # flip sign: valid → positive
            disp_for_depth1[disp_for_depth1 >= ndisp] = 0.0
        else:
            disp_for_depth1 = disp0_raw.copy()
        valid1 = disp_for_depth1 > 1.0

    if verbose:
        print(f"  [COLMAP-MVS/SGBM] disp1 valid={valid1.mean():.1%}")

    # ── Disparity → 3D points → original camera frame ─────────────────────────
    def _disp_to_depth_orig(disp_map, valid_mask, R_rect, sign=1.0):
        """Convert rectified disparity to depth in the original camera frame.

        sign=1 for positive-disparity (camera 1 to the right or reversed pair),
        sign=-1 for negative-disparity (camera 1 to the left).
        """
        rows, cols = np.where(valid_mask)
        if len(rows) == 0:
            return np.full((Hc, Wc), np.nan, dtype=np.float32)

        d_vals = disp_map[rows, cols]
        # depth = f_rect * baseline / |disparity|
        z_rect = f_rect * baseline / (sign * d_vals + 1e-15)
        z_rect = np.abs(z_rect)  # ensure positive

        x_rect = (cols - cx_rect) * z_rect / f_rect
        y_rect = (rows - cy_rect) * z_rect / f_rect
        pts_rect = np.stack([x_rect, y_rect, z_rect], axis=-1)  # (N, 3)

        # Rotate to original camera frame: R_rect maps orig→rect, so R_rect.T = inv
        pts_orig = pts_rect @ R_rect  # (N, 3) — right-multiply = apply R_rect.T

        z_o = pts_orig[:, 2]
        keep = z_o > 0
        pts_orig = pts_orig[keep]; z_o = z_o[keep]

        u_o = pts_orig[:, 0] / z_o * float(K[0, 0]) + float(K[0, 2])
        v_o = pts_orig[:, 1] / z_o * float(K[1, 1]) + float(K[1, 2])

        u_i = np.round(u_o).astype(int)
        v_i = np.round(v_o).astype(int)
        inside = (u_i >= 0) & (u_i < Wc) & (v_i >= 0) & (v_i < Hc)
        u_i = u_i[inside]; v_i = v_i[inside]; z_o = z_o[inside]

        depth_map = np.full((Hc, Wc), np.nan, dtype=np.float32)
        order = np.argsort(z_o)   # keep closest when multiple hit same pixel
        depth_map[v_i[order], u_i[order]] = z_o[order].astype(np.float32)
        return depth_map

    # For camera 0: disparity sign depends on which side camera 1 is on
    disp_sign0 = -1.0 if min_d < 0 else 1.0
    depth0 = _disp_to_depth_orig(disp0_raw, valid0, np.array(R1), sign=disp_sign0)
    # For camera 1 reversed: always positive disparity
    depth1 = _disp_to_depth_orig(disp_for_depth1, valid1, np.array(R2), sign=1.0)

    if verbose:
        for vi, d in enumerate([depth0, depth1]):
            v = np.isfinite(d) & (d > 0)
            if v.sum() > 0:
                print(f"  [COLMAP-MVS/SGBM] depth{vi} valid={v.mean():.1%} "
                      f"range=[{d[v].min():.2f},{d[v].max():.2f}]")
            else:
                print(f"  [COLMAP-MVS/SGBM] depth{vi} valid=0% (all invalid)")

    # Return None if coverage is too low to be useful (< 2%)
    v0 = np.isfinite(depth0) & (depth0 > 0)
    v1 = np.isfinite(depth1) & (depth1 > 0)
    if v0.mean() < 0.02 or v1.mean() < 0.02:
        return None, None

    return depth0, depth1


# ─────────────────────────────────────────────────────────────────────────────
# Internal: PatchMatch dense depth estimation
# ─────────────────────────────────────────────────────────────────────────────

def _run_patch_match(
    tmpdir: Path,
    sparse_dir: Path,
    dense_dir: Path,
    img_dir: Path,
    img0_path: Path,
    img1_path: Path,
    verbose: bool = False,
) -> tuple:
    """Run COLMAP image undistortion + PatchMatch stereo.

    Returns (depth0, depth1) as (H, W) float32 arrays, or (None, None) on failure.
    """
    # Undistort images (prepares dense MVS workspace)
    # pycolmap.undistort_images signature:
    #   (output_path, input_path, image_path, image_names=[], output_type='COLMAP',
    #    copy_policy=..., num_patch_match_src_images=20, undistort_options=...)
    uopts = pycolmap.UndistortCameraOptions()
    try:
        pycolmap.undistort_images(
            output_path=str(dense_dir),
            input_path=str(sparse_dir),
            image_path=str(img_dir),
            output_type="COLMAP",
            undistort_options=uopts,
        )
    except Exception as e:
        if verbose:
            print(f"  [COLMAP-MVS] undistort_images failed: {e}")
        return None, None

    # PatchMatch Stereo — requires CUDA; returns immediately on CPU-only builds
    pmvs_opts = pycolmap.PatchMatchOptions()
    # depth_min / depth_max: -1 = auto-estimated from sparse points
    pmvs_opts.geom_consistency = True

    try:
        pycolmap.patch_match_stereo(
            workspace_path=str(dense_dir),
            workspace_format="COLMAP",
            options=pmvs_opts,
        )
    except Exception as e:
        if verbose:
            print(f"  [COLMAP-MVS] patch_match_stereo failed: {e}")
        return None, None

    # Read depth maps — COLMAP writes them under dense/stereo/depth_maps/
    dm_dir = dense_dir / "stereo" / "depth_maps"
    # Try geometric (preferred) then photometric
    def _try_load(name):
        for suffix in (".geometric.bin", ".photometric.bin"):
            p = dm_dir / (name + suffix)
            d = _read_colmap_depthmap(p)
            if d is not None:
                return d
        return None

    depth0 = _try_load(img0_path.name)
    depth1 = _try_load(img1_path.name)

    if verbose:
        for vi, d in [(0, depth0), (1, depth1)]:
            if d is not None:
                valid = np.isfinite(d) & (d > 0)
                print(f"  [COLMAP-MVS] depth v{vi}: {d.shape} "
                      f"valid={valid.mean():.1%} "
                      f"range=[{d[valid].min():.1f}, {d[valid].max():.1f}]")
            else:
                print(f"  [COLMAP-MVS] depth v{vi}: not found")

    return depth0, depth1


# ─────────────────────────────────────────────────────────────────────────────
# Internal: back-project depth map to world-frame 3D
# ─────────────────────────────────────────────────────────────────────────────

def _backproject(
    depth_map: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
    Hc: int,
    Wc: int,
) -> tuple:
    """Back-project a depth map to world-frame 3D points.

    The depth map may have a different resolution than (Hc, Wc); we
    centre-crop (or pad) it to match.

    Returns:
        pts_world  : (Hc*Wc, 3)
        mask_valid : (Hc*Wc,)  bool
    """
    if depth_map is None:
        return np.zeros((Hc * Wc, 3), np.float32), np.zeros(Hc * Wc, bool)

    # Resize / centre-crop depth map to (Hc, Wc)
    H0, W0 = depth_map.shape
    if (H0, W0) != (Hc, Wc):
        y0 = max(0, (H0 - Hc) // 2)
        x0 = max(0, (W0 - Wc) // 2)
        depth_map = depth_map[y0:y0 + Hc, x0:x0 + Wc]
        # If still wrong size, resize with nearest-neighbour
        if depth_map.shape != (Hc, Wc):
            try:
                from skimage.transform import resize as _resize
                depth_map = _resize(
                    depth_map, (Hc, Wc),
                    order=0, mode="constant", cval=np.nan, anti_aliasing=False
                ).astype(np.float32)
            except Exception:
                depth_map = np.full((Hc, Wc), np.nan, np.float32)

    u, v = np.meshgrid(np.arange(Wc), np.arange(Hc))
    z = depth_map.copy()
    z[(~np.isfinite(z)) | (z <= 0)] = np.nan

    x_c = (u - K[0, 2]) * z / K[0, 0]
    y_c = (v - K[1, 2]) * z / K[1, 1]
    pts_cam = np.stack([x_c, y_c, z], axis=-1).reshape(-1, 3)

    hom = np.concatenate([pts_cam, np.ones((Hc * Wc, 1))], axis=1).T
    pts_world = (T_wc @ hom)[:3].T.astype(np.float32)
    mask_valid = np.isfinite(pts_world).all(axis=1)

    return pts_world, mask_valid


# ─────────────────────────────────────────────────────────────────────────────
# Batch evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_colmap_mvs_folder(
    gt_folder_path,
    K_GT: np.ndarray,
    folder_key: Optional[str] = None,
    output_root: Optional[Path] = None,
    max_pairs: Optional[int] = None,
    min_inliers: int = 15,
    use_icp: bool = True,
    save_viz: bool = False,
    verbose: bool = False,
) -> List[Dict]:
    """Run COLMAP MVS on all consecutive image pairs in a GT folder.

    Same pair ordering as MoonEvaluator.evaluate_folder().

    Returns list of result dicts compatible with moon_eval CSV format.
    """
    gt_folder = Path(gt_folder_path)
    if folder_key is None:
        folder_key = gt_folder.name

    images = sorted(gt_folder.glob("*.jpg"))
    pairs = list(zip(images[0::2], images[1::2]))

    if max_pairs is not None and max_pairs < len(pairs):
        step = max(1, len(pairs) // max_pairs)
        pairs = pairs[::step][:max_pairs]

    results = []
    for img0, img1 in pairs:
        print(f"  [COLMAP-MVS] {img0.name} vs {img1.name}")
        pair_key = f"{img0.stem}_{img1.stem}"

        viz_out = None
        if save_viz and output_root is not None:
            viz_out = Path(output_root) / "COLMAP-MVS" / folder_key / pair_key / "viz"

        try:
            row = run_colmap_mvs_pair(
                img0, img1, gt_folder, K_GT,
                min_inliers=min_inliers,
                use_icp=use_icp,
                save_viz=save_viz,
                viz_out_dir=viz_out,
                verbose=verbose,
            )
            row["Folder"] = folder_key
        except Exception as e:
            import traceback
            print(f"    FAILED: {e}")
            if verbose:
                traceback.print_exc()
            row = {
                "Model": "COLMAP-MVS",
                "Folder": folder_key,
                "pair": pair_key,
                "error": str(e),
            }
        results.append(row)

    return results
