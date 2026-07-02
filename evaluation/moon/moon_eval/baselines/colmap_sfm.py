"""
moon_eval/baselines/colmap_sfm.py — Classical SIFT-SfM baseline via pycolmap.

Pipeline for a single stereo pair:
  1. Copy the two images to a temporary directory.
  2. Extract SIFT keypoints + descriptors  (pycolmap.extract_features).
  3. Exhaustive matching                   (pycolmap.match_exhaustive).
  4. Read keypoints + inlier matches from the DB.
  5. Estimate calibrated two-view geometry (Essential matrix, RANSAC).
  6. Extract R, t from cam2_from_cam1.
  7. Compare to GT relative pose → RRA / RTA.

Only camera-pose metrics are produced.  Dense depth / 3D metrics are
not available from this pipeline and are reported as NaN.

Requirements:
  pip install pycolmap   (already in the mast3r conda env)
"""

import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import pycolmap


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rra_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """Relative Rotation Accuracy — angular error between two rotation matrices."""
    R_err = R_est @ R_gt.T
    trace = float(np.trace(R_err))
    cos_a = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _rta_deg(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    """Relative Translation Accuracy — angular error between translation directions."""
    n_est = t_est / (np.linalg.norm(t_est) + 1e-15)
    n_gt  = t_gt  / (np.linalg.norm(t_gt)  + 1e-15)
    cos_a = np.clip(float(np.dot(n_est, n_gt)), -1.0, 1.0)
    angle = float(np.degrees(np.arccos(np.abs(cos_a))))  # sign ambiguity
    return angle


def _gt_relative_pose(gt_folder: Path, stem0: str, stem1: str):
    """Load GT cam2world for both views and return GT relative pose R_gt, t_gt.

    T_rel = inv(T_wc1) @ T_wc0   (pose of cam0 in cam1 frame)
    """
    d0 = np.load(gt_folder / f"{stem0}.npz")
    d1 = np.load(gt_folder / f"{stem1}.npz")
    T_wc0 = d0["cam2world"].astype(np.float64)
    T_wc1 = d1["cam2world"].astype(np.float64)
    T_rel = np.linalg.inv(T_wc1) @ T_wc0
    R_gt = T_rel[:3, :3]
    t_gt = T_rel[:3, 3]
    return R_gt, t_gt


# ─────────────────────────────────────────────────────────────────────────────
# Core: single-pair SIFT-SfM
# ─────────────────────────────────────────────────────────────────────────────

def run_colmap_pair(
    img0_path,
    img1_path,
    gt_folder,
    K_GT: np.ndarray,
    sift_options: Optional[dict] = None,
    matching_options: Optional[dict] = None,
    min_inliers: int = 15,
    verbose: bool = False,
) -> Dict:
    """Run SIFT + exhaustive matching + Essential-matrix pose for one pair.

    Parameters
    ----------
    img0_path, img1_path : paths to the two JPEG images.
    gt_folder            : path to folder with .npz GT files.
    K_GT                 : (3, 3) known camera intrinsics (shared for both views).
    sift_options         : override dict for pycolmap.SiftExtractionOptions.
    matching_options     : override dict for pycolmap.SiftMatchingOptions.
    min_inliers          : minimum inlier count to consider pose valid.
    verbose              : print intermediate info.

    Returns
    -------
    Flat dict with keys:
        pair, rra_colmap, rta_colmap, pose_error_colmap,
        n_inliers_colmap, n_keypoints_colmap,
        Model='COLMAP-SIFT', Folder=<inferred from gt_folder.name>
    All dense/depth keys are absent (NaN in downstream DataFrame).
    """
    img0_path = Path(img0_path)
    img1_path = Path(img1_path)
    gt_folder = Path(gt_folder)
    stem0 = img0_path.stem
    stem1 = img1_path.stem
    pair_key = f"{stem0}_{stem1}"

    result: Dict = {
        "pair": pair_key,
        "Model": "COLMAP-SIFT",
        "Folder": gt_folder.name,
        "gt_focal": float(K_GT[0, 0]),
        # Pose metrics — filled below or left NaN on failure
        "rra_colmap": float("nan"),
        "rta_colmap": float("nan"),
        "pose_error_colmap": float("nan"),
        "n_inliers_colmap": 0,
        "n_keypoints_colmap": 0,
    }

    with tempfile.TemporaryDirectory(prefix="moon_colmap_") as tmpdir:
        tmpdir = Path(tmpdir)
        img_dir = tmpdir / "images"
        img_dir.mkdir()
        db_path = str(tmpdir / "colmap.db")

        # Copy images into tmpdir/images/ (pycolmap needs them there)
        shutil.copy2(img0_path, img_dir / img0_path.name)
        shutil.copy2(img1_path, img_dir / img1_path.name)

        # Silence pycolmap / glog INFO+WARNING output (very chatty by default).
        # minloglevel: 0=INFO, 1=WARNING, 2=ERROR, 3=FATAL
        pycolmap.logging.minloglevel = 2
        pycolmap.logging.stderrthreshold = 2

        # ── 1. Build SIFT extraction options ─────────────────────────────────
        sift_ext = pycolmap.SiftExtractionOptions()
        sift_ext.max_num_features = 8192
        sift_ext.num_threads = 4   # cap threads to avoid the "Creating SIFT..." spam
        if sift_options:
            for k, v in sift_options.items():
                setattr(sift_ext, k, v)

        # Use SIMPLE_PINHOLE: f, cx, cy.  No radial distortion (lunar imagery
        # from a calibrated sensor — K is fixed and known).
        # CameraMode.SINGLE tells pycolmap to use one camera entry for all images.
        fx = float(K_GT[0, 0])
        cx = float(K_GT[0, 2])
        cy = float(K_GT[1, 2])
        camera_model = "SIMPLE_PINHOLE"

        reader_opts = pycolmap.ImageReaderOptions()
        reader_opts.camera_model = camera_model
        # Comma-separated params string: f,cx,cy  (SIMPLE_PINHOLE order)
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
                device=pycolmap.Device.cuda if _has_cuda() else pycolmap.Device.cpu,
            )
        except Exception as e:
            if verbose:
                print(f"  [COLMAP] extract_features failed: {e}")
            result["error"] = str(e)
            return result

        # ── 2. Exhaustive matching ────────────────────────────────────────────
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
                device=pycolmap.Device.cuda if _has_cuda() else pycolmap.Device.cpu,
            )
        except Exception as e:
            if verbose:
                print(f"  [COLMAP] match_exhaustive failed: {e}")
            result["error"] = str(e)
            return result

        # ── 3. Read keypoints + inlier matches from DB ────────────────────────
        try:
            db = pycolmap.Database(db_path)
            images_db = db.read_all_images()
            # read_all_images() returns a list of Image objects in pycolmap ≥ 3.x
            # Map name → image_id
            img_list = images_db.values() if hasattr(images_db, "values") else images_db
            name_to_id = {img.name: img.image_id for img in img_list}

            id0 = name_to_id.get(img0_path.name)
            id1 = name_to_id.get(img1_path.name)
            if id0 is None or id1 is None:
                result["error"] = "Image IDs not found in DB"
                return result

            kp0 = db.read_keypoints(id0)   # (N, 2+) array of pixel coords
            kp1 = db.read_keypoints(id1)

            result["n_keypoints_colmap"] = int(kp0.shape[0]) + int(kp1.shape[0])

            # TwoViewGeometry stores inlier_matches (M, 2) index pairs
            tvg = db.read_two_view_geometry(id0, id1)

            if verbose:
                print(f"  [COLMAP] {pair_key}: kp0={kp0.shape[0]} kp1={kp1.shape[0]} "
                      f"inliers={len(tvg.inlier_matches)}")

            if len(tvg.inlier_matches) < min_inliers:
                result["error"] = f"Too few inliers ({len(tvg.inlier_matches)})"
                return result

            result["n_inliers_colmap"] = int(len(tvg.inlier_matches))

            # ── 4. Recover relative pose from the Essential matrix ────────────
            # match_exhaustive stores the Essential matrix (E) in the DB but
            # does not decompose it into R/t.  We call estimate_two_view_geometry_pose
            # which decomposes E using the chirality test and fills cam2_from_cam1.
            #
            # Read camera from DB (same for both images — CameraMode.SINGLE)
            cams_db = db.read_all_cameras()
            cam_list = cams_db.values() if hasattr(cams_db, "values") else cams_db
            cam_map = {c.camera_id: c for c in cam_list}

            # Retrieve the image objects to get their camera_id
            img_map = {img.name: img for img in img_list}
            cam_img0 = cam_map[img_map[img0_path.name].camera_id]
            cam_img1 = cam_map[img_map[img1_path.name].camera_id]

            # Use only the x,y pixel coords of the inlier matches
            inlier_idx = np.array(tvg.inlier_matches)  # (M, 2)
            pts0_inl = kp0[inlier_idx[:, 0], :2].astype(np.float64)
            pts1_inl = kp1[inlier_idx[:, 1], :2].astype(np.float64)

            # estimate_two_view_geometry_pose decomposes E → cam2_from_cam1 in tvg
            ok = pycolmap.estimate_two_view_geometry_pose(
                cam_img0, pts0_inl, cam_img1, pts1_inl, tvg
            )
            if not ok:
                result["error"] = "estimate_two_view_geometry_pose failed"
                return result

            cam2_from_cam1 = tvg.cam2_from_cam1
            R_est = np.array(cam2_from_cam1.rotation.matrix())   # (3, 3)
            t_est = np.array(cam2_from_cam1.translation)          # (3,) unit vector

            if verbose:
                print(f"  [COLMAP] R_est diag={np.diag(R_est)}, t_est={t_est}")

        except Exception as e:
            if verbose:
                print(f"  [COLMAP] DB read / pose extraction failed: {e}")
            result["error"] = str(e)
            return result

        # ── 5. Compare to GT pose ─────────────────────────────────────────────
        try:
            R_gt, t_gt = _gt_relative_pose(gt_folder, stem0, stem1)
            rra = _rra_deg(R_est, R_gt)
            rta = _rta_deg(t_est, t_gt)
            pose_err = max(rra, rta)
            result["rra_colmap"]        = float(rra)
            result["rta_colmap"]        = float(rta)
            result["pose_error_colmap"] = float(pose_err)
        except Exception as e:
            if verbose:
                print(f"  [COLMAP] GT pose comparison failed: {e}")
            result["error"] = str(e)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Batch evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_colmap_folder(
    gt_folder_path,
    K_GT: np.ndarray,
    folder_key: Optional[str] = None,
    max_pairs: Optional[int] = None,
    min_inliers: int = 15,
    verbose: bool = False,
) -> List[Dict]:
    """Run COLMAP-SIFT on all consecutive image pairs in a GT folder.

    Same pair ordering as MoonEvaluator.evaluate_folder(): sorted images
    paired as (even, odd) indices.

    Returns list of result dicts (compatible with moon_eval CSV format).
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
        print(f"  [COLMAP-SIFT] {img0.name} vs {img1.name}")
        try:
            row = run_colmap_pair(
                img0, img1, gt_folder, K_GT,
                min_inliers=min_inliers,
                verbose=verbose,
            )
        except Exception as e:
            import traceback
            print(f"    FAILED: {e}")
            if verbose:
                traceback.print_exc()
            row = {
                "Model": "COLMAP-SIFT",
                "Folder": folder_key,
                "pair": f"{img0.stem}_{img1.stem}",
                "error": str(e),
            }
        results.append(row)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
