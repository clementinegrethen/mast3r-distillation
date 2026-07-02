#!/usr/bin/env python3
"""
MASt3REvaluator.py - CORRECTED VERSION WITH ALL FEATURES

Seulement les corrections de formatage - TOUTES les métriques et visualisations conservées
"""
import torch
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
from dust3r.utils.device import to_numpy
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import sobel, uniform_filter
from matplotlib.colors import LightSource
import seaborn as sns
import OpenEXR, Imath
import warnings
from datetime import datetime
from Mast3rtiny import load_tiny_mast3r_for_inference

warnings.simplefilter(action='ignore', category=FutureWarning)

# Configuration des visualisations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class MASt3REvaluator:
    """
    Classe complète pour l'évaluation de MASt3R avec toutes les métriques et visualisations.
    """
    
    def __init__(self, gt_folder, name, img1, img2, device, ckpt_path, output_dir=None, model=None, save_viz=True, use_sparse_ga=False, skip_stitch=False, disable_icp=False):
        self.gt_folder = Path(gt_folder)
        self.name = name
        self.img1 = Path(img1)
        self.img2 = Path(img2)
        self.device = device
        self.ckpt_path = Path(ckpt_path)
        self.model = model  # pre-loaded model (skip loading in run_mast3r_inference)
        self.save_viz = save_viz  # whether to generate and save visualizations
        self.use_sparse_ga = use_sparse_ga  # use sparse_global_alignment for scene reconstruction
        self.skip_stitch = skip_stitch
        self.disable_icp = disable_icp
        
        # Créer dossier de sortie
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(f"evaluation_results_{timestamp}")
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Variables pour stocker les résultats
        self.metrics = {}
        self.terrain_metrics = {}
        self.profile_metrics = {}
        self.error_stats = {}
        
        print(f"MASt3R Evaluator initialized")
        print(f"Output directory: {self.output_dir}")

    def _fmt(self, val, fmt=".3f"):
        """Formate un nombre ou renvoie la chaîne telle quelle si non numérique - CORRECTION ICI."""
        try:
            # Si c'est déjà une chaîne, on la retourne telle quelle
            if isinstance(val, str):
                return val
            # Si c'est None, on retourne 'N/A'
            if val is None:
                return 'N/A'
            # Sinon on essaie de formater
            return format(float(val), fmt)
        except (ValueError, TypeError):
            return str(val)

    def read_exr(self, path):
        ex = OpenEXR.InputFile(str(path))
        dw = ex.header()["dataWindow"]
        W0, H0 = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
        buf = ex.channel("Y", Imath.PixelType(Imath.PixelType.FLOAT))
        d = np.frombuffer(buf, dtype=np.float32).reshape(H0, W0)
        return d

    def make_pcd(self, pts, color):
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(pts.reshape(-1, 3))
        p.paint_uniform_color(color)
        return p

    def save_figure(self, fig, filename):
        filepath = self.output_dir / f"{filename}.png"
        fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   Saved: {filepath.name}")
        return filepath

    def load_ground_truth(self):
        print("Loading ground truth data...")
        
        # Caméra 0
        data0 = np.load(self.gt_folder / f"{self.name}.npz")
        self.K_gt0     = data0["intrinsics"]
        self.T_w_c_gt0 = data0["cam2world"]
        # On crée aussi les alias historiques
        self.K_gt      = self.K_gt0
        self.T_w_c_gt  = self.T_w_c_gt0
        
        # Caméra 1
        name1 = self.img2.stem
        data1 = np.load(self.gt_folder / f"{name1}.npz")
        self.K_gt1     = data1["intrinsics"]
        self.T_w_c_gt1 = data1["cam2world"]
        
        # Charger depth EXR + crop
        depth0 = self.read_exr(self.gt_folder / f"{self.name}.exr")
        self.Hc, self.Wc = 384, 512
        y0 = (depth0.shape[0] - self.Hc)//2
        x0 = (depth0.shape[1] - self.Wc)//2
        self.depth_gt = depth0[y0:y0+self.Hc, x0:x0+self.Wc]
        print('okkk')
        print(depth0.shape)

        # Back-projection
        u, v = np.meshgrid(np.arange(self.Wc), np.arange(self.Hc))
        z     = self.depth_gt
        Kc    = self.K_gt.copy()   # c'est bien self.K_gt0 grâce à l'alias
        Kc[0, 2] -= x0
        Kc[1, 2] -= y0
        
        x = (u - Kc[0, 2]) * z / Kc[0, 0]
        y = (v - Kc[1, 2]) * z / Kc[1, 1]
        pts_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        hom     = np.concatenate([pts_cam, np.ones((self.Hc*self.Wc,1))], axis=1).T
        
        # Utilisation de self.T_w_c_gt (alias de T_w_c_gt0)
        self.pts_bl_world = (self.T_w_c_gt @ hom)[:3].T
        self.mask_g       = np.isfinite(self.pts_bl_world).all(axis=1)

        print(f"GT view0 loaded: {self.pts_bl_world.shape}, valid: {self.mask_g.sum()}")

        # ---- Vue 1 : même procédure ----
        name1 = self.img2.stem
        depth1_full = self.read_exr(self.gt_folder / f"{name1}.exr")
        y0_1 = (depth1_full.shape[0] - self.Hc) // 2
        x0_1 = (depth1_full.shape[1] - self.Wc) // 2
        self.depth_gt1 = depth1_full[y0_1:y0_1+self.Hc, x0_1:x0_1+self.Wc]

        z1 = self.depth_gt1
        Kc1 = self.K_gt1.copy()
        Kc1[0, 2] -= x0_1
        Kc1[1, 2] -= y0_1

        x1 = (u - Kc1[0, 2]) * z1 / Kc1[0, 0]
        y1 = (v - Kc1[1, 2]) * z1 / Kc1[1, 1]
        pts_cam1 = np.stack([x1, y1, z1], axis=-1).reshape(-1, 3)
        hom1 = np.concatenate([pts_cam1, np.ones((self.Hc*self.Wc, 1))], axis=1).T

        self.pts_bl_world1 = (self.T_w_c_gt1 @ hom1)[:3].T
        self.mask_g1 = np.isfinite(self.pts_bl_world1).all(axis=1)

        print(f"GT view1 loaded: {self.pts_bl_world1.shape}, valid: {self.mask_g1.sum()}")

    def run_mast3r_inference(self):
        """Exécute l'inférence MASt3R"""
        print("Running MASt3R inference...")

        from MAST3RUtils import MAST3RUtils

        if self.model is not None:
            model = self.model
            print("  Using pre-loaded model")
        else:
            from mast3r.model import AsymmetricMASt3R
            teacher_ckpt = "MOONSt3R.pth"
            student_ckpt = "output/Mast3r_dstillation_exp1A_morelanding/checkpoint-best.pth"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = load_tiny_mast3r_for_inference(
                teacher_ckpt,
                student_ckpt,
                device=device,
                prefer_dinov2=True,
                verbose=True,
                strip_modules=True,
                save_slim_checkpoint_path="checkpoint-best.slim.pth",
            )
        outputs = MAST3RUtils.getMasterOutput(
            model, self.device, str(self.img1), str(self.img2),
            n_matches=100000,
            visualizeMatches=False,
            verboseFlag=False
        )
        
        ( 
          self.matches_im0,
          self.matches_im1,
          self.filtered_matches_im0,
          self.filtered_matches_im1,
          self.pts3d_im0,
          self.pts3d_im1,
          self.conf_im0,
          self.conf_im1,
          *rest
        ) = outputs

    def reconstruct_with_sparse_ga(self):
        """Reconstruct scene using sparse_global_alignment with K_GT initialization.

        Replaces self.pts3d_im0 and self.pts3d_im1 with dense pts3d from
        optimized depth maps + poses + intrinsics (initialized with K_GT).
        Also replaces self.conf_im0 and self.conf_im1 with the optimized confidences.
        """
        from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
        from dust3r.image_pairs import make_pairs
        from dust3r.utils.image import load_images
        from dust3r.utils.device import to_numpy as _to_numpy
        import tempfile

        print("Reconstructing scene with sparse_global_alignment (trying demo-style global_aligner first)...")

        model = self.model
        if model is None:
            raise ValueError("Model must be pre-loaded for sparse_ga mode")

        device = self.device
        filelist = [str(self.img1), str(self.img2)]

        # First attempt: demo-style global_aligner (PointCloudOptimizer) as used in eval_per_view
        try:
            from dust3r.utils.image import load_images
            from dust3r.inference import inference
            from dust3r.image_pairs import make_pairs
            from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

            images = load_images(filelist, size=512, verbose=False)
            pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
            output = inference(pairs, model, device, batch_size=1, verbose=False)

            print('  Running demo-style global_aligner (PointCloudOptimizer)...')
            ga = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
            try:
                ga.compute_global_alignment(init='mst', niter=300, schedule='cosine', lr=0.01)
            except Exception:
                ga.compute_global_alignment()

            pts3d_list = to_numpy(ga.get_pts3d())
            depthmaps = to_numpy(ga.get_depthmaps())
            confs = to_numpy(ga.get_masks())
            poses = ga.get_im_poses().detach().cpu().numpy()
            focals = ga.get_focals().detach().cpu().numpy()
            intrinsics = None

            print(f"  global_aligner produced pts3d shapes: {[p.shape for p in pts3d_list]}")

        except Exception as e:
            print(f"  demo-style global_aligner failed ({e}), falling back to sparse_global_alignment()")
            from dust3r.image_pairs import make_pairs
            from dust3r.utils.image import load_images
            import tempfile

            images = load_images(filelist, size=512, verbose=False)
            # Build init dict with K_GT for both images
            K_GT = np.array([
                [618.0387,   0.,      256.],
                [  0.,     618.0387,  192.],
                [  0.,       0.,        1.]
            ], dtype=np.float64)

            init_dict = {}
            for img_path in filelist:
                init_dict[img_path] = {
                    'intrinsics': torch.tensor(K_GT, dtype=torch.float32),
                }

            pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
            cache_path = tempfile.mkdtemp(prefix="mast3r_sparse_ga_")
            scene = sparse_global_alignment(
                filelist, pairs, cache_path, model,
                lr1=0.01, niter1=500,
                lr2=0.014, niter2=200,
                device=device,
                shared_intrinsics=True,
                matching_conf_thr=5.0,
                init=init_dict,
            )

            pts3d_list, depthmaps, confs = _to_numpy(scene.get_dense_pts3d(clean_depth=True))
            focals = scene.get_focals().cpu().numpy()
            poses = scene.get_im_poses().cpu().numpy()
            intrinsics = [K.cpu().numpy() for K in scene.intrinsics]

            print(f"  Sparse GA dense pts3d shapes: pts3d[0]={pts3d_list[0].shape}, conf[0]={confs[0].shape}")

        # pts3d_list[i] is (H*W, 3) flat in WORLD frame (cam2w applied)
        # confs[i] is (H, W) - use it to get spatial dimensions
        # We need pts3d in cam0 frame for the rest of the pipeline
        T_w2c0 = np.linalg.inv(poses[0])  # (4, 4) world -> cam0

        H, W = confs[0].shape[:2]

        pts3d_im0_world = pts3d_list[0].reshape(-1, 3)  # (N, 3) in world
        pts3d_im1_world = pts3d_list[1].reshape(-1, 3)  # (N, 3) in world

        # Transform to cam0 frame
        pts3d_im0_cam0 = (T_w2c0[:3, :3] @ pts3d_im0_world.T).T + T_w2c0[:3, 3]
        pts3d_im0_cam0 = pts3d_im0_cam0.reshape(H, W, 3)

        pts3d_im1_cam0 = (T_w2c0[:3, :3] @ pts3d_im1_world.T).T + T_w2c0[:3, 3]
        pts3d_im1_cam0 = pts3d_im1_cam0.reshape(H, W, 3)

        # Clean outlier points (global_aligner / sparse_ga can produce extreme values)
        for pts in [pts3d_im0_cam0, pts3d_im1_cam0]:
            norms = np.linalg.norm(pts.reshape(-1, 3), axis=1)
            finite = np.isfinite(norms)
            if finite.sum() > 0:
                p99 = np.percentile(norms[finite], 99)
                outlier = (norms > p99 * 3) | ~finite
                pts.reshape(-1, 3)[outlier] = np.nan

        # Replace pts3d and conf with outputs
        self.pts3d_im0 = pts3d_im0_cam0
        self.pts3d_im1 = pts3d_im1_cam0
        self.conf_im0 = confs[0]
        self.conf_im1 = confs[1]

        # Store extra info
        self.sparse_ga_focals = focals
        self.sparse_ga_poses = poses
        self.sparse_ga_intrinsics = intrinsics

        print(f"  Scene reconstructed: pts3d_im0 {self.pts3d_im0.shape}, pts3d_im1 {self.pts3d_im1.shape}")
        print(f"  Conf ranges: im0=[{np.nanmin(self.conf_im0):.2f}, {np.nanmax(self.conf_im0):.2f}], "
              f"im1=[{np.nanmin(self.conf_im1):.2f}, {np.nanmax(self.conf_im1):.2f}]")

    def stitch_views_from_matches(self):
        """Use MASt3R 2D matches to align view1 pts3d onto view0 pts3d in overlap.

        For each match (y0,x0) <-> (y1,x1):
          - P0 = pts3d_im0[y0, x0]  (3D point seen by view0, in cam0 frame)
          - P1 = pts3d_im1[y1, x1]  (3D point seen by view1, in cam0 frame)
        If stitching is perfect, P0 == P1. Otherwise, estimate rigid transform T
        such that T(P1) ≈ P0, and apply T to all of pts3d_im1.
        """
        print("Stitching views using MASt3R matches...")

        if not hasattr(self, 'matches_im0') or self.matches_im0 is None:
            print("  No matches available, skipping stitching")
            return

        pts3d_v0 = to_numpy(self.pts3d_im0)  # (H, W, 3)
        pts3d_v1 = to_numpy(self.pts3d_im1)  # (H, W, 3)
        H, W = pts3d_v0.shape[:2]

        # matches are (N, 2) with [y, x] pixel coordinates
        # Prefer filtered matches (higher confidence) when available
        if hasattr(self, 'filtered_matches_im0') and self.filtered_matches_im0 is not None and len(self.filtered_matches_im0) > 0:
            m0 = to_numpy(self.filtered_matches_im0).astype(int)
            m1 = to_numpy(self.filtered_matches_im1).astype(int)
        else:
            m0 = to_numpy(self.matches_im0).astype(int)
            m1 = to_numpy(self.matches_im1).astype(int)

        # Clip to valid range
        m0[:, 0] = np.clip(m0[:, 0], 0, H - 1)
        m0[:, 1] = np.clip(m0[:, 1], 0, W - 1)
        m1[:, 0] = np.clip(m1[:, 0], 0, H - 1)
        m1[:, 1] = np.clip(m1[:, 1], 0, W - 1)

        # Get 3D points at match locations
        P0 = pts3d_v0[m0[:, 0], m0[:, 1]]  # (N, 3) - view0's 3D at match pixel
        P1 = pts3d_v1[m1[:, 0], m1[:, 1]]  # (N, 3) - view1's 3D at match pixel

        # Filter valid points (finite, non-zero)
        valid = (np.isfinite(P0).all(axis=1) & np.isfinite(P1).all(axis=1)
                 & (np.linalg.norm(P0, axis=1) > 1e-6)
                 & (np.linalg.norm(P1, axis=1) > 1e-6))
        P0 = P0[valid]
        P1 = P1[valid]
        print(f"  Valid match correspondences: {len(P0)}")

        if len(P0) < 10:
            print("  Too few valid matches, skipping stitching")
            return

        # Before stitching: measure mismatch
        pre_err = np.linalg.norm(P0 - P1, axis=1)
        print(f"  Before stitching: median mismatch = {np.median(pre_err):.4f}m, "
              f"mean = {np.mean(pre_err):.4f}m")

        # Estimate rigid transform P1 -> P0 using RANSAC (Open3D)
        import open3d as o3d
        Nsub = min(5000, len(P0))
        rng = np.random.RandomState(42)
        idxs = rng.choice(len(P0), Nsub, replace=False)

        pcd_src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P1[idxs]))
        pcd_dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P0[idxs]))
        corr = o3d.utility.Vector2iVector(np.stack([np.arange(Nsub), np.arange(Nsub)], axis=1))

        # Use Sim(3) (with_scaling=True) to also correct any scale mismatch between views
        est = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
        dist_thresh = np.median(pre_err) * 5  # generous threshold for RANSAC
        res = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            pcd_src, pcd_dst, corr, dist_thresh,
            estimation_method=est,
            ransac_n=4,
            checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh)],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )

        T_stitch = res.transformation  # 4x4
        M = T_stitch[:3, :3]
        s_stitch = np.cbrt(np.linalg.det(M))
        print(f"  Stitch transform: scale={s_stitch:.6f}, inliers={len(res.correspondence_set)}/{Nsub}")

        # Apply transform to all of pts3d_im1
        flat_v1 = pts3d_v1.reshape(-1, 3)
        flat_v1_stitched = (M @ flat_v1.T).T + T_stitch[:3, 3]
        self.pts3d_im1 = flat_v1_stitched.reshape(H, W, 3)

        # After stitching: measure mismatch on same matches
        P1_stitched = self.pts3d_im1[m1[valid][:, 0], m1[valid][:, 1]]
        post_err = np.linalg.norm(P0 - P1_stitched, axis=1)
        print(f"  After stitching:  median mismatch = {np.median(post_err):.4f}m, "
              f"mean = {np.mean(post_err):.4f}m")

        improvement = (np.median(pre_err) - np.median(post_err)) / np.median(pre_err) * 100
        print(f"  Improvement: {improvement:.1f}%")

        # Optional ICP refinement (point-to-plane on overlap high-confidence subset)
        if getattr(self, 'disable_icp', False):
            print("  ICP refine disabled by flag; skipping ICP refinement")
        else:
            try:
                # Build matched clouds after initial Sim(3) stitch
                src_matched = P1_stitched.copy()
                dst_matched = P0.copy()

                # Heuristic voxel size based on pre_err scale (prevent too large/small)
                median_err = float(np.median(pre_err))
                voxel_size = max(median_err * 0.5, 0.01)

                src_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_matched))
                dst_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst_matched))

                # Downsample for robustness and speed
                src_down = src_pcd.voxel_down_sample(voxel_size)
                dst_down = dst_pcd.voxel_down_sample(voxel_size)

                # Estimate normals for point-to-plane
                radius = max(voxel_size * 2.0, 0.05)
                src_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
                dst_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

                # ICP point-to-plane (rigid, no scaling)
                thresh_icp = max(median_err * 2.0, voxel_size * 2.0)
                icp_res = o3d.pipelines.registration.registration_icp(
                    src_down, dst_down, thresh_icp, np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80)
                )

                T_icp = icp_res.transformation
                # Apply ICP refine on top of Sim(3)
                T_final = T_icp @ T_stitch
                M_final = T_final[:3, :3]
                flat_v1_refined = (M_final @ flat_v1.T).T + T_final[:3, 3]
                self.pts3d_im1 = flat_v1_refined.reshape(H, W, 3)

                # Recompute residuals on matched set
                P1_refined = self.pts3d_im1[m1[valid][:, 0], m1[valid][:, 1]]
                post_err2 = np.linalg.norm(P0 - P1_refined, axis=1)
                print(f"  After ICP point-to-plane refine: median mismatch = {np.median(post_err2):.4f}m, mean = {np.mean(post_err2):.4f}m")
            except Exception as e:
                print(f"  ICP refine failed: {e}")

    def prepare_data(self):
        """Prépare les données pour l'alignement (vue 0 + vue 1)"""
        print("Preparing data for alignment...")

        # Vue 0
        flat_A = to_numpy(self.pts3d_im0).reshape(-1, 3)
        mask_A = (to_numpy(self.conf_im0) >= 1).reshape(-1)
        self.mask_ok = mask_A & self.mask_g
        print(f"View 0: {self.mask_ok.sum()} valid correspondences")

        # Vue 1
        flat_B = to_numpy(self.pts3d_im1).reshape(-1, 3)
        mask_B = (to_numpy(self.conf_im1) >= 1).reshape(-1)
        self.mask_ok1 = mask_B & self.mask_g1
        print(f"View 1: {self.mask_ok1.sum()} valid correspondences")

    @staticmethod
    def umeyama_sim3(src, dst):
        """Closed-form Sim(3) alignment (Umeyama 1991).
        Finds s, R, t that minimize ||dst - (s*R*src + t)||^2.

        Args:
            src: (N, 3) source points
            dst: (N, 3) target points
        Returns:
            s (float), R (3x3), t (3,)
        """
        n, d = src.shape
        mu_src = src.mean(axis=0)
        mu_dst = dst.mean(axis=0)
        src_c = src - mu_src
        dst_c = dst - mu_dst

        var_src = np.sum(src_c ** 2) / n
        cov = (dst_c.T @ src_c) / n

        U, S, Vt = np.linalg.svd(cov)
        det_sign = np.linalg.det(U) * np.linalg.det(Vt)
        D = np.eye(d)
        if det_sign < 0:
            D[-1, -1] = -1

        R = U @ D @ Vt
        s = np.trace(np.diag(S) @ D) / var_src
        t = mu_dst - s * R @ mu_src
        return s, R, t

    def perform_alignment(self, Nsub=5000):
        """Sim(3) alignment using Open3D RANSAC (robust to outliers).

        Uses both views' correspondences for alignment, subsampled to Nsub
        for RANSAC efficiency. The resulting transform is applied to both views.
        """
        print("Performing Sim(3) alignment (RANSAC)...")

        # Gather correspondences from both views
        pred_v0 = to_numpy(self.pts3d_im0).reshape(-1, 3)[self.mask_ok]
        gt_v0 = self.pts_bl_world[self.mask_ok]
        pred_v1 = to_numpy(self.pts3d_im1).reshape(-1, 3)[self.mask_ok1]
        gt_v1 = self.pts_bl_world1[self.mask_ok1]

        src_all = np.vstack([pred_v0, pred_v1])
        dst_all = np.vstack([gt_v0, gt_v1])
        n_total = len(src_all)
        print(f"  Total correspondences: {n_total} (v0={len(pred_v0)}, v1={len(pred_v1)})")

        # Subsample for RANSAC
        Nsub = min(Nsub, n_total)
        rng = np.random.RandomState(42)
        idxs = rng.choice(n_total, Nsub, replace=False)
        X_sub = src_all[idxs]
        Y_sub = dst_all[idxs]

        # RANSAC with Open3D (Sim(3) = with_scaling=True)
        pcd_A = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X_sub))
        pcd_B = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Y_sub))
        corr = np.stack([np.arange(Nsub), np.arange(Nsub)], axis=1)
        corr_o3d = o3d.utility.Vector2iVector(corr)

        est = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
        dist_thresh = 1e5
        res = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            pcd_A, pcd_B, corr_o3d, dist_thresh,
            estimation_method=est,
            ransac_n=4,
            checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_thresh)],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )

        # Extract transformation
        T = res.transformation
        M = T[:3, :3]
        self.scale = np.cbrt(np.linalg.det(M))
        R = M / self.scale
        t = T[:3, 3]

        print(f"  Scale = {self.scale:.6f}")
        print(f"  Inliers = {len(res.correspondence_set)} / {Nsub}")

        # Build 4x4 transform
        self.Tsim3 = np.eye(4)
        self.Tsim3[:3, :3] = self.scale * R
        self.Tsim3[:3, 3] = t

        # Apply to view 0
        flat_A = to_numpy(self.pts3d_im0).reshape(-1, 3)
        A_aligned = (self.Tsim3[:3, :3] @ flat_A.T).T + self.Tsim3[:3, 3]
        self.A_aligned = A_aligned.reshape(self.pts3d_im0.shape)

        # Apply same transform to view 1
        flat_B = to_numpy(self.pts3d_im1).reshape(-1, 3)
        B_aligned = (self.Tsim3[:3, :3] @ flat_B.T).T + self.Tsim3[:3, 3]
        self.B_aligned = B_aligned.reshape(self.pts3d_im1.shape)
        print(f"  Both views aligned with same Sim(3)")

    def compute_chamfer_distance(self, pts_pred, pts_gt, max_points=10000):
        """
        Calcule la distance de Chamfer entre deux nuages de points
        
        Args:
            pts_pred: Points prédits (N, 3)
            pts_gt: Points ground truth (M, 3)
            max_points: Nombre max de points pour éviter l'explosion mémoire
            
        Returns:
            dict: Chamfer distance et métriques associées
        """
        # Sous-échantillonnage si nécessaire pour éviter l'explosion mémoire
        rng = np.random.RandomState(42)
        if len(pts_pred) > max_points:
            idx_pred = rng.choice(len(pts_pred), max_points, replace=False)
            pts_pred_sub = pts_pred[idx_pred]
        else:
            pts_pred_sub = pts_pred

        if len(pts_gt) > max_points:
            idx_gt = rng.choice(len(pts_gt), max_points, replace=False)
            pts_gt_sub = pts_gt[idx_gt]
        else:
            pts_gt_sub = pts_gt
        
        # Calculer les distances pour pred -> gt
        # Pour chaque point prédit, trouve le point GT le plus proche
        from scipy.spatial.distance import cdist
        
        # Calcul par blocs pour éviter l'explosion mémoire
        block_size = 1000
        dist_pred_to_gt = []
        
        for i in range(0, len(pts_pred_sub), block_size):
            end_i = min(i + block_size, len(pts_pred_sub))
            block_pred = pts_pred_sub[i:end_i]
            
            # Distance entre ce bloc et tous les points GT
            distances = cdist(block_pred, pts_gt_sub)  # (block_size, len(pts_gt_sub))
            min_distances = np.min(distances, axis=1)  # Plus proche voisin pour chaque point du bloc
            dist_pred_to_gt.extend(min_distances)
        
        dist_pred_to_gt = np.array(dist_pred_to_gt)
        
        # Calculer les distances pour gt -> pred
        dist_gt_to_pred = []
        
        for i in range(0, len(pts_gt_sub), block_size):
            end_i = min(i + block_size, len(pts_gt_sub))
            block_gt = pts_gt_sub[i:end_i]
            
            # Distance entre ce bloc et tous les points prédits
            distances = cdist(block_gt, pts_pred_sub)  # (block_size, len(pts_pred_sub))
            min_distances = np.min(distances, axis=1)
            dist_gt_to_pred.extend(min_distances)
        
        dist_gt_to_pred = np.array(dist_gt_to_pred)
        
        # Métriques de Chamfer
        chamfer_metrics = {}
        
        # Chamfer Distance = moyenne des distances moyennes dans les deux sens
        chamfer_metrics['chamfer_distance'] = (np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred)) / 2
        
        # Distances directionnelles
        chamfer_metrics['chamfer_pred_to_gt'] = np.mean(dist_pred_to_gt)
        chamfer_metrics['chamfer_gt_to_pred'] = np.mean(dist_gt_to_pred)
        
        # Statistiques additionnelles
        chamfer_metrics['chamfer_pred_to_gt_std'] = np.std(dist_pred_to_gt)
        chamfer_metrics['chamfer_gt_to_pred_std'] = np.std(dist_gt_to_pred)
        
        # Percentiles pour robustesse
        chamfer_metrics['chamfer_pred_to_gt_p95'] = np.percentile(dist_pred_to_gt, 95)
        chamfer_metrics['chamfer_gt_to_pred_p95'] = np.percentile(dist_gt_to_pred, 95)
        
        # Hausdorff distance (distance maximale)
        chamfer_metrics['hausdorff_pred_to_gt'] = np.max(dist_pred_to_gt)
        chamfer_metrics['hausdorff_gt_to_pred'] = np.max(dist_gt_to_pred)
        chamfer_metrics['hausdorff_distance'] = max(chamfer_metrics['hausdorff_pred_to_gt'], 
                                                   chamfer_metrics['hausdorff_gt_to_pred'])
        
        # Métriques de couverture (pourcentage de points à distance < seuil)
        thresholds = [10.0, 20.0, 30.0, 40.0]  # mètres
        for thresh in thresholds:
            coverage_pred = np.mean(dist_pred_to_gt < thresh) * 100
            coverage_gt = np.mean(dist_gt_to_pred < thresh) * 100
            chamfer_metrics[f'coverage_pred_to_gt_{thresh}m'] = coverage_pred
            chamfer_metrics[f'coverage_gt_to_pred_{thresh}m'] = coverage_gt
        
        return chamfer_metrics

    def _compute_depthmap_metrics(self, depth_gt_map, depth_pred_map, mask_ok_map, prefix=""):
        """Compute per-view 2D depth map metrics (SSIM, regression, Pearson on Z)."""
        m = {}
        pfx = f"{prefix}_" if prefix else ""

        valid_gt = depth_gt_map[mask_ok_map]
        valid_pred = depth_pred_map[mask_ok_map]

        if valid_gt.size == 0:
            return m

        # Pearson on Z component
        m[f'{pfx}pearson_r'], _ = pearsonr(valid_gt, valid_pred)

        # SSIM adaptive
        min_d = min(valid_gt.min(), valid_pred.min())
        max_d = max(valid_gt.max(), valid_pred.max())
        dr = max_d - min_d if max_d > min_d else 1.0
        m[f'{pfx}ssim_adaptive'], _ = ssim(depth_gt_map, depth_pred_map, data_range=dr, full=True)

        # Regression
        fin_mask = mask_ok_map & np.isfinite(depth_gt_map) & np.isfinite(depth_pred_map)
        gt_fin = depth_gt_map[fin_mask]
        pred_fin = depth_pred_map[fin_mask]
        if len(gt_fin) > 1:
            slope, intercept, r_value, _, _ = linregress(gt_fin, pred_fin)
            m[f'{pfx}regression_slope'] = slope
            m[f'{pfx}regression_intercept'] = intercept
            m[f'{pfx}regression_r2'] = r_value ** 2

        return m

    def save_aligned_ply(self):
        """Save aligned point clouds (pred + GT) as PLY for visual verification."""
        print("Saving aligned PLY files...")

        # Pred: union of view0 + view1
        pred_v0 = self.A_aligned.reshape(-1, 3)[self.mask_ok]
        pred_v1 = self.B_aligned.reshape(-1, 3)[self.mask_ok1]
        pred_all = np.vstack([pred_v0, pred_v1])

        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(pred_all)
        pcd_pred.paint_uniform_color([0.2, 0.6, 1.0])  # blue
        pred_path = self.output_dir / "aligned_pred.ply"
        o3d.io.write_point_cloud(str(pred_path), pcd_pred)

        # GT: union of view0 + view1
        gt_v0 = self.pts_bl_world[self.mask_ok]
        gt_v1 = self.pts_bl_world1[self.mask_ok1]
        gt_all = np.vstack([gt_v0, gt_v1])

        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(gt_all)
        pcd_gt.paint_uniform_color([1.0, 0.3, 0.2])  # red
        gt_path = self.output_dir / "aligned_gt.ply"
        o3d.io.write_point_cloud(str(gt_path), pcd_gt)

        print(f"  Saved: {pred_path.name} ({len(pred_all)} pts), {gt_path.name} ({len(gt_all)} pts)")

    def compute_geometric_metrics(self):
        """Compute geometric metrics: 3D scene metrics on merged cloud, 2D per-view then averaged."""
        print("Computing geometric metrics...")

        # =====================================================================
        # 1) SCENE-LEVEL 3D METRICS (merged view0 + view1)
        # =====================================================================
        pred_v0 = self.A_aligned.reshape(-1, 3)[self.mask_ok]
        gt_v0   = self.pts_bl_world[self.mask_ok]
        pred_v1 = self.B_aligned.reshape(-1, 3)[self.mask_ok1]
        gt_v1   = self.pts_bl_world1[self.mask_ok1]

        flat_pred = np.vstack([pred_v0, pred_v1])
        flat_gt   = np.vstack([gt_v0, gt_v1])

        # Sim(3) info
        self.metrics['scale'] = self.scale
        self.metrics['scale_error_pct'] = abs(self.scale - 1) * 100
        self.metrics['n_points_v0'] = len(pred_v0)
        self.metrics['n_points_v1'] = len(pred_v1)
        self.metrics['n_points_scene'] = len(flat_pred)

        # Point-to-point 3D errors on full scene
        diff = flat_pred - flat_gt
        self.metrics['rmse'] = np.sqrt((diff ** 2).mean())
        self.metrics['mae'] = np.mean(np.linalg.norm(diff, axis=1))

        # Centroid
        centroid_pred = flat_pred.mean(axis=0)
        centroid_gt = flat_gt.mean(axis=0)
        self.metrics['centroid_diff'] = np.linalg.norm(centroid_pred - centroid_gt)

        # Pearson on Z (scene-level)
        self.metrics['pearson_r'], _ = pearsonr(flat_gt[:, 2], flat_pred[:, 2])

        # Terrain scale info
        terrain_span_3d = np.linalg.norm(np.ptp(flat_gt, axis=0))
        terrain_elevation_range = np.ptp(flat_gt[:, 2])
        self.metrics['terrain_span_3d'] = terrain_span_3d
        self.metrics['terrain_elevation_range'] = terrain_elevation_range
        self.metrics['rmse_relative_span'] = (self.metrics['rmse'] / terrain_span_3d) * 100
        self.metrics['mae_relative_span'] = (self.metrics['mae'] / terrain_span_3d) * 100

        # Chamfer Distance on full scene
        print("  Computing Chamfer Distance (full scene)...")
        chamfer_metrics = self.compute_chamfer_distance(flat_pred, flat_gt)
        self.metrics.update(chamfer_metrics)

        # =====================================================================
        # 1b) OVERLAP CONSISTENCY (view0 vs view1 in overlap region)
        # =====================================================================
        print("  Computing overlap consistency...")
        try:
            from scipy.spatial import cKDTree
            # Use aligned pred points (both in GT world frame after Sim(3))
            # Build KD-tree on view1, query view0 -> nearest neighbor distances
            tree_v1 = cKDTree(pred_v1)
            dist_0to1, _ = tree_v1.query(pred_v0, k=1)
            # And reverse
            tree_v0 = cKDTree(pred_v0)
            dist_1to0, _ = tree_v0.query(pred_v1, k=1)

            # Overlap = points where nearest neighbor is close (< median distance)
            # Use terrain span as reference for threshold
            overlap_thresh = terrain_span_3d * 0.05  # 5% of terrain span
            overlap_mask_v0 = dist_0to1 < overlap_thresh
            overlap_mask_v1 = dist_1to0 < overlap_thresh

            n_overlap_v0 = overlap_mask_v0.sum()
            n_overlap_v1 = overlap_mask_v1.sum()

            if n_overlap_v0 > 10 and n_overlap_v1 > 10:
                # Symmetric overlap distances (only in overlap region)
                overlap_dist = np.concatenate([dist_0to1[overlap_mask_v0], dist_1to0[overlap_mask_v1]])
                self.metrics['overlap_n_v0'] = int(n_overlap_v0)
                self.metrics['overlap_n_v1'] = int(n_overlap_v1)
                self.metrics['overlap_pct_v0'] = float(n_overlap_v0 / len(pred_v0) * 100)
                self.metrics['overlap_pct_v1'] = float(n_overlap_v1 / len(pred_v1) * 100)
                self.metrics['overlap_dist_median'] = float(np.median(overlap_dist))
                self.metrics['overlap_dist_mean'] = float(np.mean(overlap_dist))
                self.metrics['overlap_dist_pct'] = float(np.median(overlap_dist) / terrain_span_3d * 100)
                print(f"    Overlap: {n_overlap_v0} v0 pts, {n_overlap_v1} v1 pts "
                      f"({self.metrics['overlap_pct_v0']:.1f}% / {self.metrics['overlap_pct_v1']:.1f}%)")
                print(f"    Overlap consistency: median={self.metrics['overlap_dist_median']:.4f}m "
                      f"({self.metrics['overlap_dist_pct']:.2f}% of terrain span)")
            else:
                print(f"    WARNING: Not enough overlap points (v0={n_overlap_v0}, v1={n_overlap_v1})")
                self.metrics['overlap_dist_median'] = float('nan')
                self.metrics['overlap_dist_pct'] = float('nan')
        except Exception as e:
            print(f"    Overlap consistency failed: {e}")
            self.metrics['overlap_dist_median'] = float('nan')
            self.metrics['overlap_dist_pct'] = float('nan')

        # =====================================================================
        # 2) PER-VIEW 2D DEPTH MAP METRICS (SSIM, regression, Pearson)
        # =====================================================================

        # View 0
        depth_gt_map_v0 = self.pts_bl_world.reshape(self.Hc, self.Wc, 3)[..., 2]
        depth_pred_map_v0 = self.A_aligned[..., 2]
        mask_ok_map_v0 = self.mask_ok.reshape(self.Hc, self.Wc)
        m_v0 = self._compute_depthmap_metrics(depth_gt_map_v0, depth_pred_map_v0, mask_ok_map_v0, prefix="v0")
        self.metrics.update(m_v0)

        # View 1
        depth_gt_map_v1 = self.pts_bl_world1.reshape(self.Hc, self.Wc, 3)[..., 2]
        depth_pred_map_v1 = self.B_aligned[..., 2]
        mask_ok_map_v1 = self.mask_ok1.reshape(self.Hc, self.Wc)
        m_v1 = self._compute_depthmap_metrics(depth_gt_map_v1, depth_pred_map_v1, mask_ok_map_v1, prefix="v1")
        self.metrics.update(m_v1)

        # Averaged 2D metrics
        for key in ['pearson_r', 'ssim_adaptive', 'regression_slope', 'regression_r2']:
            v0_val = m_v0.get(f'v0_{key}')
            v1_val = m_v1.get(f'v1_{key}')
            if v0_val is not None and v1_val is not None:
                self.metrics[f'{key}_avg'] = (v0_val + v1_val) / 2

        # Store for visualization (view 0, backward compat)
        self.depth_gt_map = depth_gt_map_v0
        self.depth_pred_map = depth_pred_map_v0
        self.mask_ok_map = mask_ok_map_v0

        # Save aligned PLY
        self.save_aligned_ply()

    def compute_profile_metrics(self):
        """Calcule les métriques de profils avec versions relatives"""
        print("Computing profile metrics...")
        
        depth_gt_map = self.pts_bl_world.reshape(self.Hc, self.Wc, 3)[..., 2]
        depth_pred_map = self.A_aligned[..., 2]
        mask_ok_map = self.mask_ok.reshape(self.Hc, self.Wc)
        
        # Échelle de référence pour les profils
        elevation_range = np.ptp(depth_gt_map[mask_ok_map])
        
        # Profil central
        H, W = depth_gt_map.shape
        row = H // 2
        gt_profile = depth_gt_map[row, :]
        pred_profile = depth_pred_map[row, :]
        
        self.profile_metrics['central_mse'] = np.mean((pred_profile - gt_profile) ** 2)
        self.profile_metrics['central_mae'] = np.mean(np.abs(pred_profile - gt_profile))
        self.profile_metrics['central_corr'] = np.corrcoef(gt_profile, pred_profile)[0, 1]
        self.profile_metrics['central_row'] = row
        
        # Métriques relatives pour le profil central
        profile_range = np.ptp(gt_profile)
        if profile_range > 0:
            self.profile_metrics['central_rmse_relative'] = (np.sqrt(self.profile_metrics['central_mse']) / profile_range) * 100
            self.profile_metrics['central_mae_relative'] = (self.profile_metrics['central_mae'] / profile_range) * 100
        
        # Stocker pour visualisation
        self.gt_profile = gt_profile
        self.pred_profile = pred_profile
        
        # Profils multiples
        lines = np.arange(0, H, 50)
        mse_list, mae_list, corr_list = [], [], []
        rmse_rel_list, mae_rel_list = [], []
        
        for y in lines:
            if y < H:
                gt_line = depth_gt_map[y, :]
                pred_line = depth_pred_map[y, :]
                valid = mask_ok_map[y, :] & np.isfinite(gt_line) & np.isfinite(pred_line)
                if valid.any():
                    diff_line = pred_line[valid] - gt_line[valid]
                    mse = np.mean(diff_line**2)
                    mae = np.mean(np.abs(diff_line))
                    
                    mse_list.append(mse)
                    mae_list.append(mae)
                    corr_val = pearsonr(gt_line[valid], pred_line[valid])[0]
                    corr_list.append(corr_val)
                    
                    # Métriques relatives par ligne
                    line_range = np.ptp(gt_line[valid])
                    if line_range > 0:
                        rmse_rel_list.append((np.sqrt(mse) / line_range) * 100)
                        mae_rel_list.append((mae / line_range) * 100)
        
        if mse_list:
            self.profile_metrics['multiple_mse_mean'] = np.mean(mse_list)
            self.profile_metrics['multiple_mse_std'] = np.std(mse_list)
            self.profile_metrics['multiple_mae_mean'] = np.mean(mae_list)
            self.profile_metrics['multiple_mae_std'] = np.std(mae_list)
            self.profile_metrics['multiple_corr_mean'] = np.mean(corr_list)
            self.profile_metrics['multiple_corr_std'] = np.std(corr_list)
            self.profile_metrics['multiple_count'] = len(mse_list)
            
            # Métriques relatives moyennes
            if rmse_rel_list:
                self.profile_metrics['multiple_rmse_relative_mean'] = np.mean(rmse_rel_list)
                self.profile_metrics['multiple_rmse_relative_std'] = np.std(rmse_rel_list)
                self.profile_metrics['multiple_mae_relative_mean'] = np.mean(mae_rel_list)
                self.profile_metrics['multiple_mae_relative_std'] = np.std(mae_rel_list)

    def compute_slope_aspect(self, depth_map, pixel_size=1.0):
        """Calcule la pente et l'aspect"""
        grad_x = sobel(depth_map, axis=1) / (8 * pixel_size)
        grad_y = sobel(depth_map, axis=0) / (8 * pixel_size)
        
        slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
        slope_deg = np.degrees(slope_rad)
        
        aspect_rad = np.arctan2(-grad_y, grad_x)
        aspect_deg = np.degrees(aspect_rad)
        aspect_deg = (aspect_deg + 360) % 360
        
        return slope_deg, aspect_deg, grad_x, grad_y

    def compute_curvature(self, depth_map, pixel_size=1.0):
        """Calcule la courbure"""
        d2z_dx2 = np.gradient(np.gradient(depth_map, axis=1), axis=1) / (pixel_size**2)
        d2z_dy2 = np.gradient(np.gradient(depth_map, axis=0), axis=0) / (pixel_size**2)
        d2z_dxdy = np.gradient(np.gradient(depth_map, axis=1), axis=0) / (pixel_size**2)
        
        dz_dx = np.gradient(depth_map, axis=1) / pixel_size
        dz_dy = np.gradient(depth_map, axis=0) / pixel_size
        
        p, q, r, s, t = dz_dx, dz_dy, d2z_dx2, d2z_dxdy, d2z_dy2
        
        mean_curvature = -((1 + q**2) * r - 2*p*q*s + (1 + p**2) * t) / (2 * (1 + p**2 + q**2)**(3/2))
        gaussian_curvature = (r * t - s**2) / ((1 + p**2 + q**2)**2)
        
        return mean_curvature, gaussian_curvature

    @staticmethod
    def relative_rotation_angle(R_est: np.ndarray, R_gt: np.ndarray) -> float:
        """Angle d'erreur rotation (°) entre deux matrices de rotation."""
        R_err = R_est @ R_gt.T
        ang = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        return np.degrees(ang)

    @staticmethod
    def relative_translation_angle(t_est: np.ndarray, t_gt: np.ndarray) -> float:
        """Angle d'erreur translation (°) entre deux vecteurs."""
        te = t_est / np.linalg.norm(t_est)
        tg = t_gt / np.linalg.norm(t_gt)
        ang = np.arccos(np.clip(np.dot(te, tg), -1, 1))
        return np.degrees(ang)

    def compute_ground_truth_relative_pose(self):
        """
        Calcule la pose relative GT cam0 → cam1 :
        T_rel = inv(T_w_c_gt1) @ T_w_c_gt0
        """
        T_rel = np.linalg.inv(self.T_w_c_gt1) @ self.T_w_c_gt0
        R_gt = T_rel[:3, :3]
        t_gt = T_rel[:3, 3]
        return R_gt, t_gt

    # def compute_relative_pose_metrics(self, ransac_thresh: float = 1.0):
    #     """
    #     Estime la pose relative à partir des matches filtrés et compare à la GT :
    #     - RRA  : erreur rotation
    #     - RTA  : erreur translation
    #     """
    #     from mast3r.fast_nn import fast_reciprocal_NNs  # si nécessaire
    #     # on suppose que self.filtered_matches_im0 / im1 ont été remplis en inference
    #     if not hasattr(self, 'filtered_matches_im0'):
    #         print("No filtered matches found – skipping RRA/RTA")
    #         return

    #     # Ajuster les intrinsics au crop
    #     depth0 = self.read_exr(self.gt_folder / f"{self.name}.exr")
    #     self.Hc, self.Wc = 384, 512
    #     y0 = (depth0.shape[0] - self.Hc)//2
    #     x0 = (depth0.shape[1] - self.Wc)//2
    #     Kc    = self.K_gt.copy()   # c'est bien self.K_gt0 grâce à l'alias
    #     Kc[0, 2] -= x0
    #     Kc[1, 2] -= y0                # Estimation essentielle + recoverPose
    #     E, mask = cv2.findEssentialMat(
    #         self.filtered_matches_im0.astype(np.float32),
    #         self.filtered_matches_im1.astype(np.float32),
    #         cameraMatrix=Kc, method=cv2.RANSAC, prob=0.999, threshold=ransac_thresh
    #     )
    #     _, R_est, t_est, _ = cv2.recoverPose(E,
    #         self.filtered_matches_im0[mask.ravel().astype(bool)],
    #         self.filtered_matches_im1[mask.ravel().astype(bool)],
    #         cameraMatrix=Kc
    #     )
    #     # Pose GT relative
    #     R_gt_rel, t_gt_rel = self.compute_ground_truth_relative_pose()
    #     # Calcul des métriques
    #     rra = self.relative_rotation_angle(R_est, R_gt_rel)
    #     rta = self.relative_translation_angle(t_est.ravel(), t_gt_rel)
    #     self.metrics['RRA_deg'] = rra
    #     self.metrics['RTA_deg'] = rta
    #     print(f"► RRA = {rra:.3f}°, RTA = {rta:.3f}°")
    

    def compute_relative_pose_metrics(self, ransac_thresh: float = 1.0):
        """
        Estime la pose relative à partir des matches filtrés et compare à la GT :
        - RRA  : erreur rotation
        - RTA  : erreur translation
        """
        from mast3r.fast_nn import fast_reciprocal_NNs  # si nécessaire
        # on suppose que self.filtered_matches_im0 / im1 ont été remplis en inference
        if not hasattr(self, 'filtered_matches_im0'):
            print("No filtered matches found – skipping RRA/RTA")
            return

        # Ajuster les intrinsics au crop
        depth0 = self.read_exr(self.gt_folder / f"{self.name}.exr")
        self.Hc, self.Wc = 384, 512
        y0 = (depth0.shape[0] - self.Hc) // 2
        x0 = (depth0.shape[1] - self.Wc) // 2
        Kc = self.K_gt.copy()   # c'est bien self.K_gt0 grâce à l'alias
        Kc[0, 2] -= x0
        Kc[1, 2] -= y0

        # Estimation essentielle + recoverPose
        E, mask = cv2.findEssentialMat(
            self.filtered_matches_im0.astype(np.float32),
            self.filtered_matches_im1.astype(np.float32),
            cameraMatrix=Kc, method=cv2.RANSAC, prob=0.999, threshold=ransac_thresh
        )
        _, R_est, t_est, _ = cv2.recoverPose(E,
            self.filtered_matches_im0[mask.ravel().astype(bool)],
            self.filtered_matches_im1[mask.ravel().astype(bool)],
            cameraMatrix=Kc
        )

        # Pose GT relative
        R_gt_rel, t_gt_rel = self.compute_ground_truth_relative_pose()

        # Calcul des métriques
        rra = self.relative_rotation_angle(R_est, R_gt_rel)
        rta = self.relative_translation_angle(t_est.ravel(), t_gt_rel)

        self.metrics['RRA_deg'] = rra
        self.metrics['RTA_deg'] = rta
        print(f"► RRA = {rra:.3f}°, RTA = {rta:.3f}°")

    def _compute_dust3r_crop_box(self, width: int, height: int, square_ok: bool = False):
        """Return (x0, y0, w, h) for the centred crop DUSt3R applies when size=512."""
        cx, cy = width // 2, height // 2
        halfw = ((2 * cx) // 16) * 8       # width multiple of 8
        halfh = ((2 * cy) // 16) * 8       # height multiple of 8
        if (not square_ok) and width == height:
            halfh = 3 * halfw // 4          # enforce 4:3 window on a square input
        w, h = 2 * halfw, 2 * halfh
        x0, y0 = cx - halfw, cy - halfh
        return x0, y0, w, h

    def shift_intrinsics_for_crop(self, K: np.ndarray, width: int, height: int, square_ok: bool = False):
        x0, y0, _, _ = self._compute_dust3r_crop_box(width, height, square_ok)
        Kc = K.copy().astype(np.float32)
        Kc[0, 2] -= x0
        Kc[1, 2] -= y0
        return Kc

    def estimate_relative_pose_from_matches(self, kpts0: np.ndarray, kpts1: np.ndarray, K: np.ndarray, ransac_thresh: float = 2.0):
        kpts0 = kpts0.astype(np.float32)
        kpts1 = kpts1.astype(np.float32)
        # Ensure at least 5 points for findEssentialMat
        if len(kpts0) < 5:
            raise RuntimeError(f"Not enough points for Essential Matrix estimation. Got {len(kpts0)} but need at least 5.")

        E, mask = cv2.findEssentialMat(kpts0, kpts1, cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=ransac_thresh)
        if E is None or mask is None:
            raise RuntimeError("Essential matrix estimation failed – no matrix or mask returned.")

        inliers = mask.ravel().astype(bool)
        if inliers.sum() < 5: # Need at least 5 inliers for recoverPose
            raise RuntimeError(f"Essential matrix estimation failed – not enough inliers ({inliers.sum()}).")

        _, R, t, _ = cv2.recoverPose(E, kpts0[inliers], kpts1[inliers], cameraMatrix=K)
        return R, t.ravel(), inliers

    def compute_relative_pose_from_filtered_matches(self, filtered_matches_im0: np.ndarray, filtered_matches_im1: np.ndarray, K_orig: np.ndarray, orig_width: int = 512, orig_height: int = 512, ransac_thresh: float = 0.4):
        print(f"Original K:\n{K_orig}")
        K_crop = self.shift_intrinsics_for_crop(K_orig, orig_width, orig_height)
        print(f"Cropped K:\n{K_crop}")
        R, t, inliers = self.estimate_relative_pose_from_matches(filtered_matches_im0, filtered_matches_im1, K_crop, ransac_thresh)
        return R, t, inliers, K_crop

    def compute_relative_pose_metrics_with_thresholds(self, ransac_thresholds: list = None):
        """
        Compute relative pose metrics with different RANSAC thresholds.
        """
        if ransac_thresholds is None:
            ransac_thresholds = [
                0.05, 0.075, 0.1, 0.125, 0.15, 0.175,
                0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8,
                0.9, 1.0, 1.2, 1.5, 1.8,
                2.0, 2.5, 3.0, 3.5, 4.0,
                5.0, 6.0, 7.0, 8.0, 9.0, 10.0
            ]

        gt_poses = np.stack([self.T_w_c_gt0, self.T_w_c_gt1])
        intrinsics_all = np.stack([self.K_gt0, self.K_gt1])

        # Ground-truth relative pose
        T_rel_gt = np.linalg.inv(gt_poses[1]) @ gt_poses[0]
        R_gt, t_gt = T_rel_gt[:3, :3], T_rel_gt[:3, 3]
        print("\n--- Ground-truth relative pose cam0 → cam1 ---")
        print(f"  t  : {t_gt}  (‖t‖={np.linalg.norm(t_gt):.3f})")
        print(f"  rot: {self.relative_rotation_angle(np.eye(3), R_gt):.2f}°")
        print("-" * 50)

        results = []

        for thresh in ransac_thresholds:
            print(f"\n--- Testing with ransac_thresh = {thresh:.1f} ---")
            try:
                R_est, t_est, inliers, K_crop = self.compute_relative_pose_from_filtered_matches(
                    self.filtered_matches_im0,
                    self.filtered_matches_im1,
                    intrinsics_all[0],
                    ransac_thresh=thresh
                )

                num_inliers = inliers.sum()
                total_matches = inliers.size

                print("\n Estimated relative pose (cam0 → cam1)")
                print(f"  #inliers : {num_inliers} / {total_matches}")
                print(f"  t        : {t_est}  (‖t‖={np.linalg.norm(t_est):.3f})")

                # Metrics
                rra = self.relative_rotation_angle(R_est, R_gt)
                rta = self.relative_translation_angle(t_est, t_gt)

                print("\nMetrics vs GT")
                print(f"  RRA (°) : {rra:.3f}")
                print(f"  RTA (°) : {rta:.3f}")
                
                results.append({
                    "threshold": thresh,
                    "RRA": rra,
                    "RTA": rta,
                    "inliers": num_inliers,
                    "total_matches": total_matches
                })

            except RuntimeError as e:
                print(f"Pose estimation failed for threshold {thresh}: {e}")
                results.append({
                    "threshold": thresh,
                    "RRA": np.inf, # Use infinity for failed estimations
                    "RTA": np.inf,
                    "inliers": 0,
                    "total_matches": len(self.filtered_matches_im0) if hasattr(self, 'filtered_matches_im0') else 0
                })
            print("-" * 50)

        # Find the best result
        print("\n\n--- Summary of Results ---")
        best_rra = np.inf
        best_rta = np.inf
        best_thresh_rra = None
        best_thresh_rta = None

        for res in results:
            print(f"Threshold {res['threshold']:.1f}: RRA={res['RRA']:.3f}°, RTA={res['RTA']:.3f}°, Inliers={res['inliers']}/{res['total_matches']}")

            if res["RRA"] < best_rra:
                best_rra = res["RRA"]
                best_thresh_rra = res["threshold"]

            if res["RTA"] < best_rta:
                best_rta = res["RTA"]
                best_thresh_rta = res["threshold"]

        print("\n--- Best Thresholds Found ---")
        if best_thresh_rra is not None:
            print(f"Best RRA ({best_rra:.3f}°) achieved with threshold: {best_thresh_rra:.1f}")
        if best_thresh_rta is not None:
            print(f"Best RTA ({best_rta:.3f}°) achieved with threshold: {best_thresh_rta:.1f}")

        # Find the threshold that minimizes the sum of RRA and RTA
        best_sum_error = np.inf
        best_overall_thresh = None

        for res in results:
            if res["RRA"] != np.inf and res["RTA"] != np.inf: # Only consider successful estimations
                current_sum_error = res["RRA"] + res["RTA"]
                if current_sum_error < best_sum_error:
                    best_sum_error = current_sum_error
                    best_overall_thresh = res["threshold"]
        self.metrics['RRA_deg'] = best_rra
        self.metrics['RTA_deg'] = best_rta
        if best_overall_thresh is not None:
            print(f"\nOverall best threshold (minimizing RRA + RTA): {best_overall_thresh:.1f} (Sum of errors: {best_sum_error:.3f}°)")
        else:
            print("\nNo successful pose estimations found for analysis.")

        return results




    def compute_roughness(self, depth_map, window_size=3):
        """Calcule la rugosité locale"""
        local_mean = uniform_filter(depth_map, size=window_size)
        local_var = uniform_filter(depth_map**2, size=window_size) - local_mean**2
        roughness = np.sqrt(np.maximum(local_var, 0))
        return roughness

    def compute_terrain_classification(self, slope_deg, curvature):
        """Classification du terrain"""
        classification = np.zeros_like(slope_deg, dtype=int)
        
        flat_mask = slope_deg < 5
        gentle_slope = (slope_deg >= 5) & (slope_deg < 15)
        steep_slope = slope_deg >= 15
        ridge_mask = (curvature < -0.01) & steep_slope
        valley_mask = (curvature > 0.01) & steep_slope
        
        classification[flat_mask] = 0
        classification[gentle_slope] = 1
        classification[steep_slope] = 2
        classification[ridge_mask] = 3
        classification[valley_mask] = 4
        
        return classification

    def _compute_terrain_metrics_single(self, depth_gt_map, depth_pred_map, mask_ok_map, prefix=""):
        """Compute terrain metrics for a single view. Returns dict with prefixed keys."""
        pfx = f"{prefix}_" if prefix else ""
        m = {}

        slope_gt, aspect_gt, _, _ = self.compute_slope_aspect(depth_gt_map)
        slope_pred, aspect_pred, _, _ = self.compute_slope_aspect(depth_pred_map)
        curvature_gt, gauss_curv_gt = self.compute_curvature(depth_gt_map)
        curvature_pred, gauss_curv_pred = self.compute_curvature(depth_pred_map)
        roughness_gt = self.compute_roughness(depth_gt_map)
        roughness_pred = self.compute_roughness(depth_pred_map)

        valid_mask = mask_ok_map & np.isfinite(slope_gt) & np.isfinite(slope_pred)

        if valid_mask.any():
            m[f'{pfx}slope_rmse'] = np.sqrt(np.mean((slope_pred[valid_mask] - slope_gt[valid_mask])**2))
            m[f'{pfx}slope_mae'] = np.mean(np.abs(slope_pred[valid_mask] - slope_gt[valid_mask]))
            m[f'{pfx}slope_corr'] = np.corrcoef(slope_gt[valid_mask], slope_pred[valid_mask])[0, 1]

            curv_valid = np.isfinite(curvature_gt) & np.isfinite(curvature_pred) & valid_mask
            if curv_valid.any():
                m[f'{pfx}curv_rmse'] = np.sqrt(np.mean((curvature_pred[curv_valid] - curvature_gt[curv_valid])**2))
                m[f'{pfx}curv_mae'] = np.mean(np.abs(curvature_pred[curv_valid] - curvature_gt[curv_valid]))
                m[f'{pfx}curv_corr'] = np.corrcoef(curvature_gt[curv_valid], curvature_pred[curv_valid])[0, 1]

            m[f'{pfx}rough_rmse'] = np.sqrt(np.mean((roughness_pred[valid_mask] - roughness_gt[valid_mask])**2))
            m[f'{pfx}rough_mae'] = np.mean(np.abs(roughness_pred[valid_mask] - roughness_gt[valid_mask]))
            m[f'{pfx}rough_corr'] = np.corrcoef(roughness_gt[valid_mask], roughness_pred[valid_mask])[0, 1]

            m[f'{pfx}slope_gt_mean'] = slope_gt[valid_mask].mean()
            m[f'{pfx}slope_pred_mean'] = slope_pred[valid_mask].mean()
            m[f'{pfx}rough_gt_mean'] = roughness_gt[valid_mask].mean()
            m[f'{pfx}rough_pred_mean'] = roughness_pred[valid_mask].mean()

        # Return also the raw maps for viz (caller decides whether to store)
        extras = {
            'slope_gt': slope_gt, 'slope_pred': slope_pred,
            'aspect_gt': aspect_gt, 'aspect_pred': aspect_pred,
            'curvature_gt': curvature_gt, 'curvature_pred': curvature_pred,
            'gauss_curv_gt': gauss_curv_gt, 'gauss_curv_pred': gauss_curv_pred,
            'roughness_gt': roughness_gt, 'roughness_pred': roughness_pred,
        }
        return m, extras

    def compute_terrain_metrics(self):
        """Compute terrain metrics per-view then average."""
        print("Computing terrain metrics...")

        # View 0
        dg0 = self.pts_bl_world.reshape(self.Hc, self.Wc, 3)[..., 2]
        dp0 = self.A_aligned[..., 2]
        mk0 = self.mask_ok.reshape(self.Hc, self.Wc)
        m_v0, extras_v0 = self._compute_terrain_metrics_single(dg0, dp0, mk0, prefix="v0")
        self.terrain_metrics.update(m_v0)

        # View 1
        dg1 = self.pts_bl_world1.reshape(self.Hc, self.Wc, 3)[..., 2]
        dp1 = self.B_aligned[..., 2]
        mk1 = self.mask_ok1.reshape(self.Hc, self.Wc)
        m_v1, extras_v1 = self._compute_terrain_metrics_single(dg1, dp1, mk1, prefix="v1")
        self.terrain_metrics.update(m_v1)

        # Averaged
        for key in ['slope_rmse', 'slope_mae', 'slope_corr', 'curv_rmse', 'curv_mae', 'curv_corr',
                     'rough_rmse', 'rough_mae', 'rough_corr']:
            v0_val = m_v0.get(f'v0_{key}')
            v1_val = m_v1.get(f'v1_{key}')
            if v0_val is not None and v1_val is not None:
                self.terrain_metrics[f'{key}_avg'] = (v0_val + v1_val) / 2

        # Store view 0 extras for visualization backward compat
        self.slope_gt, self.slope_pred = extras_v0['slope_gt'], extras_v0['slope_pred']
        self.aspect_gt, self.aspect_pred = extras_v0['aspect_gt'], extras_v0['aspect_pred']
        self.curvature_gt, self.curvature_pred = extras_v0['curvature_gt'], extras_v0['curvature_pred']
        self.gauss_curv_gt, self.gauss_curv_pred = extras_v0['gauss_curv_gt'], extras_v0['gauss_curv_pred']
        self.roughness_gt, self.roughness_pred = extras_v0['roughness_gt'], extras_v0['roughness_pred']

        # Terrain classification (view 0 only, for viz)
        self.terrain_class_gt = self.compute_terrain_classification(extras_v0['slope_gt'], extras_v0['curvature_gt'])
        self.terrain_class_pred = self.compute_terrain_classification(extras_v0['slope_pred'], extras_v0['curvature_pred'])
    def compute_slope_on_pointcloud(self, pts3d: np.ndarray, shape_hw: tuple, pixel_size=1.0):
        """
        Calcule la pente sur un nuage de points régulier (issu d'une depth map).
        Args:
            pts3d: (H*W, 3) ou (H, W, 3)
            shape_hw: tuple (H, W) pour reshaping
            pixel_size: taille du pixel en mètres
        Returns:
            slope_map: carte des pentes (en °)
        """
        if pts3d.ndim == 2:
            pts3d = pts3d.reshape(*shape_hw, 3)

        dz_dx = sobel(pts3d[..., 2], axis=1) / (8 * pixel_size)
        dz_dy = sobel(pts3d[..., 2], axis=0) / (8 * pixel_size)

        slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
        slope_deg = np.degrees(slope_rad)

        return slope_deg
    def compute_slope_on_pointcloud_normals(self, pts3d: np.ndarray, shape_hw: tuple, k_neighbors=30):
        """
        Calcule la pente locale d’un nuage 3D par estimation de normales Open3D.
        La pente est l’angle entre la normale locale et l’axe Z vertical.
        """
        import open3d as o3d

        if pts3d.ndim == 2:
            pts3d = pts3d.reshape(*shape_hw, 3)

        flat_pts = pts3d.reshape(-1, 3)

        # Création du nuage
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(flat_pts)

        # Estimation des normales
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors)
        )

        normals = np.asarray(pcd.normals)  # (N, 3)
        cos_theta = np.clip(normals[:, 2], -1.0, 1.0)  # projection sur Z
        slope_rad = np.arccos(np.abs(cos_theta))
        slope_deg = np.degrees(slope_rad)

        return slope_deg.reshape(*shape_hw)

    def compute_slope_correlation(self):
        """
        Corrélation entre les pentes 3D (à partir des normales) de GT et prédictions.
        """
        print("Computing 3D slope correlation on pointclouds (normals)...")

        shape_hw = (self.Hc, self.Wc)
        gt_3d = self.pts_bl_world.reshape(-1, 3)
        pred_3d = self.A_aligned.reshape(-1, 3)

        # Calcul des pentes à partir des normales
        slope_gt_pc = self.compute_slope_on_pointcloud_normals(gt_3d, shape_hw)
        slope_pred_pc = self.compute_slope_on_pointcloud_normals(pred_3d, shape_hw)

        # Flatten + filtrage
        slope_gt_flat = slope_gt_pc.flatten()
        slope_pred_flat = slope_pred_pc.flatten()
        mask_flat = self.mask_ok.flatten()
        valid = mask_flat & np.isfinite(slope_gt_flat) & np.isfinite(slope_pred_flat)

        if valid.sum() < 2:
            print("❌ Pas assez de points valides pour corrélation 3D (normales)")
            corr = np.nan
        else:
            corr = np.corrcoef(slope_gt_flat[valid], slope_pred_flat[valid])[0, 1]
            print(f"► Corrélation pente (nuage 3D - normales): {corr:.4f}")

        self.terrain_metrics['slope_corr_pointcloud'] = corr
        self.slope_gt_pc = slope_gt_pc
        self.slope_pred_pc = slope_pred_pc


    def save_3d_slope_visualization(self):
        """
        Sauvegarde une visualisation 3D colorée par pente pour GT et Pred
        """
        print("Saving 3D slope visualization (colored pointclouds)...")
        from matplotlib import cm

        # GT
        pcd_gt = o3d.geometry.PointCloud()
        pcd_gt.points = o3d.utility.Vector3dVector(self.pts_bl_world.reshape(-1, 3))
        norm_gt = (self.slope_gt_pc - self.slope_gt_pc.min()) / (self.slope_gt_pc.ptp() + 1e-6)
        cmap_gt = cm.get_cmap("plasma")
        colors_gt = cmap_gt(norm_gt.reshape(-1))[..., :3]
        pcd_gt.colors = o3d.utility.Vector3dVector(colors_gt)

        # Pred
        pcd_pred = o3d.geometry.PointCloud()
        pcd_pred.points = o3d.utility.Vector3dVector(self.A_aligned.reshape(-1, 3))
        norm_pred = (self.slope_pred_pc - self.slope_pred_pc.min()) / (self.slope_pred_pc.ptp() + 1e-6)
        cmap_pred = cm.get_cmap("plasma")
        colors_pred = cmap_pred(norm_pred.reshape(-1))[..., :3]
        pcd_pred.colors = o3d.utility.Vector3dVector(colors_pred)

        # Sauvegarde (optionnelle: capture image automatique)
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(pcd_gt)
        vis.add_geometry(pcd_pred)
        vis.poll_events()
        vis.update_renderer()

        img_path = self.output_dir / "slope_colored_pointcloud.png"
        vis.capture_screen_image(str(img_path), do_render=True)
        vis.destroy_window()

        print(f"✓ Sauvegardé: {img_path}")
    def compute_error_statistics(self):
        """Calcule les statistiques d'erreur"""
        print("Computing error statistics...")
        
        depth_gt_map = self.pts_bl_world.reshape(self.Hc, self.Wc, 3)[..., 2]
        depth_pred_map = self.A_aligned[..., 2]
        mask_ok_map = self.mask_ok.reshape(self.Hc, self.Wc)
        
        # Erreurs sur les pixels valides
        flat_gt_hist = depth_gt_map[mask_ok_map]
        flat_pred_hist = depth_pred_map[mask_ok_map]
        errors = flat_pred_hist - flat_gt_hist
        
        self.error_stats['mean'] = errors.mean()
        self.error_stats['std'] = errors.std()
        self.error_stats['median'] = np.median(errors)
        self.error_stats['min'] = errors.min()
        self.error_stats['max'] = errors.max()
        self.error_stats['count'] = len(errors)
        
        # Stocker pour visualisation
        self.errors = errors
        self.flat_gt_hist = flat_gt_hist
        self.flat_pred_hist = flat_pred_hist

    def generate_visualizations(self):
        """Génère toutes les visualisations"""
        print("Generating visualizations...")
        
        depth_gt_map = self.pts_bl_world.reshape(self.Hc, self.Wc, 3)[..., 2]
        depth_pred_map = self.A_aligned[..., 2]
        diff_map = depth_pred_map - depth_gt_map
        
        # 1) Depth maps de base
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        im1 = axes[0].imshow(depth_pred_map, cmap='viridis')
        axes[0].set_title('Depth Map MASt3R')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(depth_gt_map, cmap='viridis')
        axes[1].set_title('Depth Map GT')
        plt.colorbar(im2, ax=axes[1])
        
        im3 = axes[2].imshow(diff_map, cmap='RdBu_r', vmin=-np.std(diff_map)*2, vmax=np.std(diff_map)*2)
        axes[2].set_title('Difference (Pred - GT)')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        self.save_figure(fig, "01_depth_comparison_basic")
        plt.close()
        
        # 2) Cartes avec hillshading
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        ls = LightSource(azdeg=315, altdeg=45)
        hillshade_gt = ls.hillshade(depth_gt_map, vert_exag=2)
        hillshade_pred = ls.hillshade(depth_pred_map, vert_exag=2)
        
        im1 = axes[0, 0].imshow(depth_gt_map, cmap='terrain')
        axes[0, 0].set_title('Profondeur GT')
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].imshow(depth_pred_map, cmap='terrain')
        axes[0, 1].set_title('Profondeur Prédite')
        plt.colorbar(im2, ax=axes[0, 1])
        
        im3 = axes[0, 2].imshow(diff_map, cmap='RdBu_r', vmin=-np.std(diff_map)*2, vmax=np.std(diff_map)*2)
        axes[0, 2].set_title('Différence (Pred - GT)')
        plt.colorbar(im3, ax=axes[0, 2])
        
        axes[1, 0].imshow(hillshade_gt, cmap='gray')
        axes[1, 0].set_title('Hillshade GT')
        
        axes[1, 1].imshow(hillshade_pred, cmap='gray')
        axes[1, 1].set_title('Hillshade Prédite')
        
        abs_error = np.abs(diff_map)
        im6 = axes[1, 2].imshow(abs_error, cmap='Reds')
        axes[1, 2].set_title('Erreur Absolue')
        plt.colorbar(im6, ax=axes[1, 2])
        
        plt.tight_layout()
        self.save_figure(fig, "02_depth_comparison_hillshade")
        plt.close()
        
        # 3) Profil central
        if hasattr(self, 'gt_profile'):
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(self.gt_profile, label='GT profile', linewidth=2)
            ax.plot(self.pred_profile, label='Pred profile', linewidth=2)
            ax.set_title(f'Profil de profondeur central (ligne {self.profile_metrics["central_row"]})')
            ax.set_xlabel('Colonne de pixel')
            ax.set_ylabel('Profondeur (m)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            self.save_figure(fig, "03_central_profile")
            plt.close()
        
        # 4) Analyse des erreurs + Chamfer Distance
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Histogramme des erreurs
        axes[0, 0].hist(self.errors, bins=100, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title("Histogramme des erreurs (pred − gt)")
        axes[0, 0].set_xlabel("Erreur (m)")
        axes[0, 0].set_ylabel("Nombre de pixels")
        axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7, label='Erreur nulle')
        axes[0, 0].axvline(self.errors.mean(), color='green', linestyle='-', alpha=0.7, 
                          label=f'Moyenne: {self.errors.mean():.3f}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Scatter plot GT vs Pred
        sample_indices = np.random.choice(len(self.flat_gt_hist), min(5000, len(self.flat_gt_hist)), replace=False)
        axes[0, 1].scatter(self.flat_gt_hist[sample_indices], self.flat_pred_hist[sample_indices], alpha=0.5, s=1)
        min_val = min(self.flat_gt_hist.min(), self.flat_pred_hist.min())
        max_val = max(self.flat_gt_hist.max(), self.flat_pred_hist.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', label='Ligne parfaite')
        axes[0, 1].set_xlabel('Profondeur GT (m)')
        axes[0, 1].set_ylabel('Profondeur Pred (m)')
        axes[0, 1].set_title(f'GT vs Pred (r={self.metrics["pearson_r"]:.3f})')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Métriques Chamfer (graphique en barres)
        chamfer_names = ['Chamfer\nDistance', 'Pred→GT', 'GT→Pred', 'Hausdorff']
        chamfer_values = [
            self.metrics.get('chamfer_distance', 0),
            self.metrics.get('chamfer_pred_to_gt', 0),
            self.metrics.get('chamfer_gt_to_pred', 0),
            self.metrics.get('hausdorff_distance', 0)
        ]
        
        bars = axes[1, 0].bar(chamfer_names, chamfer_values, color=['steelblue', 'lightcoral', 'lightgreen', 'gold'])
        axes[1, 0].set_ylabel('Distance (m)')
        axes[1, 0].set_title('Métriques de Distance 3D')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for bar, value in zip(bars, chamfer_values):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Couverture Chamfer (pourcentage de points à distance < seuil)
        thresholds = [10.0, 20.0, 30.0, 40.0]  # mètres
        coverage_pred = [self.metrics.get(f'coverage_pred_to_gt_{t}m', 0) for t in thresholds]
        coverage_gt = [self.metrics.get(f'coverage_gt_to_pred_{t}m', 0) for t in thresholds]
        
        x_thresh = np.arange(len(thresholds))
        width = 0.35
        
        axes[1, 1].bar(x_thresh - width/2, coverage_pred, width, label='Pred→GT', alpha=0.8)
        axes[1, 1].bar(x_thresh + width/2, coverage_gt, width, label='GT→Pred', alpha=0.8)
        axes[1, 1].set_xlabel('Seuil de distance (m)')
        axes[1, 1].set_ylabel('Couverture (%)')
        axes[1, 1].set_title('Couverture par seuil de distance')
        axes[1, 1].set_xticks(x_thresh)
        axes[1, 1].set_xticklabels([f'{t}m' for t in thresholds])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, "04_error_chamfer_analysis")
        plt.close()
        
        # 5) Cartes de relief si disponibles
        if hasattr(self, 'slope_gt'):
            self._generate_relief_visualizations()

    def _generate_relief_visualizations(self):
        """Génère les visualisations de relief avec des échelles de couleur adaptatives."""
        
        # Masque de validité commun pour tous les calculs
        # On s'assure que toutes les cartes ont des valeurs finies aux mêmes endroits
        mask_ok_map = self.mask_ok.reshape(self.Hc, self.Wc)
        valid_mask = (mask_ok_map & 
                    np.isfinite(self.slope_gt) & np.isfinite(self.slope_pred) &
                    np.isfinite(self.roughness_gt) & np.isfinite(self.roughness_pred))

        # --- Stratégie 1 : Calcul de l'échelle commune pour les cartes de pente ---
        valid_slopes_gt = self.slope_gt[valid_mask]
        valid_slopes_pred = self.slope_pred[valid_mask]
        # Utiliser les percentiles pour une échelle robuste
        vmin_slope = np.percentile(np.concatenate([valid_slopes_gt, valid_slopes_pred]), 2)
        vmax_slope = np.percentile(np.concatenate([valid_slopes_gt, valid_slopes_pred]), 98)

        # --- Stratégie 2 : Calcul de l'échelle symétrique pour la différence de pente ---
        slope_diff = self.slope_pred - self.slope_gt
        vmax_slope_diff = np.percentile(np.abs(slope_diff[valid_mask]), 98)

        # --- Stratégie 2 : Calcul de l'échelle symétrique pour la différence de rugosité ---
        roughness_diff = self.roughness_pred - self.roughness_gt
        vmax_rough_diff = np.percentile(np.abs(roughness_diff[valid_mask]), 98)

        # --- Affichage des cartes de pente, aspect et rugosité ---
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
        fig.suptitle("Analyse des Pentes, Aspects et Rugosités", fontsize=16)

        # Pente GT (échelle adaptative)
        im1 = axes[0, 0].imshow(self.slope_gt, cmap='YlOrRd', vmin=vmin_slope, vmax=vmax_slope)
        axes[0, 0].set_title('Pente GT (°)')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Pente Prédite (même échelle adaptative)
        im2 = axes[0, 1].imshow(self.slope_pred, cmap='YlOrRd', vmin=vmin_slope, vmax=vmax_slope)
        axes[0, 1].set_title('Pente Prédite (°)')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Différence Pente (échelle symétrique adaptative)
        im3 = axes[0, 2].imshow(slope_diff, cmap='RdBu_r', vmin=-vmax_slope_diff, vmax=vmax_slope_diff)
        axes[0, 2].set_title('Différence Pente (°)')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Aspect (échelle fixe car c'est un angle 0-360)
        im4 = axes[1, 0].imshow(self.aspect_gt, cmap='hsv', vmin=0, vmax=360)
        axes[1, 0].set_title('Aspect GT (°)')
        plt.colorbar(im4, ax=axes[1, 0])
        
        im5 = axes[1, 1].imshow(self.aspect_pred, cmap='hsv', vmin=0, vmax=360)
        axes[1, 1].set_title('Aspect Prédit (°)')
        plt.colorbar(im5, ax=axes[1, 1])
        
        # Différence Rugosité (échelle symétrique adaptative)
        im6 = axes[1, 2].imshow(roughness_diff, cmap='RdBu_r', vmin=-vmax_rough_diff, vmax=vmax_rough_diff)
        axes[1, 2].set_title('Différence Rugosité')
        plt.colorbar(im6, ax=axes[1, 2])
        
        self.save_figure(fig, "05_slope_aspect_maps")
        plt.close()
        
        # --- Affichage de la classification du terrain ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
        fig.suptitle("Classification du Terrain", fontsize=16)

        # Échelle fixe car ce sont des catégories
        im1 = axes[0].imshow(self.terrain_class_gt, cmap='tab10', vmin=0, vmax=4)
        axes[0].set_title('Classification Terrain GT')
        
        im2 = axes[1].imshow(self.terrain_class_pred, cmap='tab10', vmin=0, vmax=4)
        axes[1].set_title('Classification Terrain Prédite')
        
        self.save_figure(fig, "06_terrain_classification")
        plt.close()

    def save_complete_report(self):
        print("Saving complete report...")
        report_file = self.output_dir / "evaluation_report.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RAPPORT D'ÉVALUATION ULTRA-COMPLET MASt3R\n")
            f.write("="*80 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Images: {self.name} - {self.img2.stem}\n")
            f.write(f"Checkpoint: {self.ckpt_path}\n")
            f.write(f"Output directory: {self.output_dir}\n\n")

            # Métriques géométriques absolues
            f.write("--- MÉTRIQUES GÉOMÉTRIQUES ABSOLUES ---\n")
            f.write(f"Scale                       : {self._fmt(self.metrics.get('scale', 1.0), '.6f')}\n")
            f.write(f"Scale error (%)             : {self._fmt(self.metrics.get('scale_error_pct', 0.0), '.2f')}%\n")
            f.write(f"Rotation error (°)          : {self._fmt(self.metrics.get('rotation_error_deg', 0.0), '.4f')}\n")
            f.write(f"Translation error (m)       : {self._fmt(self.metrics.get('centroid_diff', 0.0), '.3f')}\n")
            f.write(f"RMSE (m)                    : {self._fmt(self.metrics.get('rmse', 0.0), '.3f')}\n")
            f.write(f"MAE (m)                     : {self._fmt(self.metrics.get('mae', 0.0), '.3f')}\n")
            f.write(f"Pearson correlation         : {self._fmt(self.metrics.get('pearson_r', 0.0), '.4f')}\n\n")

            # Métriques géométriques relatives
            f.write("--- MÉTRIQUES GÉOMÉTRIQUES RELATIVES ---\n")
            f.write(f"Terrain span 3D (m)         : {self._fmt(self.metrics.get('terrain_span_3d', 0.0), '.2f')}\n")
            f.write(f"Terrain elevation range (m) : {self._fmt(self.metrics.get('terrain_elevation_range', 0.0), '.2f')}\n")
            f.write(f"Terrain std total (m)       : {self._fmt(self.metrics.get('terrain_std_total', 0.0), '.2f')}\n")
            f.write(f"RMSE relative (%)           : {self._fmt(self.metrics.get('rmse_relative_span', 0.0), '.3f')}%\n")
            f.write(f"MAE relative (%)            : {self._fmt(self.metrics.get('mae_relative_span', 0.0), '.3f')}%\n")
            f.write(f"Centroid diff relative (%)  : {self._fmt(self.metrics.get('centroid_diff_relative_span', 0.0), '.3f')}%\n\n")

            # Métriques Chamfer
            f.write("--- MÉTRIQUES CHAMFER ABSOLUES ---\n")
            f.write(f"Chamfer Distance (m)        : {self._fmt(self.metrics.get('chamfer_distance', 0.0), '.4f')}\n")
            f.write(f"Pred→GT (m)                 : {self._fmt(self.metrics.get('chamfer_pred_to_gt', 0.0), '.4f')}\n")
            f.write(f"GT→Pred (m)                 : {self._fmt(self.metrics.get('chamfer_gt_to_pred', 0.0), '.4f')}\n")
            f.write(f"Hausdorff (m)               : {self._fmt(self.metrics.get('hausdorff_distance', 0.0), '.4f')}\n\n")

            f.write("--- MÉTRIQUES CHAMFER RELATIVES ---\n")
            f.write(f"Chamfer rel. (%)            : {self._fmt(self.metrics.get('chamfer_distance_relative_span', 0.0), '.4f')}%\n")
            f.write(f"Hausdorff rel. (%)          : {self._fmt(self.metrics.get('hausdorff_distance_relative_span', 0.0), '.4f')}%\n\n")

            # Couverture Chamfer
            f.write("--- COUVERTURE CHAMFER ---\n")
            for t in [10.0, 20.0, 30.0, 40.0]:
                p = self._fmt(self.metrics.get(f'coverage_pred_to_gt_{t}m', 0.0), '.1f')
                g = self._fmt(self.metrics.get(f'coverage_gt_to_pred_{t}m', 0.0), '.1f')
                f.write(f"Points < {int(t)}m : Pred→GT={p}% | GT→Pred={g}%\n")
            f.write("\n")

            # SSIM (per-view + averaged)
            f.write("--- MÉTRIQUES SSIM ---\n")
            f.write(f"SSIM view 0   : {self._fmt(self.metrics.get('v0_ssim_adaptive', 0.0), '.3f')}\n")
            f.write(f"SSIM view 1   : {self._fmt(self.metrics.get('v1_ssim_adaptive', 0.0), '.3f')}\n")
            f.write(f"SSIM avg      : {self._fmt(self.metrics.get('ssim_adaptive_avg', 0.0), '.3f')}\n")
            f.write("\n")

            # Regression (per-view + averaged)
            f.write("--- RÉGRESSION LINÉAIRE ---\n")
            for vx in ['v0', 'v1']:
                if f'{vx}_regression_slope' in self.metrics:
                    f.write(f"[{vx}] Slope     : {self._fmt(self.metrics.get(f'{vx}_regression_slope', 0.0), '.4f')}\n")
                    f.write(f"[{vx}] R-squared : {self._fmt(self.metrics.get(f'{vx}_regression_r2', 0.0), '.4f')}\n")
            if 'regression_slope_avg' in self.metrics:
                f.write(f"[avg] Slope     : {self._fmt(self.metrics.get('regression_slope_avg', 0.0), '.4f')}\n")
                f.write(f"[avg] R-squared : {self._fmt(self.metrics.get('regression_r2_avg', 0.0), '.4f')}\n")
            f.write("\n")

            # Métriques de terrain
            if self.terrain_metrics:
                f.write("--- MÉTRIQUES DE RELIEF ---\n")
                f.write(f"Pente RMSE (°)              : {self._fmt(self.terrain_metrics.get('slope_rmse', 0.0), '.3f')}\n")
                f.write(f"Pente MAE (°)               : {self._fmt(self.terrain_metrics.get('slope_mae', 0.0), '.3f')}\n")
                f.write(f"Pente corrélation           : {self._fmt(self.terrain_metrics.get('slope_corr', 0.0), '.4f')}\n")
                if 'slope_corr_pointcloud' in self.terrain_metrics:
                    f.write(f"Pente corr. (nuage 3D)      : {self._fmt(self.terrain_metrics.get('slope_corr_pointcloud', 0.0), '.4f')}\n")
                if 'curv_rmse' in self.terrain_metrics:
                    f.write(f"Courbure RMSE               : {self._fmt(self.terrain_metrics.get('curv_rmse', 0.0), '.6f')}\n")
                    f.write(f"Courbure MAE                : {self._fmt(self.terrain_metrics.get('curv_mae', 0.0), '.6f')}\n")
                    f.write(f"Courbure corrélation        : {self._fmt(self.terrain_metrics.get('curv_corr', 0.0), '.4f')}\n")
                f.write(f"Rugosité RMSE               : {self._fmt(self.terrain_metrics.get('rough_rmse', 0.0), '.6f')}\n")
                f.write(f"Rugosité MAE                : {self._fmt(self.terrain_metrics.get('rough_mae', 0.0), '.6f')}\n")
                f.write(f"Rugosité corrélation        : {self._fmt(self.terrain_metrics.get('rough_corr', 0.0), '.4f')}\n\n")

            # Métriques de profils
            if self.profile_metrics:
                f.write("--- MÉTRIQUES DE PROFILS ---\n")
                f.write(f"Profil central MSE          : {self._fmt(self.profile_metrics.get('central_mse', 0.0), '.6f')}\n")
                f.write(f"Profil central MAE          : {self._fmt(self.profile_metrics.get('central_mae', 0.0), '.3f')}\n")
                f.write(f"Profil central corrélation  : {self._fmt(self.profile_metrics.get('central_corr', 0.0), '.4f')}\n")
                
                if 'multiple_count' in self.profile_metrics:
                    f.write(f"Profils multiples (count)   : {self.profile_metrics.get('multiple_count', 0)}\n")
                    f.write(f"Profils MSE moyenne         : {self._fmt(self.profile_metrics.get('multiple_mse_mean', 0.0), '.6f')}\n")
                    f.write(f"Profils MAE moyenne         : {self._fmt(self.profile_metrics.get('multiple_mae_mean', 0.0), '.3f')}\n")
                    f.write(f"Profils corr. moyenne       : {self._fmt(self.profile_metrics.get('multiple_corr_mean', 0.0), '.4f')}\n")
                f.write("\n")

            # Statistiques d'erreur
            if self.error_stats:
                f.write("--- STATISTIQUES D'ERREUR ---\n")
                f.write(f"Erreur moyenne (m)          : {self._fmt(self.error_stats.get('mean', 0.0), '.3f')}\n")
                f.write(f"Erreur écart-type (m)       : {self._fmt(self.error_stats.get('std', 0.0), '.3f')}\n")
                f.write(f"Erreur médiane (m)          : {self._fmt(self.error_stats.get('median', 0.0), '.3f')}\n")
                f.write(f"Erreur minimale (m)         : {self._fmt(self.error_stats.get('min', 0.0), '.3f')}\n")
                f.write(f"Erreur maximale (m)         : {self._fmt(self.error_stats.get('max', 0.0), '.3f')}\n")
                f.write(f"Pixels valides              : {self.error_stats.get('count', 0)}\n\n")

            # RRA/RTA si disponibles
            if 'RRA_deg' in self.metrics:
                f.write("--- MÉTRIQUES DE POSE RELATIVE ---\n")
                f.write(f"RRA (erreur rotation) (°)   : {self._fmt(self.metrics.get('RRA_deg', 0.0), '.3f')}\n")
                f.write(f"RTA (erreur translation)(°) : {self._fmt(self.metrics.get('RTA_deg', 0.0), '.3f')}\n\n")

            # Résumé final
            f.write("="*40 + "\nFIN DU RAPPORT\n" + "="*40 + "\n")

        print(f"✓ Rapport complet sauvegardé: {report_file}")

    def run_complete_evaluation(self):
        """Exécute l'évaluation complète"""
        print("\n=== DÉMARRAGE ÉVALUATION COMPLÈTE MASt3R ===")

        try:
            # Étapes principales
            self.load_ground_truth()
            self.run_mast3r_inference()
            if self.use_sparse_ga:
                self.reconstruct_with_sparse_ga()
                if not getattr(self, 'skip_stitch', False):
                    self.stitch_views_from_matches()
            self.prepare_data()
            self.perform_alignment()

            # Calculs des métriques
            self.compute_geometric_metrics()
            self.compute_terrain_metrics()
            self.compute_profile_metrics()
            self.compute_error_statistics()
            self.compute_slope_correlation()

            # Visualisations
            if self.save_viz:
                self.generate_visualizations()
                self.save_complete_report()
                print(f"\n✅ ÉVALUATION TERMINÉE")
                print(f"📁 Résultats dans: {self.output_dir}")
                print(f"📊 {len(list(self.output_dir.glob('*.png')))} visualisations générées")
                print(f"📄 Rapport: {self.output_dir / 'evaluation_report.txt'}")
            else:
                print(f"\n✅ ÉVALUATION TERMINÉE (visualisations désactivées)")

            return self.metrics, self.terrain_metrics, self.profile_metrics, self.error_stats

        except Exception as e:
            print(f"❌ Erreur dans run_complete_evaluation: {e}")
            import traceback
            traceback.print_exc()
            raise
