#!/usr/bin/env python3
"""
Compare ALL BlendedMVS student models across training regimes:
  - GT baseline (trained with ground truth, no teacher)
  - Distillation baseline (general_s*)
  - Improved distillation v1 (CGFD / RFD variants)
  - Improved distillation v3 (TeacherConf + gamma=0.01)

Also evaluates the teacher (MASt3R ViT-L) as upper bound.

Metrics per pair:
  - Depth: abs_rel, RMSE, delta<1.25
  - 3D pts: mean L2 after Sim(3) Umeyama alignment
  - Matching: reprojection VCRE (px)

Usage:
  python evaluation/blendedmvs/eval_compare_all_bmvs.py --max_pairs 200
  python evaluation/blendedmvs/eval_compare_all_bmvs.py --groups gt v3 --max_pairs 100
  python evaluation/blendedmvs/eval_compare_all_bmvs.py --only v3_tcr v3_tcr_rfd teacher

SLURM:
  sbatch scripts/eval/eval_compare_all_bmvs.sh
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict, OrderedDict

# ── Ensure project root is on sys.path ──────────────────────────────
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── MASt3R imports ──────────────────────────────────────────────────
import mast3r.utils.path_to_dust3r  # noqa
from mast3r.model import AsymmetricMASt3R
from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

# ── Student builders ────────────────────────────────────────────────
from distillation_dual import build_vit_student, build_vit_tiny_student

# ── Dataset ─────────────────────────────────────────────────────────
BMVS_ROOT = "Datas/blendedmvs_processed"
TEACHER_CKPT = "mast3r/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"


# =====================================================================
#  ALL STUDENT CONFIGS — grouped by training regime
# =====================================================================
STUDENT_CONFIGS = OrderedDict()

# ── Group: GT baseline (no teacher, trained with GT pointmaps) ──────
STUDENT_CONFIGS["gt_s2_full"] = dict(
    group="gt", label="GT S2-full (512d/6L)",
    builder=build_vit_student,
    kwargs=dict(dec_embed_dim=512, dec_depth=6, dec_num_heads=4, mlp_ratio=1.0,
                prefer_dinov2=True, backbone_type="dinov2"),
    ckpt="output/gt_bmvs_s2_full/checkpoint-50.pth",
)
STUDENT_CONFIGS["gt_s4_reduced"] = dict(
    group="gt", label="GT S4-reduced (256d/4L)",
    builder=build_vit_student,
    kwargs=dict(dec_embed_dim=256, dec_depth=4, dec_num_heads=4, mlp_ratio=1.0,
                prefer_dinov2=True, backbone_type="dinov2"),
    ckpt="output/gt_bmvs_s4_reduced/checkpoint-50.pth",
)

# ── Group: Distillation baseline (SVD + CosineFeat + GradSmooth, 50ep)
#    train_general_distillation.sh — 7 students, StdConf
#    S2: full (512d/6L), SVD+Feat+Grad
#    S4: reduced (256d/4L), SVD+Feat+Grad
#    S5: ViT-Tiny encoder, SVD+Feat+Grad
#    S6: no-SVD ablation (512d/6L), Feat+Grad only
#    S7: no-Feat ablation (512d/6L), SVD+Grad only
STUDENT_CONFIGS["distill_s2_full"] = dict(
    group="distill", label="Distill S2 SVD+Feat+Grad",
    builder=build_vit_student,
    kwargs=dict(dec_embed_dim=512, dec_depth=6, dec_num_heads=4, mlp_ratio=1.0,
                prefer_dinov2=True, backbone_type="dinov2"),
    ckpt="output/general_s2_full/checkpoint-50.pth",
)
STUDENT_CONFIGS["distill_s4_reduced"] = dict(
    group="distill", label="Distill S4 reduced",
    builder=build_vit_student,
    kwargs=dict(dec_embed_dim=256, dec_depth=4, dec_num_heads=4, mlp_ratio=1.0,
                prefer_dinov2=True, backbone_type="dinov2"),
    ckpt="output/general_s4_reduced/checkpoint-50.pth",
)
STUDENT_CONFIGS["distill_s5_tiny"] = dict(
    group="distill", label="Distill S5 ViT-Tiny",
    builder=build_vit_tiny_student,
    kwargs=dict(dec_embed_dim=512, dec_depth=6, dec_num_heads=4, mlp_ratio=1.0),
    ckpt="output/general_s5_vit_tiny/checkpoint-50.pth",
)
STUDENT_CONFIGS["distill_s6_no_svd"] = dict(
    group="distill", label="Distill S6 no-SVD",
    builder=build_vit_student,
    kwargs=dict(dec_embed_dim=512, dec_depth=6, dec_num_heads=4, mlp_ratio=1.0,
                prefer_dinov2=True, backbone_type="dinov2"),
    ckpt="output/general_s6_no_svd/checkpoint-50.pth",
)
STUDENT_CONFIGS["distill_s7_no_feat"] = dict(
    group="distill", label="Distill S7 no-Feat",
    builder=build_vit_student,
    kwargs=dict(dec_embed_dim=512, dec_depth=6, dec_num_heads=4, mlp_ratio=1.0,
                prefer_dinov2=True, backbone_type="dinov2"),
    ckpt="output/general_s7_no_feat/checkpoint-50.pth",
)
STUDENT_CONFIGS["s7_frozen_enc"] = dict(
    group="distill", label="S7 frozen-encoder (SVD+Grad)",
    builder=build_vit_student,
    kwargs=dict(dec_embed_dim=512, dec_depth=6, dec_num_heads=4, mlp_ratio=1.0,
                prefer_dinov2=True, backbone_type="dinov2"),
    ckpt="output/s7_frozen_enc/checkpoint-last.pth",
)

STUDENT_CONFIGS["bmvs_s2_frozen"] = dict(
    group="distill", label="S2 frozen-encoder (7 GPU, fair vs general_s2_full)",
    builder=build_vit_student,
    kwargs=dict(dec_embed_dim=512, dec_depth=6, dec_num_heads=4, mlp_ratio=1.0,
                prefer_dinov2=True, backbone_type="dinov2"),
    ckpt="output/bmvs_s2_frozen/checkpoint-50.pth",
)

STUDENT_CONFIGS["bmvs_s2_no_svd_no_feat"] = dict(
    group="distill", label="B3: no SVD, no feat, unfrozen (7 GPU)",
    builder=build_vit_student,
    kwargs=dict(dec_embed_dim=512, dec_depth=6, dec_num_heads=4, mlp_ratio=1.0,
                prefer_dinov2=True, backbone_type="dinov2"),
    ckpt="output/bmvs_s2_no_svd_no_feat/checkpoint-50.pth",
)
STUDENT_CONFIGS["bmvs_s2_frozen_no_feat"] = dict(
    group="distill", label="B5: SVD, no feat, frozen (7 GPU)",
    builder=build_vit_student,
    kwargs=dict(dec_embed_dim=512, dec_depth=6, dec_num_heads=4, mlp_ratio=1.0,
                prefer_dinov2=True, backbone_type="dinov2"),
    ckpt="output/bmvs_s2_frozen_no_feat/checkpoint-50.pth",
)
STUDENT_CONFIGS["bmvs_s3_reduced"] = dict(
    group="distill", label="Paper S3: ViT-S reduced decoder, SVD+no feat (7 GPU)",
    builder=build_vit_student,
    kwargs=dict(dec_embed_dim=256, dec_depth=4, dec_num_heads=4, mlp_ratio=1.0,
                prefer_dinov2=True, backbone_type="dinov2"),
    ckpt="output/bmvs_s3_reduced/checkpoint-50.pth",
)
STUDENT_CONFIGS["bmvs_s4_vit_tiny"] = dict(
    group="distill", label="Paper S4: ViT-Tiny, SVD+no feat (7 GPU)",
    builder=build_vit_tiny_student,
    kwargs=dict(dec_embed_dim=512, dec_depth=6, dec_num_heads=4, mlp_ratio=1.0),
    ckpt="output/bmvs_s4_vit_tiny/checkpoint-50.pth",
)

# ── Group: v4 — Ablation around S7 best baseline ────────────────────
#    train_ablation_v4.sh
#    A: S7-reduced   — DINOv2 ViT-S, 256d/4L, SVD+Grad, no Feat
#    B: S7-vit-tiny  — ViT-Tiny,     512d/6L, SVD+Grad, no Feat
#    C: Pure Grad    — DINOv2 ViT-S, 512d/6L, NO SVD, no Feat, Grad only
#    D: S7+RFD       — DINOv2 ViT-S, 512d/6L, SVD+Grad+RFD, no Feat
STUDENT_CONFIGS["v4_A_reduced"] = dict(
    group="v4", label="v4 S7-reduced (256d/4L)",
    builder=build_vit_student,
    kwargs=dict(dec_embed_dim=256, dec_depth=4, dec_num_heads=4, mlp_ratio=1.0,
                prefer_dinov2=True, backbone_type="dinov2"),
    ckpt="output/v4_A_s7_reduced/checkpoint-50.pth",
)
STUDENT_CONFIGS["v4_B_vit_tiny"] = dict(
    group="v4", label="v4 S7-ViT-Tiny",
    builder=build_vit_tiny_student,
    kwargs=dict(dec_embed_dim=512, dec_depth=6, dec_num_heads=4, mlp_ratio=1.0),
    ckpt="output/v4_B_s7_vit_tiny/checkpoint-50.pth",
)
STUDENT_CONFIGS["v4_C_pure_grad"] = dict(
    group="v4", label="v4 Pure Grad (no SVD)",
    builder=build_vit_student,
    kwargs=dict(dec_embed_dim=512, dec_depth=6, dec_num_heads=4, mlp_ratio=1.0,
                prefer_dinov2=True, backbone_type="dinov2"),
    ckpt="output/v4_C_pure_grad/checkpoint-50.pth",
)
STUDENT_CONFIGS["v4_D_s7_rfd"] = dict(
    group="v4", label="v4 SVD+Grad+RFD",
    builder=build_vit_student,
    kwargs=dict(dec_embed_dim=512, dec_depth=6, dec_num_heads=4, mlp_ratio=1.0,
                prefer_dinov2=True, backbone_type="dinov2"),
    ckpt="output/v4_D_s7_rfd/checkpoint-50.pth",
)



# ── Group: v4ef — SVD+Grad+RFD on lighter architectures ─────────────
#    train_bmvs_v4ef_rfd.sh
#    E: ViT-Tiny,     512d/6L, SVD+Grad+RFD  (lighter encoder, same decoder as D)
#    F: DINOv2 ViT-S, 256d/4L, SVD+Grad+RFD  (same encoder as D, reduced decoder)
STUDENT_CONFIGS["v4_E_tiny_rfd"] = dict(
    group="v4ef", label="v4 E ViT-Tiny 512d/6L SVD+Grad+RFD",
    builder=build_vit_tiny_student,
    kwargs=dict(dec_embed_dim=512, dec_depth=6, dec_num_heads=4, mlp_ratio=1.0),
    ckpt="output/v4_E_tiny_rfd/checkpoint-50.pth",
)
STUDENT_CONFIGS["v4_F_reduced_rfd"] = dict(
    group="v4ef", label="v4 F DINOv2 256d/4L SVD+Grad+RFD",
    builder=build_vit_student,
    kwargs=dict(dec_embed_dim=256, dec_depth=4, dec_num_heads=4, mlp_ratio=1.0,
                prefer_dinov2=True, backbone_type="dinov2"),
    ckpt="output/v4_F_reduced_rfd/checkpoint-50.pth",
)


# =====================================================================
#  Metrics
# =====================================================================
def depth_metrics(pred_depth, gt_depth, valid_mask):
    """Standard depth metrics on valid pixels (with median scale alignment)."""
    pred = pred_depth[valid_mask]
    gt = gt_depth[valid_mask]
    if len(gt) < 10:
        return {}
    scale = np.median(gt) / (np.median(pred) + 1e-8)
    pred = pred * scale

    thresh = np.maximum(gt / (pred + 1e-8), pred / (gt + 1e-8))
    delta1 = (thresh < 1.25).mean()
    abs_rel = np.mean(np.abs(gt - pred) / (gt + 1e-8))
    rmse = np.sqrt(np.mean((gt - pred) ** 2))

    return dict(abs_rel=abs_rel, rmse=rmse, delta1=delta1, scale=scale)


def umeyama_alignment(src, dst):
    """Sim(3) alignment: dst ~ s*R@src + t."""
    n, d = src.shape
    mu_s, mu_d = src.mean(0), dst.mean(0)
    src_c, dst_c = src - mu_s, dst - mu_d
    sigma_src = np.mean(np.sum(src_c ** 2, axis=1))
    cov = dst_c.T @ src_c / n
    U, S, Vt = np.linalg.svd(cov)
    det_sign = np.linalg.det(U) * np.linalg.det(Vt)
    D = np.eye(d)
    if det_sign < 0:
        D[-1, -1] = -1
    R = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / (sigma_src + 1e-12)
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def pts3d_metrics(pred_pts, gt_pts, valid_mask, n_align=5000):
    """L2 error after Sim(3) alignment."""
    pred = pred_pts[valid_mask]
    gt = gt_pts[valid_mask]
    if len(gt) < 50:
        return {}
    rng = np.random.default_rng(0)
    n = min(n_align, len(gt))
    idx = rng.choice(len(gt), n, replace=False)
    s, R, t = umeyama_alignment(pred[idx], gt[idx])
    aligned = s * (pred @ R.T) + t
    errors = np.linalg.norm(aligned - gt, axis=1)
    return dict(
        pts3d_mean=errors.mean(),
        pts3d_median=np.median(errors),
        pts3d_p90=np.percentile(errors, 90),
    )


def mutual_nn_matches(desc1, desc2, conf1, conf2, conf_thresh=2.0, max_n=4096):
    """GPU mutual-NN on L2-normalised descriptors. Returns (pts_2d_1, pts_2d_2) in pixels."""
    H, W, D = desc1.shape
    valid1 = (conf1 > conf_thresh).ravel()
    valid2 = (conf2 > conf_thresh).ravel()
    idx1 = np.where(valid1)[0]
    idx2 = np.where(valid2)[0]
    if len(idx1) < 8 or len(idx2) < 8:
        return None, None

    # Subsample BEFORE matmul to avoid OOM (matmul is M1×M2 — must stay in VRAM)
    rng = np.random.default_rng(0)
    if len(idx1) > max_n:
        idx1 = rng.choice(idx1, max_n, replace=False)
    if len(idx2) > max_n:
        idx2 = rng.choice(idx2, max_n, replace=False)

    d1 = desc1.reshape(-1, D)[idx1].astype(np.float32)
    d2 = desc2.reshape(-1, D)[idx2].astype(np.float32)
    d1 /= np.linalg.norm(d1, axis=1, keepdims=True) + 1e-8
    d2 /= np.linalg.norm(d2, axis=1, keepdims=True) + 1e-8

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    t1 = torch.from_numpy(d1).to(dev)
    t2 = torch.from_numpy(d2).to(dev)
    sim = t1 @ t2.T
    nn12 = sim.argmax(dim=1)
    nn21 = sim.argmax(dim=0)
    mutual = (nn21[nn12] == torch.arange(len(idx1), device=dev))
    m1 = torch.where(mutual)[0].cpu().numpy()
    m2 = nn12[mutual].cpu().numpy()
    if len(m1) < 8:
        return None, None

    # subsample for speed
    if len(m1) > max_n:
        sel = np.random.default_rng(0).choice(len(m1), max_n, replace=False)
        m1, m2 = m1[sel], m2[sel]

    rows1, cols1 = idx1[m1] // W, idx1[m1] % W
    rows2, cols2 = idx2[m2] // W, idx2[m2] % W
    pts1 = np.stack([cols1, rows1], axis=1).astype(np.float64)
    pts2 = np.stack([cols2, rows2], axis=1).astype(np.float64)
    return pts1, pts2


def pose_from_essential(desc1, desc2, K, conf1, conf2, conf_thresh=2.0):
    """Estimate relative pose via Essential Matrix on descriptor mutual-NN matches.
    pts_2d_1 (cam0 pixels) ↔ pts_2d_2 (cam1 pixels) — proper cross-view correspondences.
    Returns emat_rra, emat_inliers, and (R, t) for VCRE. Returns {} if desc unavailable."""
    import cv2
    if desc1 is None or desc2 is None:
        return {}

    pts_2d_1, pts_2d_2 = mutual_nn_matches(desc1, desc2, conf1, conf2,
                                            conf_thresh=conf_thresh)
    if pts_2d_1 is None or len(pts_2d_1) < 8:
        return {}

    E, inlier_mask = cv2.findEssentialMat(
        pts_2d_1, pts_2d_2, K, method=cv2.RANSAC, threshold=1.0, prob=0.999)
    if E is None:
        return {}
    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    _, R, t, _ = cv2.recoverPose(E, pts_2d_1, pts_2d_2, K, mask=inlier_mask)

    cos_angle = np.clip((np.trace(R) - 1) / 2, -1, 1)
    rra = np.degrees(np.arccos(cos_angle))
    return dict(emat_rra=rra, emat_inliers=n_inliers, _R_pred=R, _t_pred=t.ravel())


def vcre_from_relative_pose(R_pred, t_pred_unit, R_gt, t_gt, K,
                            image_size=(512, 384), n_per_axis=6):
    """VCRE (Virtual Correspondence Reprojection Error) — Arnold et al., ECCV 2022.

    Places a cube of virtual 3D points in front of the cameras, projects them
    with both the GT and predicted relative poses, and returns the median
    reprojection error in pixels.  t_pred_unit is scaled to ||t_gt||.
    """
    W, H = image_size
    diag = np.sqrt(W ** 2 + H ** 2)

    # Scale predicted translation to GT translation norm
    t_gt_norm = np.linalg.norm(t_gt)
    t_pred = t_pred_unit * t_gt_norm  # (3,)

    # Virtual 3D cube: centred in front of cam0 at depth ~ mean scene depth
    depth_ref = max(t_gt_norm * 5.0, 1.0)
    lin = np.linspace(-depth_ref * 0.3, depth_ref * 0.3, n_per_axis)
    gx, gy, gz = np.meshgrid(lin, lin, lin)
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel() + depth_ref], axis=-1)  # (N, 3)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def project(R, t, pts3d):
        """Project pts3d (N,3) in cam0 frame into cam1 pixels via relative pose R,t."""
        p = (pts3d @ R.T) + t[None]  # (N, 3) in cam1 frame
        z = p[:, 2]
        valid = z > 1e-3
        u = np.where(valid, p[:, 0] * fx / (z + 1e-8) + cx, np.nan)
        v = np.where(valid, p[:, 1] * fy / (z + 1e-8) + cy, np.nan)
        return np.stack([u, v], axis=1)

    px_gt   = project(R_gt,   t_gt,   pts)
    px_pred = project(R_pred, t_pred, pts)

    err = np.linalg.norm(px_gt - px_pred, axis=1)
    valid = np.isfinite(err)
    if valid.sum() == 0:
        return {}
    err = err[valid]
    return dict(
        vcre_median_px=float(np.median(err)),
        vcre_mean_px=float(np.mean(err)),
        vcre_pct=float(np.median(err) / diag * 100),
    )


# =====================================================================
#  Model loading
# =====================================================================
def load_teacher(device):
    print(f"[Teacher] Loading {TEACHER_CKPT}")
    model = AsymmetricMASt3R.from_pretrained(TEACHER_CKPT).to(device)
    model.eval()
    return model


def load_student(name, cfg, device):
    ckpt_path = cfg['ckpt']
    if not os.path.isfile(ckpt_path):
        print(f"  [{name}] checkpoint not found: {ckpt_path}, skipping")
        return None
    student = cfg['builder'](device=device, **cfg['kwargs'])
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model', ckpt)
    missing, unexpected = student.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [{name}] Missing keys: {len(missing)}")
    student.eval()
    epoch = ckpt.get('epoch', '?')
    print(f"  [{name}] Loaded from {ckpt_path} (epoch {epoch})")
    return student


# =====================================================================
#  Inference
# =====================================================================
def run_inference(model, img1_path, img2_path, device, resolution=(512, 384)):
    imgs = load_images([img1_path, img2_path], size=resolution[0])
    output = inference([tuple(imgs)], model, device, batch_size=1, verbose=False)
    pred1, pred2 = output['pred1'], output['pred2']
    pts3d_1 = to_numpy(pred1['pts3d'].squeeze(0))
    pts3d_2 = to_numpy(pred2['pts3d_in_other_view'].squeeze(0))
    conf1   = to_numpy(pred1['conf'].squeeze(0))
    conf2   = to_numpy(pred2['conf'].squeeze(0))
    # descriptors for proper matching (24-dim, may be absent for GT-trained models)
    desc1 = to_numpy(pred1['desc'].squeeze(0)) if 'desc' in pred1 else None
    desc2 = to_numpy(pred2['desc'].squeeze(0)) if 'desc' in pred2 else None
    return pts3d_1, pts3d_2, conf1, conf2, desc1, desc2


def run_inference_with_aligner(model, img1_path, img2_path, device, resolution=(512, 384),
                                niter=300):
    """Run MASt3R inference + PointCloudOptimizer global aligner.

    pts3d/conf/desc: standard single-pair inference — IDENTICAL to run_inference.
    Do NOT use the symmetric 2-pair inference for these: the model produces
    wildly different conf values in that batch context (max diff ~74), which
    corrupts the valid-pixel mask and therefore Chamfer / depth metrics.

    aligner_poses: separate symmetric inference + PointCloudOptimizer.
    The aligner MUST run outside torch.no_grad() — Adam needs autograd.

    Returns same shapes as run_inference PLUS aligner_poses (2, 4, 4).
    """
    # Step 1: standard single-pair inference — identical to run_inference
    with torch.no_grad():
        pts3d_1, pts3d_2, conf1, conf2, desc1, desc2 = run_inference(
            model, img1_path, img2_path, device, resolution)

    # Step 2: symmetric inference for aligner (poses only, no grad needed for fwd)
    images = load_images([img1_path, img2_path], size=resolution[0], verbose=False)
    img_pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    with torch.no_grad():
        output = inference(img_pairs, model, device, batch_size=1, verbose=False)

    # Aligner optimises via Adam — must be OUTSIDE torch.no_grad()
    ga = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    try:
        ga.compute_global_alignment(init='mst', niter=niter, schedule='cosine', lr=0.01)
    except Exception:
        pass  # fall back to MST initialisation
    aligner_poses = ga.get_im_poses().detach().cpu().numpy()  # (2, 4, 4) cam2world

    del ga
    return pts3d_1, pts3d_2, conf1, conf2, desc1, desc2, aligner_poses


# =====================================================================
#  Load test pairs
# =====================================================================
def load_test_pairs():
    """Load BlendedMVS test split pairs."""
    pairs_path = os.path.join(BMVS_ROOT, 'blendedmvs_pairs.npy')
    if not os.path.isfile(pairs_path):
        # Fallback: enumerate scenes and create sequential pairs
        print("No pairs file found, building pairs from dataset...")
        from dust3r.datasets.blendedmvs import BlendedMVS as BMVS
        ds = BMVS(split='val', ROOT=BMVS_ROOT, resolution=(512, 384), seed=777)
        pairs = []
        for i in range(len(ds)):
            view = ds[i]
            pairs.append((view['filepath'][0], view['filepath'][1]))
        return pairs

    pairs = np.load(pairs_path)
    existing = []
    for i in range(len(pairs)):
        seqh, seql = int(pairs[i]['seq_high']), int(pairs[i]['seq_low'])
        seq = f"{seqh:08x}{seql:016x}"
        seq_path = os.path.join(BMVS_ROOT, seq)
        if os.path.isdir(seq_path):
            existing.append(i)
    pairs = pairs[existing]
    test_mask = (pairs['seq_low'] % 10) == 1
    return pairs[test_mask]


# =====================================================================
#  Visualization helpers
# =====================================================================
def write_ply(path, points, colors=None, max_pts=200000):
    """Write colored PLY file (subsampled to max_pts)."""
    pts = points.reshape(-1, 3)
    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid]
    if colors is not None:
        colors = colors.reshape(-1, 3)[valid]
    # Subsample
    if len(pts) > max_pts:
        idx = np.random.default_rng(0).choice(len(pts), max_pts, replace=False)
        pts = pts[idx]
        if colors is not None:
            colors = colors[idx]
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(len(pts)):
            line = f"{pts[i,0]:.4f} {pts[i,1]:.4f} {pts[i,2]:.4f}"
            if colors is not None:
                line += f" {int(colors[i,0])} {int(colors[i,1])} {int(colors[i,2])}"
            f.write(line + "\n")


def save_depth_figure(out_path, img_rgb, gt_depth, pred_depth, gt_mask, view_name=""):
    """Save 4-panel depth comparison: input | GT | pred | error (matplotlib)."""

    # Scale-align pred to GT (median ratio)
    valid = gt_mask & (pred_depth > 0.01) & np.isfinite(pred_depth)
    if valid.sum() > 10:
        scale_ratio = np.median(gt_depth[valid]) / np.median(pred_depth[valid])
        pred_aligned = pred_depth * scale_ratio
    else:
        pred_aligned = pred_depth
        scale_ratio = 1.0

    # Common depth range from GT
    gt_valid = gt_depth[gt_mask]
    vmin = np.percentile(gt_valid, 2) if len(gt_valid) > 10 else 0
    vmax = np.percentile(gt_valid, 98) if len(gt_valid) > 10 else 10

    gt_vis = np.where(gt_mask, gt_depth, np.nan)
    pred_vis = np.where(gt_mask, pred_aligned, np.nan)
    error_map = np.abs(pred_aligned - gt_depth)
    error_map[~valid] = np.nan

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Input {view_name}")
    axes[0].axis('off')

    axes[1].imshow(gt_vis, cmap='turbo', vmin=vmin, vmax=vmax)
    axes[1].set_title("GT Depth")
    axes[1].axis('off')

    axes[2].imshow(pred_vis, cmap='turbo', vmin=vmin, vmax=vmax)
    axes[2].set_title(f"Pred Depth (x{scale_ratio:.2f})")
    axes[2].axis('off')

    emax = np.percentile(error_map[valid], 95) if valid.sum() > 10 else 1
    axes[3].imshow(error_map, cmap='hot', vmin=0, vmax=emax)
    axes[3].set_title("Abs Error")
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def backproject(depth, K, H_out, W_out):
    """Back-project depth map to 3D points using intrinsics K."""
    import cv2
    H_orig, W_orig = depth.shape[:2]
    if (H_orig, W_orig) != (H_out, W_out):
        depth = cv2.resize(depth, (W_out, H_out), interpolation=cv2.INTER_NEAREST)
    sx, sy = W_out / W_orig, H_out / H_orig
    fx, fy = K[0, 0] * sx, K[1, 1] * sy
    cx, cy = K[0, 2] * sx, K[1, 2] * sy
    u, v = np.meshgrid(np.arange(W_out), np.arange(H_out))
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    return np.stack([x, y, depth], axis=-1)


# =====================================================================
#  Evaluate one model
# =====================================================================
def evaluate_model(model, model_name, pairs, device, max_pairs=200,
                   save_vis=False, vis_dir=None, use_aligner=False,
                   common_mask=False, fair_align=False):
    import cv2
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

    all_metrics = defaultdict(list)
    n = min(max_pairs, len(pairs))
    rng = np.random.default_rng(42)
    indices = rng.choice(len(pairs), n, replace=False) if n < len(pairs) else np.arange(n)

    t0 = time.time()
    for step, pair_idx in enumerate(indices):
        try:
            seqh = int(pairs[pair_idx]['seq_high'])
            seql = int(pairs[pair_idx]['seq_low'])
            seq = f"{seqh:08x}{seql:016x}"
            img1_idx = int(pairs[pair_idx]['img1'])
            img2_idx = int(pairs[pair_idx]['img2'])
        except (IndexError, ValueError):
            continue

        seq_path = os.path.join(BMVS_ROOT, seq)
        img1_name = f"{img1_idx:08d}"
        img2_name = f"{img2_idx:08d}"
        img1_path = os.path.join(seq_path, img1_name + ".jpg")
        img2_path = os.path.join(seq_path, img2_name + ".jpg")

        if not os.path.isfile(img1_path) or not os.path.isfile(img2_path):
            continue

        # Load GT for both views
        try:
            gt1 = np.load(os.path.join(seq_path, img1_name + ".npz"))
            gt2 = np.load(os.path.join(seq_path, img2_name + ".npz"))
            gt_depth1 = cv2.imread(
                os.path.join(seq_path, img1_name + ".exr"),
                cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
            gt_depth2 = cv2.imread(
                os.path.join(seq_path, img2_name + ".exr"),
                cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
            if gt_depth1 is None or gt_depth2 is None:
                continue
        except Exception:
            continue

        # Run inference (+ optional global aligner for pose recovery)
        aligner_poses = None
        try:
            with torch.no_grad():
                if use_aligner:
                    pts3d_1, pts3d_2, conf1, conf2, desc1, desc2, aligner_poses = \
                        run_inference_with_aligner(model, img1_path, img2_path, device)
                else:
                    pts3d_1, pts3d_2, conf1, conf2, desc1, desc2 = run_inference(
                        model, img1_path, img2_path, device)
        except Exception as e:
            print(f"  [{model_name}] Inference error pair {step}: {e}")
            continue

        Hp, Wp = pts3d_1.shape[:2]
        H_gt, W_gt = gt_depth1.shape[:2]

        # ── Resize GT depth + backproject → world frame ──
        gt_depth1_r = cv2.resize(gt_depth1, (Wp, Hp), interpolation=cv2.INTER_NEAREST)
        gt_depth2_r = cv2.resize(gt_depth2, (Wp, Hp), interpolation=cv2.INTER_NEAREST)
        K1 = np.float32(gt1['intrinsics'])
        K2 = np.float32(gt2['intrinsics'])
        # Backproject in camera frame
        gt_pts3d_1_cam = backproject(gt_depth1, K1, Hp, Wp)  # (Hp, Wp, 3) in cam1
        gt_pts3d_2_cam = backproject(gt_depth2, K2, Hp, Wp)  # (Hp, Wp, 3) in cam2
        # Transform to world frame so both views are aligned
        R1 = np.float64(gt1['R_cam2world'])  # (3,3)
        t1 = np.float64(gt1['t_cam2world'])  # (3,)
        R2 = np.float64(gt2['R_cam2world'])
        t2 = np.float64(gt2['t_cam2world'])
        gt_pts3d_1 = (gt_pts3d_1_cam.reshape(-1, 3) @ R1.T + t1).reshape(Hp, Wp, 3)
        gt_pts3d_2 = (gt_pts3d_2_cam.reshape(-1, 3) @ R2.T + t2).reshape(Hp, Wp, 3)

        # ── Masks: valid GT depth + confident predictions ──
        gt_mask1 = (gt_depth1_r > 0.01) & (gt_depth1_r < 1000)
        gt_mask2 = (gt_depth2_r > 0.01) & (gt_depth2_r < 1000)
        # Three mask regimes:
        #   default         : align_mask = metric_mask = gt_mask & finite & (conf>1.5)
        #   --common_mask   : align_mask = metric_mask = gt_mask & finite
        #                     (align gets noisy because low-conf outliers drag the fit)
        #   --fair_align    : align_mask = gt_mask & finite & (conf>1.5)  ← best per-model fit
        #                     metric_mask = gt_mask & finite              ← same pixels for all
        #                     → best of both worlds: clean Sim(3), no hiding on metrics.
        pred_finite1 = np.isfinite(pts3d_1).all(axis=-1)
        pred_finite2 = np.isfinite(pts3d_2).all(axis=-1)
        if fair_align:
            align_mask1 = gt_mask1 & pred_finite1 & (conf1 > 1.5)
            align_mask2 = gt_mask2 & pred_finite2 & (conf2 > 1.5)
            metric_mask1 = gt_mask1 & pred_finite1
            metric_mask2 = gt_mask2 & pred_finite2
        elif common_mask:
            align_mask1 = gt_mask1 & pred_finite1
            align_mask2 = gt_mask2 & pred_finite2
            metric_mask1 = align_mask1
            metric_mask2 = align_mask2
        else:
            align_mask1 = gt_mask1 & pred_finite1 & (conf1 > 1.5)
            align_mask2 = gt_mask2 & pred_finite2 & (conf2 > 1.5)
            metric_mask1 = align_mask1
            metric_mask2 = align_mask2
        # `mask1`/`mask2` are kept as aliases pointing to the METRIC mask so
        # downstream viz code (depth figure, PLY dump) uses the same pixels as
        # the metrics — the Sim(3) fit uses `align_mask*` below.
        mask1 = metric_mask1
        mask2 = metric_mask2

        # ── Sim(3) alignment (pred MASt3R frame → GT world frame) ──
        pred_flat1 = pts3d_1.reshape(-1, 3)
        gt_flat1 = gt_pts3d_1.reshape(-1, 3)
        mask_flat1 = mask1.reshape(-1)            # for metrics + viz
        align_flat1 = align_mask1.reshape(-1)     # for Umeyama fit only

        if align_flat1.sum() < 100:
            continue

        n_sub = min(5000, int(align_flat1.sum()))
        align_idx = np.where(align_flat1)[0]
        sub_idx = rng.choice(align_idx, n_sub, replace=False)
        s, R, t = umeyama_alignment(pred_flat1[sub_idx], gt_flat1[sub_idx])

        # Apply alignment to both views
        aligned1 = s * (pred_flat1 @ R.T) + t
        aligned2 = s * (pts3d_2.reshape(-1, 3) @ R.T) + t

        # ── 3D point metrics (both views combined) ──
        errors1 = np.linalg.norm(aligned1[mask_flat1] - gt_flat1[mask_flat1], axis=1)
        mask_flat2 = mask2.reshape(-1)
        gt_flat2 = gt_pts3d_2.reshape(-1, 3)
        errors2 = np.linalg.norm(aligned2[mask_flat2] - gt_flat2[mask_flat2], axis=1) if mask_flat2.sum() > 0 else np.array([])
        errors_all = np.concatenate([errors1, errors2])

        all_metrics['pts3d_mean'].append(float(errors_all.mean()))
        all_metrics['pts3d_median'].append(float(np.median(errors_all)))
        all_metrics['pts3d_p90'].append(float(np.percentile(errors_all, 90)))

        # ── Chamfer / accuracy / completeness (view 1) ──
        try:
            from scipy.spatial import cKDTree
            metric_idx1 = np.where(mask_flat1)[0]
            n_ch = min(50000, len(metric_idx1))
            ch_idx = rng.choice(metric_idx1, n_ch, replace=False)
            tree_gt = cKDTree(gt_flat1[ch_idx])
            tree_pred = cKDTree(aligned1[ch_idx])
            d_pred2gt, _ = tree_gt.query(aligned1[ch_idx])
            d_gt2pred, _ = tree_pred.query(gt_flat1[ch_idx])
            accuracy = float(np.mean(d_pred2gt))
            completeness = float(np.mean(d_gt2pred))
            chamfer = (accuracy + completeness) / 2
            all_metrics['accuracy'].append(accuracy)
            all_metrics['completeness'].append(completeness)
            all_metrics['chamfer'].append(chamfer)
        except Exception:
            pass

        # ── Depth metrics (both views, transform aligned pts back to cam frame) ──
        # world2cam: p_cam = R_cam2world^T @ (p_world - t_cam2world)
        R1_inv, t1_inv = R1.T, -R1.T @ t1
        R2_inv, t2_inv = R2.T, -R2.T @ t2
        for view_i, (al_flat, gt_d_r, msk, Ri, ti) in enumerate([
            (aligned1, gt_depth1_r, mask1, R1_inv, t1_inv),
            (aligned2, gt_depth2_r, mask2, R2_inv, t2_inv),
        ]):
            al_cam = (al_flat @ Ri.T + ti).reshape(Hp, Wp, 3)
            pred_z = al_cam[:, :, 2]
            valid_d = msk & (pred_z > 0)
            dm = depth_metrics(pred_z, gt_d_r, valid_d)
            for k, v in dm.items():
                if k != 'scale':
                    all_metrics[k].append(v)

        # ── Pose metrics (VCRE via aligner or Essential Matrix) ──
        sx, sy = Wp / W_gt, Hp / H_gt
        fx_s, fy_s = K1[0, 0] * sx, K1[1, 1] * sy
        cx_s, cy_s = K1[0, 2] * sx, K1[1, 2] * sy
        K_scaled = np.array([[fx_s, 0, cx_s], [0, fy_s, cy_s], [0, 0, 1]], dtype=np.float64)

        # GT relative pose: cam1_from_cam0
        R_gt_rel = R2.T @ R1           # (3,3)
        t_gt_rel = R2.T @ (t1 - t2)   # (3,)

        if use_aligner and aligner_poses is not None:
            # ── Aligner path (same as lunar eval) ──────────────────
            # aligner_poses[i] is 4×4 cam2world in the aligner frame
            try:
                R0_a = aligner_poses[0, :3, :3]
                t0_a = aligner_poses[0, :3, 3]
                R1_a = aligner_poses[1, :3, :3]
                t1_a = aligner_poses[1, :3, 3]
                # relative pose: cam1_from_cam0
                R_pred_rel = R1_a.T @ R0_a
                t_pred_rel = R1_a.T @ (t0_a - t1_a)

                # RRA between pred and GT relative rotation
                R_err = R_pred_rel @ R_gt_rel.T
                cos_angle = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
                rra = float(np.degrees(np.arccos(cos_angle)))
                all_metrics['emat_rra'].append(rra)

                # vcre_from_relative_pose expects a unit vector and rescales to
                # ||t_gt|| internally. Aligner t has arbitrary scale → normalize.
                t_pred_unit = t_pred_rel / (np.linalg.norm(t_pred_rel) + 1e-8)
                vcre = vcre_from_relative_pose(
                    R_pred_rel, t_pred_unit, R_gt_rel, t_gt_rel,
                    K=K_scaled, image_size=(Wp, Hp),
                )
                for k, v in vcre.items():
                    all_metrics[k].append(v)
            except Exception:
                pass
        else:
            # ── Essential Matrix path ───────────────────────────────
            pe = pose_from_essential(desc1, desc2, K_scaled, conf1, conf2)
            R_pred = pe.pop('_R_pred', None)
            t_pred = pe.pop('_t_pred', None)
            for k, v in pe.items():
                all_metrics[k].append(v)

            if R_pred is not None and t_pred is not None:
                try:
                    vcre = vcre_from_relative_pose(
                        R_pred, t_pred, R_gt_rel, t_gt_rel,
                        K=K_scaled, image_size=(Wp, Hp),
                    )
                    for k, v in vcre.items():
                        all_metrics[k].append(v)
                except Exception:
                    pass

        # ── Save visualizations ──
        if save_vis and vis_dir is not None:
            pair_dir = os.path.join(vis_dir, f"{seq}_{img1_name}_{img2_name}")
            os.makedirs(pair_dir, exist_ok=True)

            # Load images for colors
            img1_rgb = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
            img1_rgb = cv2.resize(img1_rgb, (Wp, Hp))
            img2_rgb = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.resize(img2_rgb, (Wp, Hp))

            # PLY — pred (aligned) + GT, both views combined, colored
            pred_combined = np.vstack([aligned1[mask_flat1], aligned2[mask_flat2]])
            gt_combined = np.vstack([gt_flat1[mask_flat1], gt_flat2[mask_flat2]])
            colors_pred = np.vstack([
                img1_rgb.reshape(-1, 3)[mask_flat1],
                img2_rgb.reshape(-1, 3)[mask_flat2],
            ])
            colors_gt = colors_pred.copy()
            write_ply(os.path.join(pair_dir, "pred_colored.ply"),
                      pred_combined, colors_pred)
            write_ply(os.path.join(pair_dir, "gt_colored.ply"),
                      gt_combined, colors_gt)

            # Depth figures (4-panel: input | GT | pred | error)
            # Transform aligned pts back to camera frame for Z comparison
            pred_cam1 = (aligned1 @ R1_inv.T + t1_inv).reshape(Hp, Wp, 3)
            pred_cam2 = (aligned2 @ R2_inv.T + t2_inv).reshape(Hp, Wp, 3)
            save_depth_figure(
                os.path.join(pair_dir, "depth_v1.png"),
                img1_rgb, gt_depth1_r, pred_cam1[:, :, 2], gt_mask1,
                view_name=img1_name)
            save_depth_figure(
                os.path.join(pair_dir, "depth_v2.png"),
                img2_rgb, gt_depth2_r, pred_cam2[:, :, 2], gt_mask2,
                view_name=img2_name)

            # Confidence map
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].imshow(conf1, cmap='viridis')
            axes[0].set_title(f"Conf v1 (std={conf1.std():.2f})")
            axes[0].axis('off')
            axes[1].imshow(conf2, cmap='viridis')
            axes[1].set_title(f"Conf v2 (std={conf2.std():.2f})")
            axes[1].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(pair_dir, "confidence.png"), dpi=100)
            plt.close(fig)

            # Save input images
            cv2.imwrite(os.path.join(pair_dir, "input_v1.jpg"), cv2.imread(img1_path))
            cv2.imwrite(os.path.join(pair_dir, "input_v2.jpg"), cv2.imread(img2_path))

        if (step + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (step + 1) * (n - step - 1)
            ar = np.mean(all_metrics.get('abs_rel', [0]))
            ch = np.mean(all_metrics.get('chamfer', [0]))
            p3 = np.mean(all_metrics.get('pts3d_mean', [0]))
            print(f"  [{model_name}] {step+1}/{n}  "
                  f"abs_rel={ar:.4f}  chamfer={ch:.3f}  pts3d={p3:.3f}  "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)")

    # Aggregate
    results = {}
    for k, vals in all_metrics.items():
        results[k] = float(np.mean(vals))
        results[k + "_std"] = float(np.std(vals))
    results['n_pairs'] = len(all_metrics.get('abs_rel', []))
    results['total_time'] = time.time() - t0

    # VCRE Precision@t and median-of-medians (proper population statistics)
    vcre_vals = all_metrics.get('vcre_median_px', [])
    if vcre_vals:
        arr = np.array(vcre_vals)
        results['vcre_median_of_medians'] = float(np.median(arr))
        for t in [5, 10, 20, 50, 100]:
            results[f'vcre_prec@{t}'] = float((arr < t).mean() * 100)

    return results


# =====================================================================
#  Print comparison table
# =====================================================================
def print_table(all_results):
    metrics_cols = ['abs_rel', 'rmse', 'delta1', 'pts3d_mean', 'pts3d_median', 'emat_rra',
                    'vcre_median_of_medians', 'vcre_prec@5', 'vcre_prec@10', 'vcre_prec@20', 'vcre_prec@50', 'vcre_prec@100']
    col_names =    ['AbsRel↓', 'RMSE↓', 'δ<1.25↑', 'Pts3D↓', 'Pts3D_med↓', 'RRA(°)↓',
                    'VCRE_med↓', 'P@5↑', 'P@10↑', 'P@20↑', 'P@50↑', 'P@100↑']
    col_fmt =      ['.4f',     '.3f',    '.4f',      '.3f',     '.3f',         '.2f',
                    '.1f',      '.1f',   '.1f',      '.1f',     '.1f',          '.1f']

    header = f"{'Model':<25} {'Group':<10}"
    for cn in col_names:
        header += f" {cn:>10}"
    header += f" {'#pairs':>7}"

    print("\n" + "=" * len(header))
    print("  BLENDEDMVS — FULL COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    current_group = None
    for name, r in all_results.items():
        group = r.get('_group', '')
        if group != current_group:
            if current_group is not None:
                print("-" * len(header))
            current_group = group
        label = r.get('_label', name)
        row = f"{label:<25} {group:<10}"
        for mc, fmt in zip(metrics_cols, col_fmt):
            val = r.get(mc, float('nan'))
            row += f" {val:>10{fmt}}"
        row += f" {r.get('n_pairs', 0):>7d}"
        print(row)

    print("=" * len(header))

    # 50 per group
    print("\n── 50 per group ──")
    groups = {}
    for name, r in all_results.items():
        g = r.get('_group', 'other')
        if g not in groups:
            groups[g] = []
        groups[g].append((name, r))

    for g, items in groups.items():
        if len(items) == 0:
            continue
        # Sort by pts3d_mean (lower = better)
        best = min(items, key=lambda x: x[1].get('pts3d_mean', 1e9))
        bname, br = best
        print(f"  {g:<12} → {br.get('_label', bname):<25}  "
              f"pts3d={br.get('pts3d_mean', 0):.3f}  "
              f"abs_rel={br.get('abs_rel', 0):.4f}  "
              f"delta1={br.get('delta1', 0):.4f}")


# =====================================================================
#  Main
# =====================================================================
def run_evaluation(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Filter configs
    configs_to_eval = OrderedDict()
    only = getattr(args, "only", None)
    if isinstance(only, str):
        only = [only]

    groups = getattr(args, "groups", None)
    if only:
        for key in only:
            if key == "teacher":
                continue  # handled separately
            if key in STUDENT_CONFIGS:
                configs_to_eval[key] = STUDENT_CONFIGS[key]
            else:
                print(f"Unknown model key: {key}. Available: {list(STUDENT_CONFIGS.keys())}")
        if "teacher" in only:
            args.include_teacher = True
    elif groups:
        for key, cfg in STUDENT_CONFIGS.items():
            if cfg['group'] in groups:
                configs_to_eval[key] = cfg
    else:
        configs_to_eval = STUDENT_CONFIGS

    # Override checkpoint paths
    checkpoint = getattr(args, "checkpoint", None)
    if checkpoint:
        for cfg in configs_to_eval.values():
            base_dir = os.path.dirname(cfg['ckpt'])
            cfg['ckpt'] = os.path.join(base_dir, f"checkpoint-{checkpoint}.pth")

    # Load test pairs
    test_pairs = load_test_pairs()

    # Optional filter: keep only a single pair matching "{seq_hex}_{img1:08d}_{img2:08d}"
    pair_filter = getattr(args, "pair", None)
    if pair_filter:
        parts = pair_filter.split("_")
        if len(parts) != 3:
            raise ValueError(f"--pair expects 'SEQHEX_IMG1_IMG2', got {pair_filter}")
        seq_hex, t_img1, t_img2 = parts
        if len(seq_hex) != 24:
            raise ValueError(f"seq hex must be 24 chars (8+16), got {seq_hex}")
        t_seqh = int(seq_hex[:8], 16)
        t_seql = int(seq_hex[8:], 16)
        t_img1 = int(t_img1)
        t_img2 = int(t_img2)
        keep = (
            (test_pairs['seq_high'] == t_seqh)
            & (test_pairs['seq_low']  == t_seql)
            & (test_pairs['img1']     == t_img1)
            & (test_pairs['img2']     == t_img2)
        )
        test_pairs = test_pairs[keep]
        if len(test_pairs) == 0:
            raise RuntimeError(f"--pair {pair_filter} not found in test split")
        args.max_pairs = len(test_pairs)
        print(f"Pair filter active → {len(test_pairs)} pair(s): {pair_filter}")

    print(f"BlendedMVS test split: {len(test_pairs)} pairs "
          f"(evaluating up to {args.max_pairs})\n")

    all_results = OrderedDict()

    # ── Teacher (upper bound) ───────────────────────────────────────
    if getattr(args, "include_teacher", False):
        print("=" * 60)
        teacher = load_teacher(device)
        vis_dir = os.path.join(args.vis_root, "teacher") if args.save_vis else None
        r = evaluate_model(teacher, "Teacher", test_pairs, device, args.max_pairs,
                           save_vis=args.save_vis, vis_dir=vis_dir,
                           use_aligner=args.use_aligner,
                           common_mask=getattr(args, "common_mask", False),
                           fair_align=getattr(args, "fair_align", False))
        r['_group'] = 'teacher'
        r['_label'] = 'Teacher (ViT-L)'
        all_results['teacher'] = r
        del teacher
        torch.cuda.empty_cache()
        print(f"  → abs_rel={r.get('abs_rel',0):.4f}  "
              f"delta1={r.get('delta1',0):.4f}  "
              f"pts3d={r.get('pts3d_mean',0):.3f}\n")

    # ── Students ────────────────────────────────────────────────────
    for name, cfg in configs_to_eval.items():
        print(f"{'='*60}")
        print(f"[{name}] group={cfg['group']}  ckpt={cfg['ckpt']}")
        model = load_student(name, cfg, device)
        if model is None:
            continue
        vis_dir = os.path.join(args.vis_root, name) if args.save_vis else None
        r = evaluate_model(model, name, test_pairs, device, args.max_pairs,
                           save_vis=args.save_vis, vis_dir=vis_dir,
                           use_aligner=args.use_aligner,
                           common_mask=getattr(args, "common_mask", False),
                           fair_align=getattr(args, "fair_align", False))
        r['_group'] = cfg['group']
        r['_label'] = cfg['label']
        r['_ckpt'] = cfg['ckpt']
        all_results[name] = r
        print(f"  → abs_rel={r.get('abs_rel',0):.4f}  "
              f"delta1={r.get('delta1',0):.4f}  "
              f"pts3d={r.get('pts3d_mean',0):.3f}")
        del model
        torch.cuda.empty_cache()

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Compare all BlendedMVS student models")
    parser.add_argument("--max_pairs", type=int, default=200,
                        help="Max test pairs per model")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--groups", nargs="*", default=None,
                        help="Only evaluate these groups: gt, distill, v1, v3")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Only evaluate these specific models (by key)")
    parser.add_argument("--include_teacher", action="store_true",
                        help="Also evaluate the teacher (upper bound)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint: 'best', 'last', or epoch number")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default depends on --use_aligner)")
    parser.add_argument("--use_aligner", action="store_true",
                        help="Use global aligner (PointCloudOptimizer) for VCRE instead of "
                             "Essential Matrix. Slower (~3-5× per pair) but consistent with "
                             "lunar eval. Results saved to results/blendedmvs/bench_aligner/ "
                             "by default.")
    parser.add_argument("--common_mask", action="store_true",
                        help="Use common GT mask (gt_mask only, no conf>1.5 filter) "
                             "so every model is evaluated on the same pixels — "
                             "makes Chamfer/depth metrics directly comparable.")
    parser.add_argument("--fair_align", action="store_true",
                        help="Fit Sim(3) alignment on (gt_mask & conf>1.5) for a "
                             "clean per-model fit, then evaluate metrics on gt_mask "
                             "only (same pixels for every model). Best methodology: "
                             "no 'hiding' bad predictions via conf filter, but "
                             "alignment isn't polluted by low-conf outliers.")
    parser.add_argument("--pair", default=None,
                        help="Filter to a single pair: 'SEQHEX_IMG1_IMG2' "
                             "(e.g. '5aa235f64a17b335eeaf9609_00000056_00000014')")
    parser.add_argument("--save_vis", action="store_true",
                        help="Save PLY point clouds + depth map PNGs per pair")
    parser.add_argument("--vis_root", default="results/blendedmvs/vis",
                        help="Root dir for visual outputs (subdir per model)")
    args = parser.parse_args()

    # Default output path depends on pose mode
    if args.output is None:
        if args.use_aligner:
            args.output = "results/blendedmvs/bench_aligner/comparison_all.json"
        else:
            args.output = "results/blendedmvs/comparison_all.json"

    all_results = run_evaluation(args)

    # ── Results ─────────────────────────────────────────────────────
    print_table(all_results)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # Strip internal keys for JSON
    save_results = {}
    for name, r in all_results.items():
        save_results[name] = {k: v for k, v in r.items() if not k.startswith('_')}
        save_results[name]['group'] = r.get('_group', '')
        save_results[name]['label'] = r.get('_label', name)
    with open(args.output, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
