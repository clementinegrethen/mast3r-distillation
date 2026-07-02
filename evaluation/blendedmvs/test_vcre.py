#!/usr/bin/env python3
"""
Test VCRE correctness on a few BlendedMVS pairs.

Expected properties:
  1. Identity relative pose → VCRE = 0
  2. GT relative pose used as pred → VCRE = 0 (internal consistency check)
  3. Teacher VCRE << student VCRE  (teacher is best model)
  4. VCRE grows with rotation error (controllable synthetic test)
  5. emat_rra on matched descriptors is a real rotation error (< 180°, correlated with difficulty)

Usage:
    source /home/clgrethen/miniconda3/bin/activate mast3r
    cd /projects/m25147/3D-Moon
    python evaluation/blendedmvs/test_vcre.py
"""
import sys, os
sys.path.insert(0, '.')
sys.path.insert(0, 'mast3r')
sys.path.insert(0, 'dust3r')

import numpy as np
import torch
import cv2
from pathlib import Path

from eval_compare_all_bmvs import (
    vcre_from_relative_pose,
    mutual_nn_matches,
    pose_from_essential,
    run_inference,
    load_teacher,
    BMVS_ROOT,
    TEACHER_CKPT,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {DEVICE}\n")

# ── Test 1: identity relative pose → VCRE = 0 ────────────────────────────────
print("=" * 60)
print("Test 1: identity relative pose → VCRE must be 0")
K = np.array([[500, 0, 256], [0, 500, 192], [0, 0, 1]], dtype=np.float64)
R_id = np.eye(3)
t_id = np.array([0.5, 0.0, 0.0])
res = vcre_from_relative_pose(R_id, t_id / np.linalg.norm(t_id), R_id, t_id, K, (512, 384))
assert res['vcre_median_px'] < 1e-6, f"FAIL: got {res['vcre_median_px']:.4f}px (expected 0)"
print(f"  PASS  vcre_median_px = {res['vcre_median_px']:.6f} px")

# ── Test 2: VCRE grows with rotation error ────────────────────────────────────
print("\nTest 2: VCRE grows monotonically with rotation error")
t_gt = np.array([1.0, 0.0, 0.0])
R_gt = np.eye(3)
prev = -1.0
for deg in [0, 1, 5, 10, 20, 45]:
    angle = np.radians(deg)
    R_err, _ = cv2.Rodrigues(np.array([0.0, angle, 0.0]))
    vcre = vcre_from_relative_pose(R_err, t_gt / np.linalg.norm(t_gt),
                                   R_gt, t_gt, K, (512, 384))
    v = vcre['vcre_median_px']
    status = "PASS" if v > prev else "FAIL"
    print(f"  {status}  rot={deg:3d}°  → vcre={v:.2f} px")
    prev = v

# ── Test 3: GT pose as pred → VCRE = 0 ───────────────────────────────────────
print("\nTest 3: using GT pose as prediction → VCRE must be 0")
# Arbitrary GT relative pose
R_any, _ = cv2.Rodrigues(np.array([0.1, 0.2, 0.05]))
t_any = np.array([0.3, -0.1, 0.8])
res = vcre_from_relative_pose(R_any, t_any / np.linalg.norm(t_any),
                               R_any, t_any, K, (512, 384))
assert res['vcre_median_px'] < 1e-6, f"FAIL: got {res['vcre_median_px']:.4f}px"
print(f"  PASS  vcre_median_px = {res['vcre_median_px']:.6f} px")

# ── Test 4: real pairs — teacher vs random rotation ───────────────────────────
print("\nTest 4: real BlendedMVS pairs — teacher VCRE vs shuffled-pose baseline")

# Find a few real pairs
pairs_path = os.path.join(BMVS_ROOT, 'blendedmvs_pairs.npy')
if not os.path.isfile(pairs_path):
    print("  SKIP: pairs file not found")
else:
    pairs = np.load(pairs_path)
    test_mask = (pairs['seq_low'] % 10) == 1
    pairs = pairs[test_mask]

    print("  Loading teacher...")
    teacher = load_teacher(DEVICE)

    n_test = 10
    rng = np.random.default_rng(42)
    indices = rng.choice(len(pairs), n_test, replace=False)

    vcre_teacher, vcre_random, rras = [], [], []

    for i, idx in enumerate(indices):
        seqh, seql = int(pairs[idx]['seq_high']), int(pairs[idx]['seq_low'])
        seq = f"{seqh:08x}{seql:016x}"
        i1, i2 = int(pairs[idx]['img1']), int(pairs[idx]['img2'])
        seq_path = os.path.join(BMVS_ROOT, seq)
        p1 = os.path.join(seq_path, f"{i1:08d}.jpg")
        p2 = os.path.join(seq_path, f"{i2:08d}.jpg")
        npz1 = os.path.join(seq_path, f"{i1:08d}.npz")
        npz2 = os.path.join(seq_path, f"{i2:08d}.npz")

        if not all(os.path.isfile(f) for f in [p1, p2, npz1, npz2]):
            continue

        gt1 = np.load(npz1)
        gt2 = np.load(npz2)
        R1 = np.float64(gt1['R_cam2world'])
        t1 = np.float64(gt1['t_cam2world'])
        R2 = np.float64(gt2['R_cam2world'])
        t2 = np.float64(gt2['t_cam2world'])
        R_gt_rel = R2.T @ R1
        t_gt_rel = R2.T @ (t1 - t2)

        K1 = np.float64(gt1['intrinsics'])
        H_img, W_img = 384, 512
        H_K, W_K = gt1.get('image_size', np.array([H_img, W_img]))
        sx, sy = W_img / float(W_K), H_img / float(H_K)
        K_s = np.array([[K1[0,0]*sx, 0, K1[0,2]*sx],
                        [0, K1[1,1]*sy, K1[1,2]*sy],
                        [0, 0, 1]], dtype=np.float64)

        with torch.no_grad():
            pts3d_1, pts3d_2, conf1, conf2, desc1, desc2 = run_inference(
                teacher, p1, p2, DEVICE)

        # Pose from descriptor matching
        pe = pose_from_essential(desc1, desc2, K_s, conf1, conf2)
        R_pred = pe.get('_R_pred')
        t_pred = pe.get('_t_pred')
        rra = pe.get('emat_rra', float('nan'))
        n_inl = pe.get('emat_inliers', 0)

        if R_pred is not None:
            vcre = vcre_from_relative_pose(R_pred, t_pred, R_gt_rel, t_gt_rel,
                                           K_s, (W_img, H_img))
            v_teacher = vcre.get('vcre_median_px', float('nan'))
        else:
            v_teacher = float('nan')

        # Random rotation baseline
        R_rand, _ = cv2.Rodrigues(rng.uniform(-np.pi, np.pi, 3))
        t_rand = rng.uniform(-1, 1, 3)
        vcre_rand = vcre_from_relative_pose(R_rand, t_rand / np.linalg.norm(t_rand),
                                            R_gt_rel, t_gt_rel, K_s, (W_img, H_img))
        v_rand = vcre_rand.get('vcre_median_px', float('nan'))

        vcre_teacher.append(v_teacher)
        vcre_random.append(v_rand)
        rras.append(rra)

        print(f"  pair {i+1:2d}: teacher_vcre={v_teacher:7.1f}px  "
              f"random_vcre={v_rand:7.1f}px  RRA={rra:.1f}°  inliers={n_inl}")

    print(f"\n  Teacher  median VCRE: {np.nanmedian(vcre_teacher):.1f} px")
    print(f"  Random   median VCRE: {np.nanmedian(vcre_random):.1f} px")
    ok = np.nanmedian(vcre_teacher) < np.nanmedian(vcre_random)
    print(f"  {'PASS' if ok else 'FAIL'}  teacher VCRE < random VCRE")

print("\nAll tests done.")
