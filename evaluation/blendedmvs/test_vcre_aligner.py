#!/usr/bin/env python3
"""
Test aligner-based VCRE + chamfer consistency on a few BlendedMVS pairs.

Expected:
  1. conf / pts3d from run_inference_with_aligner == run_inference  (max diff < 1e-4)
  2. Aligner VCRE << random VCRE  (poses are meaningful)
  3. VCRE is small for the teacher after t-normalisation fix
  4. aligner_poses shape (2,4,4), RRA in [0°,180°]

Usage:
    source /home/clgrethen/miniconda3/bin/activate mast3r
    cd /projects/m25147/3D-Moon
    python evaluation/blendedmvs/test_vcre_aligner.py
"""
import sys, os
sys.path.insert(0, '.')
sys.path.insert(0, 'mast3r')
sys.path.insert(0, 'dust3r')

import numpy as np
import torch
import cv2

from evaluation.blendedmvs.eval_compare_all_bmvs import (
    vcre_from_relative_pose,
    run_inference,
    run_inference_with_aligner,
    load_teacher,
    BMVS_ROOT,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device: {DEVICE}\n")

# ── Load one real pair ────────────────────────────────────────────────────────
pairs = np.load(os.path.join(BMVS_ROOT, 'blendedmvs_pairs.npy'))
test_mask = (pairs['seq_low'] % 10) == 1
pairs = pairs[test_mask]
rng = np.random.default_rng(42)

def get_pair(idx):
    seqh, seql = int(pairs[idx]['seq_high']), int(pairs[idx]['seq_low'])
    seq = f"{seqh:08x}{seql:016x}"
    i1, i2 = int(pairs[idx]['img1']), int(pairs[idx]['img2'])
    seq_path = os.path.join(BMVS_ROOT, seq)
    p1 = os.path.join(seq_path, f"{i1:08d}.jpg")
    p2 = os.path.join(seq_path, f"{i2:08d}.jpg")
    npz1 = os.path.join(seq_path, f"{i1:08d}.npz")
    npz2 = os.path.join(seq_path, f"{i2:08d}.npz")
    if not all(os.path.isfile(f) for f in [p1, p2, npz1, npz2]):
        return None
    gt1 = np.load(npz1); gt2 = np.load(npz2)
    R1 = np.float64(gt1['R_cam2world']); t1 = np.float64(gt1['t_cam2world'])
    R2 = np.float64(gt2['R_cam2world']); t2 = np.float64(gt2['t_cam2world'])
    K1 = np.float64(gt1['intrinsics'])
    H_img, W_img = 384, 512
    H_K, W_K = gt1.get('image_size', np.array([H_img, W_img]))
    sx, sy = W_img / float(W_K), H_img / float(H_K)
    K_s = np.array([[K1[0,0]*sx, 0, K1[0,2]*sx],
                    [0, K1[1,1]*sy, K1[1,2]*sy],
                    [0, 0, 1]], dtype=np.float64)
    R_gt_rel = R2.T @ R1
    t_gt_rel = R2.T @ (t1 - t2)
    return p1, p2, K_s, R_gt_rel, t_gt_rel

print("Loading teacher...")
teacher = load_teacher(DEVICE)
print()

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: conf/pts3d from run_inference_with_aligner must equal run_inference
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Test 1: pts3d/conf identical between run_inference and run_inference_with_aligner")

pair = get_pair(0)
assert pair is not None
p1, p2, K_s, R_gt_rel, t_gt_rel = pair

with torch.no_grad():
    a_pts1, a_pts2, a_c1, a_c2, a_d1, a_d2 = run_inference(teacher, p1, p2, DEVICE)

b_pts1, b_pts2, b_c1, b_c2, b_d1, b_d2, poses = run_inference_with_aligner(teacher, p1, p2, DEVICE)

diff_pts1 = float(np.max(np.abs(a_pts1 - b_pts1)))
diff_pts2 = float(np.max(np.abs(a_pts2 - b_pts2)))
diff_conf1 = float(np.max(np.abs(a_c1 - b_c1)))

print(f"  pts3d_1 max diff: {diff_pts1:.6f}  {'PASS' if diff_pts1 < 1e-4 else 'FAIL'}")
print(f"  pts3d_2 max diff: {diff_pts2:.6f}  {'PASS' if diff_pts2 < 1e-4 else 'FAIL'}")
print(f"  conf1   max diff: {diff_conf1:.6f}  {'PASS' if diff_conf1 < 1e-4 else 'FAIL'}")
print(f"  aligner_poses shape: {poses.shape}  {'PASS' if poses.shape == (2,4,4) else 'FAIL'}")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: VCRE with normalised t — teacher should be << random, and sensible
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Test 2: aligner VCRE vs random baseline (6 pairs)")
print(f"\n  {'pair':>4}  {'align_vcre':>10}  {'rand_vcre':>10}  {'RRA':>8}")

n_test = 6
vcre_aligner, vcre_random, rras = [], [], []

for i, idx in enumerate(rng.choice(len(pairs), n_test, replace=False)):
    pair = get_pair(idx)
    if pair is None:
        continue
    p1, p2, K_s, R_gt_rel, t_gt_rel = pair

    _, _, _, _, _, _, aligner_poses = run_inference_with_aligner(teacher, p1, p2, DEVICE)

    R0_a, t0_a = aligner_poses[0, :3, :3], aligner_poses[0, :3, 3]
    R1_a, t1_a = aligner_poses[1, :3, :3], aligner_poses[1, :3, 3]
    R_pred = R1_a.T @ R0_a
    t_pred = R1_a.T @ (t0_a - t1_a)
    t_pred_unit = t_pred / (np.linalg.norm(t_pred) + 1e-8)  # ← normalisation fix

    R_err = R_pred @ R_gt_rel.T
    rra = float(np.degrees(np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))))

    v_align = vcre_from_relative_pose(R_pred, t_pred_unit, R_gt_rel, t_gt_rel,
                                      K_s, (512, 384)).get('vcre_median_px', float('nan'))

    R_rand, _ = cv2.Rodrigues(rng.uniform(-np.pi, np.pi, 3))
    t_rand = rng.uniform(-1, 1, 3)
    v_rand = vcre_from_relative_pose(R_rand, t_rand / np.linalg.norm(t_rand),
                                     R_gt_rel, t_gt_rel, K_s, (512, 384)
                                     ).get('vcre_median_px', float('nan'))

    vcre_aligner.append(v_align)
    vcre_random.append(v_rand)
    rras.append(rra)
    print(f"  {i+1:4d}  {v_align:10.1f}  {v_rand:10.1f}  {rra:7.2f}°")

med_align = float(np.nanmedian(vcre_aligner))
med_rand  = float(np.nanmedian(vcre_random))
print(f"\n  Aligner  median VCRE : {med_align:.1f} px")
print(f"  Random   median VCRE : {med_rand:.1f} px")
ok_vs_rand = med_align < med_rand
ok_rra     = all(0 <= r <= 180 for r in rras)
ok_vcre    = med_align < 100    # teacher should be well below 100px after fix
print(f"  {'PASS' if ok_vs_rand else 'FAIL'}  aligner VCRE < random VCRE")
print(f"  {'PASS' if ok_rra    else 'FAIL'}  all RRA in [0°,180°]")
print(f"  {'PASS' if ok_vcre   else 'FAIL'}  teacher median VCRE < 100 px")

print("\nAll tests done.")
