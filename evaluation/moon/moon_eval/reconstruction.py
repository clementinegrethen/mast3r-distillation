"""
moon_eval/reconstruction.py — Scene reconstruction from image pairs.

Two reconstruction back-ends:
  run_global_aligner — dust3r PointCloudOptimizer (demo-style)
  run_sparse_ga      — mast3r sparse_global_alignment (more robust matching)

Both return the same 5-tuple:
  (pts3d_list, depthmaps, confs, poses, focals)

The public router `get_reconstruction` tries run_global_aligner first and
falls back to run_sparse_ga on any exception.
"""

import sys
import os
import tempfile
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional

# Must come before any dust3r imports
import mast3r.utils.path_to_dust3r  # noqa


def _to_numpy(x):
    """Convert a tensor (or list/array) to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return [_to_numpy(v) for v in x]
    return np.array(x)


# ─────────────────────────────────────────────────────────────────────────────
# PointCloudOptimizer (demo-style global aligner)
# ─────────────────────────────────────────────────────────────────────────────

def run_global_aligner(
    model: torch.nn.Module,
    device,
    img1_path,
    img2_path,
    niter: int = 300,
    schedule: str = "cosine",
    lr: float = 0.01,
    verbose: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    """Demo-style PointCloudOptimizer reconstruction of a stereo pair.

    Runs dust3r inference followed by the global_aligner optimisation loop
    (PointCloudOptimizer).  Both views are reconstructed in a single unified
    world frame that is internally consistent.

    Args:
        model     : loaded MASt3R / DUSt3R model (eval mode)
        device    : torch device
        img1_path : path to first image
        img2_path : path to second image
        niter     : number of optimisation iterations
        schedule  : learning-rate schedule ('cosine' | 'linear')
        lr        : peak learning rate
        verbose   : print progress

    Returns 5-tuple:
        pts3d_list : list of 2 arrays, each (H, W, 3) in world frame
        depthmaps  : list of 2 arrays, each (H, W)
        confs      : list of 2 arrays, each (H, W) confidence/mask
        poses      : (2, 4, 4) cam2world poses from ga.get_im_poses()
        focals     : (2,) optimised focal lengths
    """
    from dust3r.utils.image import load_images
    from dust3r.inference import inference
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

    filelist = [str(img1_path), str(img2_path)]
    images = load_images(filelist, size=512, verbose=verbose)
    pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)

    with torch.no_grad():
        output = inference(pairs, model, device, batch_size=1, verbose=verbose)

    ga = global_aligner(
        output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer
    )

    # API may differ between DUSt3R versions — try with kwargs first
    try:
        ga.compute_global_alignment(
            init="mst", niter=niter, schedule=schedule, lr=lr
        )
    except TypeError:
        ga.compute_global_alignment()

    pts3d_list = _to_numpy(ga.get_pts3d())
    depthmaps = _to_numpy(ga.get_depthmaps())
    confs = _to_numpy(ga.get_masks())
    poses = ga.get_im_poses().detach().cpu().numpy()
    focals = ga.get_focals().detach().cpu().numpy()

    return pts3d_list, depthmaps, confs, poses, focals


# ─────────────────────────────────────────────────────────────────────────────
# Sparse global alignment (mast3r back-end)
# ─────────────────────────────────────────────────────────────────────────────

def run_sparse_ga(
    model: torch.nn.Module,
    device,
    img1_path,
    img2_path,
    K_GT: np.ndarray,
    matching_conf_thr: float = 5.0,
    niter1: int = 500,
    niter2: int = 200,
    opt_depth: bool = False,
    verbose: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    """MASt3R sparse_global_alignment reconstruction.

    Uses mast3r.cloud_opt.sparse_ga.sparse_global_alignment with known
    intrinsics K_GT injected as initialisation.

    Args:
        model              : loaded model
        device             : torch device
        img1_path, img2_path: image paths
        K_GT               : (3, 3) known camera intrinsics
        matching_conf_thr  : confidence threshold for feature matching
        niter1, niter2     : iteration counts for two optimisation stages
        opt_depth          : also optimise per-pixel depths
        verbose            : print progress

    Returns same 5-tuple as run_global_aligner.
    """
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs

    filelist = [str(img1_path), str(img2_path)]
    images = load_images(filelist, size=512, verbose=verbose)
    pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)

    init_dict = {
        p: {"intrinsics": torch.tensor(K_GT, dtype=torch.float32)}
        for p in filelist
    }

    cache_path = tempfile.mkdtemp(prefix="moon_eval_sparse_ga_")
    scene = sparse_global_alignment(
        filelist, pairs, cache_path, model,
        lr1=0.01, niter1=niter1,
        lr2=0.014, niter2=niter2,
        device=device,
        shared_intrinsics=True,
        matching_conf_thr=matching_conf_thr,
        opt_depth=opt_depth,
        init=init_dict,
    )

    pts3d_confs = _to_numpy(scene.get_dense_pts3d(clean_depth=True))
    pts3d_list, depthmaps, confs = pts3d_confs

    poses = scene.get_im_poses().detach().cpu().numpy()
    focals = scene.get_focals().detach().cpu().numpy()

    return pts3d_list, depthmaps, confs, poses, focals


# ─────────────────────────────────────────────────────────────────────────────
# Public router
# ─────────────────────────────────────────────────────────────────────────────

def get_reconstruction(
    model: torch.nn.Module,
    device,
    img1_path,
    img2_path,
    K_GT: np.ndarray,
    mode: str = "global_aligner",
    verbose: bool = False,
    **kwargs,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    """Route to the appropriate reconstruction backend.

    Modes:
      'global_aligner'   : PointCloudOptimizer (tries first; fast)
      'sparse_ga'        : sparse_global_alignment
      'sparse_ga_depth'  : sparse_ga with opt_depth=True

    For mode='global_aligner', falls back to 'sparse_ga' on failure.

    Args:
        model, device    : model and device
        img1_path, img2_path: image paths
        K_GT             : (3, 3) intrinsics
        mode             : reconstruction mode (see above)
        verbose          : verbosity
        **kwargs         : forwarded to the backend (niter1, niter2, etc.)

    Returns same 5-tuple as the individual backends.
    """
    if mode == "global_aligner":
        try:
            return run_global_aligner(
                model, device, img1_path, img2_path,
                niter=kwargs.get("niter", 300),
                schedule=kwargs.get("schedule", "cosine"),
                lr=kwargs.get("lr", 0.01),
                verbose=verbose,
            )
        except Exception as e:
            if verbose:
                print(f"  [reconstruction] global_aligner failed ({e}), "
                      f"falling back to sparse_ga")
            return run_sparse_ga(
                model, device, img1_path, img2_path, K_GT,
                matching_conf_thr=kwargs.get("matching_conf_thr", 5.0),
                niter1=kwargs.get("niter1", 500),
                niter2=kwargs.get("niter2", 200),
                opt_depth=False,
                verbose=verbose,
            )

    elif mode in ("sparse_ga", "sparse_ga_depth"):
        opt_depth = (mode == "sparse_ga_depth")
        return run_sparse_ga(
            model, device, img1_path, img2_path, K_GT,
            matching_conf_thr=kwargs.get("matching_conf_thr", 5.0),
            niter1=kwargs.get("niter1", 500),
            niter2=kwargs.get("niter2", 200),
            opt_depth=opt_depth,
            verbose=verbose,
        )

    else:
        raise ValueError(f"Unknown mode: {mode!r}. "
                         f"Choose from 'global_aligner', 'sparse_ga', 'sparse_ga_depth'.")
