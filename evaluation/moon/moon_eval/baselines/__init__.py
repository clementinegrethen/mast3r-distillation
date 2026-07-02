"""
moon_eval/baselines/ — Classical SfM / third-party baselines.

Available:
  colmap_sfm  : SIFT feature extraction + exhaustive matching + Essential-matrix
                pose estimation, all via pycolmap.  Provides camera-pose metrics
                (RRA / RTA) only — no dense depth.

  colmap_mvs  : SIFT + PatchMatch Stereo dense MVS, all via pycolmap.  Provides
                the full metric suite (3D, depth, slope, terrain) comparable to
                MASt3R/DUSt3R.  Requires CUDA for acceptable speed.
"""

from .colmap_sfm import run_colmap_pair, evaluate_colmap_folder
from .colmap_mvs import run_colmap_mvs_pair, evaluate_colmap_mvs_folder

__all__ = [
    "run_colmap_pair", "evaluate_colmap_folder",
    "run_colmap_mvs_pair", "evaluate_colmap_mvs_folder",
]
