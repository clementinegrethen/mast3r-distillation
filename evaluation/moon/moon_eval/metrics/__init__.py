"""
moon_eval/metrics — Metric computation modules.

Re-exports the most commonly used functions for convenience.
"""

from .classic import (
    compute_accuracy_completeness,
    compute_depth_metrics,
    compute_3d_metrics,
    compute_profile_metrics,
    compute_overlap_consistency,
)
from .terrain import (
    compute_slope_map,
    compute_slope_metrics,
    compute_hda_metrics,
    compute_curvature_maps,
    compute_roughness_map,
    compute_relief_metrics,
)
from .camera import (
    rra_deg,
    rta_deg,
    compute_rra_rta,
    compute_auc,
    compute_pose_from_essential,
    extract_matches_from_model,
    compute_pose_from_aligner,
    compute_camera_metrics_for_pair,
)

__all__ = [
    # classic
    "compute_accuracy_completeness",
    "compute_depth_metrics",
    "compute_3d_metrics",
    "compute_profile_metrics",
    "compute_overlap_consistency",
    # terrain
    "compute_slope_map",
    "compute_slope_metrics",
    "compute_hda_metrics",
    "compute_curvature_maps",
    "compute_roughness_map",
    "compute_relief_metrics",
    # camera
    "rra_deg",
    "rta_deg",
    "compute_rra_rta",
    "compute_auc",
    "compute_pose_from_essential",
    "extract_matches_from_model",
    "compute_pose_from_aligner",
    "compute_camera_metrics_for_pair",
]
