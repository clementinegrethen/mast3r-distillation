"""
distillation — core components for MASt3R knowledge distillation.

  students.py      — student builder functions (MobileNet, ViT, ViT-Tiny)
  svd_init.py      — SVD-based decoder initialization
  feature_loss.py  — cosine-margin and mixed feature alignment losses
  losses.py        — gradient smoothness and relational distillation losses
"""

from .svd_init import svd_init_student_from_teacher, layer_mapping
from .feature_loss import (
    DepthAnythingFeatureAlignLoss,
    MixedFeatureAlignLoss,
    build_feature_loss,
)
from .students import (
    build_mobilenet_student,
    build_vit_student,
    build_vit_tiny_student,
    build_dinov3_student,
)

__all__ = [
    "svd_init_student_from_teacher",
    "layer_mapping",
    "DepthAnythingFeatureAlignLoss",
    "MixedFeatureAlignLoss",
    "build_feature_loss",
    "build_mobilenet_student",
    "build_vit_student",
    "build_vit_tiny_student",
    "build_dinov3_student",
]
