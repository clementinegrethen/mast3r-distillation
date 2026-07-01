"""
Feature alignment losses for MASt3R knowledge distillation.

Two modes are provided:

  cosine_margin (recommended):
      L_feat = 1 - mean_i cos(f_s_i, f_t_i)   [only pixels where cos < alpha]
      Follows Depth Anything V2 (Eq. 9). The tolerance margin alpha (default 0.9)
      skips pixels that are already well-aligned, focusing capacity on hard regions.

  mixed:
      L_feat = MSE(norm(f_s), norm(f_t)) + cosine + attention-transfer
      Original formulation; less principled but sometimes useful.

When the student and teacher have different feature dimensions, a learnable
linear projector (student_dim -> teacher_dim) is created per student and
trained jointly. Projecting in this direction lets gradients flow back into
the student encoder, enabling both the projector and the encoder to adapt.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthAnythingFeatureAlignLoss(nn.Module):
    """Cosine-margin feature alignment loss (Depth Anything V2, Eq. 9).

    Args:
        alpha: tolerance margin in [0, 1]. Pixels with cos(f_s, f_t) >= alpha
               are considered already aligned and excluded from the loss.
               Set alpha=1.0 to align all pixels unconditionally.
               Default: 0.9 (DA-V2 setting).
    """

    def __init__(self, alpha: float = 0.9):
        super().__init__()
        self.alpha = alpha
        # Per-student projectors: keyed by student name (not shared)
        self._projectors = nn.ModuleDict()

    def get_proj(self, student_name: str, C_s: int, C_t: int, device) -> nn.Linear | None:
        """Return (and lazily create) the learnable projector for this student.

        Returns None when C_s == C_t (no projection needed).
        """
        if C_s == C_t:
            return None
        key = student_name.replace('-', '_')
        if key not in self._projectors:
            proj = nn.Linear(C_s, C_t, bias=False).to(device)
            nn.init.kaiming_uniform_(proj.weight, a=0.01)
            self._projectors[key] = proj
        return self._projectors[key]

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor,
                student_name: str = '') -> torch.Tensor:
        """Compute the cosine-margin alignment loss.

        Args:
            f_s: student encoder features, shape (B, C_s, H, W).
            f_t: teacher encoder features, shape (B, C_t, H, W). Must be detached.
            student_name: used to look up the per-student projector.

        Returns:
            Scalar loss.
        """
        # Spatial alignment
        if f_s.shape[-2:] != f_t.shape[-2:]:
            f_t = F.interpolate(f_t, size=f_s.shape[-2:], mode='bilinear', align_corners=False)

        B, C_s, H, W = f_s.shape
        C_t = f_t.shape[1]

        # Project student -> teacher space when dims differ
        proj = self.get_proj(student_name, C_s, C_t, f_s.device)
        if proj is not None:
            f_s_proj = proj(
                f_s.permute(0, 2, 3, 1).reshape(B * H * W, C_s)
            ).reshape(B, H, W, C_t).permute(0, 3, 1, 2)
            C_cmp = C_t
        else:
            f_s_proj = f_s
            C_cmp = C_s

        # Per-pixel cosine similarity
        f_s_flat = F.normalize(f_s_proj.permute(0, 2, 3, 1).reshape(-1, C_cmp), dim=-1)
        f_t_flat = F.normalize(f_t.permute(0, 2, 3, 1).reshape(-1, C_cmp), dim=-1)
        cos_sim = (f_s_flat * f_t_flat).sum(dim=-1)

        # Tolerance margin: skip already-aligned pixels
        if self.alpha < 1.0:
            mask = cos_sim < self.alpha
            if mask.sum() == 0:
                return f_s_proj.sum() * 0.0  # keep graph alive, zero loss
            cos_sim = cos_sim[mask]

        return 1.0 - cos_sim.mean()


class MixedFeatureAlignLoss(nn.Module):
    """Mixed feature alignment: MSE + cosine + attention-transfer.

    Args:
        mse_weight: weight for normalized MSE term.
        cosine_weight: weight for cosine similarity term.
        at_weight: weight for attention-transfer (spatial attention map MSE).
    """

    def __init__(self, mse_weight: float = 1.0, cosine_weight: float = 0.5,
                 at_weight: float = 0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.cosine_weight = cosine_weight
        self.at_weight = at_weight

    def forward(self, f_s: torch.Tensor, f_t: torch.Tensor,
                student_name: str = '') -> torch.Tensor:
        """Compute mixed alignment loss.

        Args:
            f_s: student features (B, C_s, H, W).
            f_t: teacher features (B, C_t, H, W). Should be detached.
            student_name: unused, kept for API compatibility.

        Returns:
            Scalar loss.
        """
        if f_s.shape[-2:] != f_t.shape[-2:]:
            f_t = F.interpolate(f_t, size=f_s.shape[-2:], mode='bilinear', align_corners=False)
        if f_s.shape[1] != f_t.shape[1]:
            # Project via average pooling (no learnable weights in this mode)
            f_s = F.adaptive_avg_pool2d(f_s, f_t.shape[-2:])

        f_s_n = F.normalize(f_s, dim=1)
        f_t_n = F.normalize(f_t, dim=1)
        mse = F.mse_loss(f_s_n, f_t_n)

        f_s_flat = F.normalize(f_s.flatten(2).transpose(1, 2), dim=-1)
        f_t_flat = F.normalize(f_t.flatten(2).transpose(1, 2), dim=-1)
        cos = 1.0 - (f_s_flat * f_t_flat).sum(-1).mean()

        def attn(x):
            return F.normalize(x.pow(2).mean(1, keepdim=True).flatten(2), dim=2)

        at = F.mse_loss(attn(f_s), attn(f_t))

        return self.mse_weight * mse + self.cosine_weight * cos + self.at_weight * at


def build_feature_loss(mode: str = 'cosine_margin', alpha: float = 0.9) -> nn.Module:
    """Factory function for feature alignment losses.

    Args:
        mode: 'cosine_margin' (recommended, DA-V2 style) or 'mixed'.
        alpha: tolerance margin for cosine_margin mode.

    Returns:
        An nn.Module with signature forward(f_s, f_t, student_name='').
    """
    if mode == 'cosine_margin':
        return DepthAnythingFeatureAlignLoss(alpha=alpha)
    elif mode == 'mixed':
        return MixedFeatureAlignLoss()
    else:
        raise ValueError(f"Unknown feature loss mode: {mode!r}. "
                         f"Choose from 'cosine_margin' or 'mixed'.")
