"""
Auxiliary distillation losses.

  GradientSmoothnessLoss  - penalizes differences in spatial depth gradients
                            between student and teacher (reduces noisy predictions
                            when there is a large capacity gap).

  RelationalFeatureDistillation (RFD) - distills the cross-view correlation
                            structure rather than point-wise features, which is
                            more robust to student-teacher capacity differences.

  TeacherConfLoss         - replaces the student-confidence-weighted ConfLoss
                            with the teacher's confidence as fixed weights
                            (Distill3R-style, prevents student from cheating).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dust3r.losses import MultiLoss
from mast3r.losses import (
    ConfMatchingLoss, MatchingLoss, InfoNCE, Regr3D,
)


# ---------------------------------------------------------------------------
# Gradient Smoothness Loss
# ---------------------------------------------------------------------------

class GradientSmoothnessLoss(nn.Module):
    """Penalize Sobel-gradient differences between student and teacher depth maps.

    L_grad = ||nabla_x D_s - nabla_x D_t||_1 + ||nabla_y D_s - nabla_y D_t||_1

    Particularly useful when distilling from large backbones (ViT-L/14) to
    smaller ones (MobileNet, ViT-Tiny) where the capacity gap causes noisy outputs.

    Args:
        edge_aware: unused placeholder for future edge-weighted variant.
    """

    def __init__(self, edge_aware: bool = False):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 4.0
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3) / 4.0
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _gradients(self, depth: torch.Tensor):
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        depth_pad = F.pad(depth, (1, 1, 1, 1), mode='replicate')
        grad_x = F.conv2d(depth_pad, self.sobel_x.to(depth.device))
        grad_y = F.conv2d(depth_pad, self.sobel_y.to(depth.device))
        return grad_x, grad_y

    def forward(self, depth_student: torch.Tensor, depth_teacher: torch.Tensor,
                valid_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            depth_student: [B, H, W, 3] pts3d from student (Z is depth).
            depth_teacher: [B, H, W, 3] pts3d from teacher.
            valid_mask:    [B, H, W] optional validity mask.

        Returns:
            Scalar loss.
        """
        # Extract depth (Z component)
        d_s = depth_student[..., 2] if depth_student.dim() == 4 else depth_student
        d_t = depth_teacher[..., 2] if depth_teacher.dim() == 4 else depth_teacher

        gx_s, gy_s = self._gradients(d_s)
        gx_t, gy_t = self._gradients(d_t)
        loss_x = torch.abs(gx_s - gx_t)
        loss_y = torch.abs(gy_s - gy_t)

        if valid_mask is not None:
            mask = valid_mask.unsqueeze(1).float()
            if mask.shape != loss_x.shape:
                mask = F.interpolate(mask, size=loss_x.shape[-2:], mode='nearest')
            loss_x = loss_x * mask
            loss_y = loss_y * mask
            num_valid = mask.sum().clamp(min=1)
            return (loss_x.sum() + loss_y.sum()) / num_valid

        return loss_x.mean() + loss_y.mean()


class DepthGradientLoss(nn.Module):
    """Full 3D gradient loss applied to pts3d predictions.

    Applies GradientSmoothnessLoss to depth (Z) and optionally to X/Y positions.

    Args:
        depth_weight: weight for Z-gradient loss.
        xy_weight: weight for X/Y-gradient losses (0 to disable).
    """

    def __init__(self, depth_weight: float = 1.0, xy_weight: float = 0.1):
        super().__init__()
        self.grad_loss = GradientSmoothnessLoss()
        self.depth_weight = depth_weight
        self.xy_weight = xy_weight

    def forward(self, pts3d_student: torch.Tensor, pts3d_teacher: torch.Tensor,
                valid_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pts3d_student: [B, H, W, 3] student predictions.
            pts3d_teacher: [B, H, W, 3] teacher predictions (pseudo-GT).
            valid_mask:    [B, H, W] optional validity mask.

        Returns:
            Scalar loss.
        """
        loss = self.depth_weight * self.grad_loss(pts3d_student, pts3d_teacher, valid_mask)

        if self.xy_weight > 0:
            for dim in (0, 1):
                d_s = pts3d_student[..., dim]
                d_t = pts3d_teacher[..., dim]
                gx_s, gy_s = self.grad_loss._gradients(d_s)
                gx_t, gy_t = self.grad_loss._gradients(d_t)
                loss = loss + self.xy_weight * (
                    torch.abs(gx_s - gx_t).mean() + torch.abs(gy_s - gy_t).mean()
                )
        return loss


# ---------------------------------------------------------------------------
# Relational Feature Distillation (RFD)
# ---------------------------------------------------------------------------

class RelationalFeatureDistillation(nn.Module):
    """Cross-view correlation distillation (RFD).

    Instead of aligning features point-by-point, distill the cross-view
    correlation structure:
        C[i,j] = cos(f_view1[i], f_view2[j])
        L_RFD  = KL(softmax(C_student / tau) || softmax(C_teacher / tau))

    The student does not need to match teacher features in absolute terms,
    only the relative correspondence structure. Inspired by RKD (Park et al.,
    CVPR 2019) and PointDistiller (CVPR 2023).

    Args:
        temperature: softmax temperature (lower = sharper distribution).
        n_samples: number of spatial tokens to subsample (for memory).
    """

    def __init__(self, temperature: float = 0.1, n_samples: int = 512):
        super().__init__()
        self.temperature = temperature
        self.n_samples = n_samples

    def forward(self, feat_s_v1: torch.Tensor, feat_s_v2: torch.Tensor,
                feat_t_v1: torch.Tensor, feat_t_v2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_s_v1, feat_s_v2: student features for views 1 and 2, [B, C_s, H, W].
            feat_t_v1, feat_t_v2: teacher features for views 1 and 2, [B, C_t, H, W].

        Returns:
            Scalar RFD loss.
        """
        def _flatten_norm(feat):
            B, C, H, W = feat.shape
            return F.normalize(feat.flatten(2).transpose(1, 2), dim=-1)

        fs1, fs2 = _flatten_norm(feat_s_v1), _flatten_norm(feat_s_v2)
        ft1, ft2 = _flatten_norm(feat_t_v1), _flatten_norm(feat_t_v2)
        B, N, _ = fs1.shape
        n_sub = min(self.n_samples, N)

        total = 0.0
        for b in range(B):
            idx = torch.randperm(N, device=fs1.device)[:n_sub]
            C_s = torch.mm(fs1[b, idx], fs2[b, idx].T) / self.temperature
            C_t = torch.mm(ft1[b, idx], ft2[b, idx].T) / self.temperature
            total = total + F.kl_div(
                F.log_softmax(C_s, dim=-1),
                F.softmax(C_t, dim=-1),
                reduction='batchmean',
            )
        return total / B


# ---------------------------------------------------------------------------
# Teacher-confidence weighted regression (Distill3R-style)
# ---------------------------------------------------------------------------

class TeacherConfLoss(MultiLoss):
    """Teacher-confidence-weighted regression loss.

    Replaces the standard ConfLoss (which uses student confidence, enabling
    the student to cheat by predicting low confidence everywhere) with the
    teacher's cached confidence as fixed, non-trainable weights:

        L_regr = (1/|M|) sum_i C_teacher[i] * ||pred_i - gt_i||
        L_conf = (1/|M|) sum_i |conf_student[i] - conf_teacher[i]|
        L_total = L_regr + gamma * L_conf

    Args:
        pixel_loss: a Regr3D instance (will be used with reduction='none').
        gamma: weight for confidence distillation L1 term.
    """

    def __init__(self, pixel_loss, gamma: float = 0.2):
        super().__init__()
        self.pixel_loss = pixel_loss.with_reduction('none')
        self.gamma = gamma

    def get_name(self) -> str:
        return f'TeacherConfLoss({self.pixel_loss})'

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        ((loss1, msk1), (loss2, msk2)), details = self.pixel_loss(gt1, gt2, pred1, pred2, **kw)

        def _weighted(loss, msk, gt):
            c = gt.get('teacher_conf', None)
            if c is not None and loss.numel() > 0:
                return (loss * c[msk].clamp(min=0)).mean()
            return loss.mean() if loss.numel() > 0 else loss.new_zeros(())

        regr = _weighted(loss1, msk1, gt1) + _weighted(loss2, msk2, gt2)

        def _conf_l1(pred, gt, msk):
            c_t = gt.get('teacher_conf', None)
            if c_t is None or 'conf' not in pred:
                return torch.tensor(0., device=regr.device)
            c_s = pred['conf'].squeeze(1) if pred['conf'].dim() == 4 else pred['conf']
            return F.l1_loss(c_s[msk], c_t[msk])

        conf_loss = _conf_l1(pred1, gt1, msk1) + _conf_l1(pred2, gt2, msk2)
        total = regr + self.gamma * conf_loss
        details.update({'conf_l1': float(conf_loss)})
        return total, details
