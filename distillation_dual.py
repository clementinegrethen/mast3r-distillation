#!/usr/bin/env python3
"""
distillation_dual.py — Multi-student MASt3R knowledge distillation.

The teacher (MASt3R) runs one forward pass per batch and produces pseudo-GT
3D point maps.  Any number of lightweight student models are then trained
independently against this pseudo-GT using a combination of:
  - Geometry loss: ConfLoss(Regr3D) + ConfMatchingLoss  (MASt3R-style)
  - Feature alignment (optional): cosine-margin loss on encoder features
  - Gradient smoothness (optional): Sobel-gradient difference on depth maps

All students share the teacher forward pass but each has its own optimizer
and checkpoint directory.

Usage:
    torchrun --standalone --nproc-per-node=N distillation_dual.py \
        --teacher_ckpt $TEACHER_CKPT \
        --train_dataset "LunarDataset(...) + LunarDataset(...)" \
        --output_dir $OUTPUT_DIR/s1 \
        --s2_output_dir $OUTPUT_DIR/s2 \
        [options]

See README.md for full documentation and worked examples.
"""

import argparse
import datetime
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

torch.backends.cuda.matmul.allow_tf32 = True

# ===== MASt3R / DUSt3R imports =====
import mast3r.utils.path_to_dust3r  # noqa: F401
from mast3r.model import AsymmetricMASt3R
from dust3r.losses import *                    # noqa: F401,F403  (Regr3D, ConfLoss, MultiLoss …)
from mast3r.losses import (
    ConfMatchingLoss, MatchingLoss, APLoss, Regr3D, InfoNCE, Regr3D_ScaleShiftInv,
)
from mast3r.datasets import (
    ARKitScenes, BlendedMVS, Co3d, MegaDepth, ScanNetpp,
    StaticThings3D, Waymo, WildRGBD, LunarDataset,
)
from mast3r.model import load_model as load_mast3r_teacher

import dust3r.training
import dust3r.datasets

# Expose MASt3R symbols into dust3r namespaces (required by dust3r helpers)
dust3r.training.AsymmetricMASt3R = AsymmetricMASt3R
dust3r.training.Regr3D = Regr3D
dust3r.training.Regr3D_ScaleShiftInv = Regr3D_ScaleShiftInv
dust3r.training.MatchingLoss = MatchingLoss
dust3r.training.ConfMatchingLoss = ConfMatchingLoss
dust3r.training.InfoNCE = InfoNCE
dust3r.training.APLoss = APLoss
dust3r.datasets.ARKitScenes = ARKitScenes
dust3r.datasets.BlendedMVS = BlendedMVS
dust3r.datasets.Co3d = Co3d
dust3r.datasets.MegaDepth = MegaDepth
dust3r.datasets.ScanNetpp = ScanNetpp
dust3r.datasets.StaticThings3D = StaticThings3D
dust3r.datasets.Waymo = Waymo
dust3r.datasets.WildRGBD = WildRGBD
dust3r.datasets.LunarDataset = LunarDataset
dust3r.datasets.MoonTest = LunarDataset

from dust3r.inference import loss_of_one_batch
from dust3r.datasets import get_data_loader
from dust3r.datasets.utils.transforms import ColorJitter
import croco.utils.misc as misc
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler

from distillation.students import (
    build_mobilenet_student,
    build_vit_student,
    build_vit_tiny_student,
    build_dinov3_student,
)
from distillation.svd_init import svd_init_student_from_teacher
from distillation.feature_loss import build_feature_loss, DepthAnythingFeatureAlignLoss
from distillation.losses import DepthGradientLoss, RelationalFeatureDistillation, TeacherConfLoss


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class ZeroLoss(nn.Module):
    """Dummy loss that returns 0 (used for teacher-only forward passes)."""

    def forward(self, *args, **kwargs):
        dev = next((a.device for a in args if torch.is_tensor(a)), torch.device('cpu'))
        return torch.tensor(0.0, device=dev), {}


@torch.no_grad()
def build_pseudo_gt_cam1(view1, view2, pred1_T, pred2_T, conf_thresh: float = 0.0):
    """Build pseudo-GT dicts from teacher predictions.

    Sets camera_pose = Identity (all points are already in camera-1 frame).
    Optionally filters low-confidence pixels when conf_thresh > 0.

    Args:
        view1, view2: raw batch views.
        pred1_T, pred2_T: teacher prediction dicts (pts3d, conf, …).
        conf_thresh: minimum teacher confidence to include a pixel.

    Returns:
        (gt1_t, gt2_t): pseudo-GT dicts suitable as input to student criterion.
    """
    B = pred1_T['pts3d'].shape[0]
    I = torch.eye(4, device=pred1_T['pts3d'].device).view(1, 4, 4).expand(B, 4, 4)

    pts1 = pred1_T['pts3d']
    pts2 = pred2_T.get('pts3d_in_other_view', pred2_T['pts3d'])

    mask1 = view1.get('valid_mask', torch.ones_like(pts1[..., 0], dtype=torch.bool))
    mask2 = view2.get('valid_mask', torch.ones_like(pts2[..., 0], dtype=torch.bool))

    def _get_conf(pred):
        for k in ('conf', 'confidence', 'conf_map'):
            if k in pred and torch.is_tensor(pred[k]):
                return pred[k]
        return None

    c1, c2 = _get_conf(pred1_T), _get_conf(pred2_T)
    if conf_thresh > 0:
        if c1 is not None:
            if c1.dim() == 4 and c1.shape[1] != 1:
                c1 = c1.mean(1, keepdim=True)
            mask1 = mask1 & (c1.squeeze(1) > conf_thresh)
        if c2 is not None:
            if c2.dim() == 4 and c2.shape[1] != 1:
                c2 = c2.mean(1, keepdim=True)
            mask2 = mask2 & (c2.squeeze(1) > conf_thresh)

    gt1_t = dict(view1, pts3d=pts1, valid_mask=mask1, camera_pose=I)
    gt2_t = dict(view2, pts3d=pts2, valid_mask=mask2, camera_pose=I)

    # Cache raw teacher confidence for optional TeacherConfLoss
    if c1 is not None:
        gt1_t['teacher_conf'] = c1.squeeze(1) if c1.dim() == 4 else c1
    if c2 is not None:
        gt2_t['teacher_conf'] = c2.squeeze(1) if c2.dim() == 4 else c2

    return gt1_t, gt2_t


def ddp_getattr(m, name, default=None):
    """Get attribute from a DDP-wrapped or plain model."""
    return getattr(m.module, name, default) if hasattr(m, 'module') else getattr(m, name, default)


def _align_feat_batches(f_a, f_b):
    """Handle batch-size doubling from symmetrize_batch=True in teacher."""
    if f_a is None or f_b is None:
        return f_a, f_b
    Ba, Bb = f_a.shape[0], f_b.shape[0]
    if Ba == 2 * Bb:
        return f_a[:Bb], f_b
    if Bb == 2 * Ba:
        return f_a, f_b[:Ba]
    return f_a, f_b


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def build_loader(dataset_str: str, batch_size: int, num_workers: int, test: bool = False):
    """Build a DataLoader from a dataset specification string.

    Supports "N @ Dataset(...)" syntax to subsample N pairs.
    """
    import re
    m = re.match(r'^\s*(\d[\d_]*)\s*@\s*(.+)$', dataset_str)
    if m:
        max_samples = int(m.group(1).replace('_', ''))
        dataset = eval(m.group(2))
        if hasattr(dataset, 'pairs') and len(dataset) > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(dataset.pairs), max_samples, replace=False)
            dataset.pairs = dataset.pairs[idx]
    else:
        dataset = eval(dataset_str)
    return get_data_loader(dataset, batch_size=batch_size, num_workers=num_workers,
                           pin_mem=True, shuffle=not test, drop_last=not test)


# ---------------------------------------------------------------------------
# Feature KD helper
# ---------------------------------------------------------------------------

def _compute_feat_kd(student, teacher, feature_loss, kd_feat_weight: float,
                     device, student_name: str = '') -> torch.Tensor:
    """Compute feature alignment KD loss for one student."""
    if feature_loss is None or kd_feat_weight <= 0:
        return torch.tensor(0., device=device)

    feat_s = ddp_getattr(student, 'last_feat_encoder', None)
    feat_t = ddp_getattr(teacher, 'last_feat_encoder', None)
    if feat_s is None or feat_t is None:
        return torch.tensor(0., device=device)

    if feat_s.shape[2:] != feat_t.shape[2:]:
        feat_t = F.interpolate(feat_t, size=feat_s.shape[-2:], mode='bilinear', align_corners=False)
    feat_s, feat_t = _align_feat_batches(feat_s, feat_t)
    if feat_s is None or feat_t is None:
        return torch.tensor(0., device=device)

    feat_s = feat_s.float()
    feat_t = feat_t.float().detach()
    return feature_loss(feat_s, feat_t, student_name=student_name) * kd_feat_weight


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch_dual(students_info, teacher, criterion, data_loader,
                         optimizers, device, epoch: int, loss_scaler, args,
                         log_writers=None, feature_loss=None, gradient_loss=None,
                         teacher_conf_criterion=None, rfd_module=None):
    """Train all active students for one epoch.

    Args:
        students_info: list of (name, wrapped_model, model_without_ddp, output_dir, weight).
        teacher: frozen teacher model.
        criterion: training criterion (MASt3R-style).
        data_loader: training DataLoader.
        optimizers: dict {student_name: optimizer}.
        device: torch.device.
        epoch: current epoch index.
        loss_scaler: NativeScaler for AMP.
        args: parsed arguments.
        log_writers: dict {student_name: SummaryWriter}.
        feature_loss: DepthAnythingFeatureAlignLoss or MixedFeatureAlignLoss, or None.
        gradient_loss: DepthGradientLoss or None.
        teacher_conf_criterion: TeacherConfLoss-based criterion or None.
        rfd_module: RelationalFeatureDistillation or None.
    """
    for _, s_wrapped, _, _, _ in students_info:
        s_wrapped.train(True)
    teacher.eval()

    metric_logger = misc.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    for sname, _, _, _, _ in students_info:
        metric_logger.add_meter(f'loss_{sname}', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
    metric_logger.add_meter('epoch', misc.SmoothedValue(window_size=1, fmt='{value:.3f}'))

    accum_iter = args.accum_iter
    for opt in (optimizers.values() if isinstance(optimizers, dict) else [optimizers]):
        if opt is not None:
            opt.zero_grad()

    first_opt = next((o for o in optimizers.values() if o is not None), None)
    lr_display = first_opt.param_groups[0]['lr'] if first_opt is not None else 0.0
    metric_logger.update(lr=lr_display)

    # Cache exclusion sets on args to avoid per-step parsing
    if not hasattr(args, '_no_feat_set'):
        args._no_feat_set = set(
            s.strip() for s in getattr(args, 'no_feat_students', '').split(',') if s.strip())
    if not hasattr(args, '_no_teacher_conf_set'):
        args._no_teacher_conf_set = set(
            s.strip() for s in getattr(args, 'no_teacher_conf_students', '').split(',') if s.strip())
    if not hasattr(args, '_no_rfd_set'):
        args._no_rfd_set = set(
            s.strip() for s in getattr(args, 'no_rfd_students', '').split(',') if s.strip())
    if not hasattr(args, '_no_grad_set'):
        args._no_grad_set = set(
            s.strip() for s in getattr(args, 'no_grad_students', '').split(',') if s.strip())

    header = f'Epoch: [{epoch}]'
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        epoch_f = epoch + data_iter_step / len(data_loader)
        if data_iter_step % accum_iter == 0:
            for opt in (optimizers.values() if isinstance(optimizers, dict) else [optimizers]):
                if opt is not None:
                    misc.adjust_learning_rate(opt, epoch_f, args)

        # Teacher forward (shared, no grad)
        with torch.no_grad():
            out_T = loss_of_one_batch(batch, teacher, ZeroLoss().to(device),
                                      device, symmetrize_batch=True, use_amp=False, ret=None)
            gt1_t, gt2_t = build_pseudo_gt_cam1(
                out_T['view1'], out_T['view2'],
                out_T['pred1'], out_T['pred2'],
                conf_thresh=args.teacher_conf_thresh,
            )

        batch_student = (gt1_t, gt2_t)
        loss_total = torch.tensor(0., device=device)
        per_student = []
        log_kwargs = dict(epoch=epoch_f, lr=lr_display)

        for sname, s_wrapped, _, _, sweight in students_info:
            use_teacher_conf = (teacher_conf_criterion is not None
                                and sname not in args._no_teacher_conf_set)
            student_criterion = teacher_conf_criterion if use_teacher_conf else criterion

            out_S = loss_of_one_batch(batch_student, s_wrapped, student_criterion, device,
                                      symmetrize_batch=False, use_amp=bool(args.amp), ret=None)
            loss_s, details_s = out_S['loss']

            # Feature alignment
            if sname in args._no_feat_set:
                loss_feat = torch.tensor(0., device=device)
            else:
                loss_feat = _compute_feat_kd(
                    s_wrapped, teacher, feature_loss, args.kd_feat_weight, device,
                    student_name=sname)

            # Gradient smoothness
            loss_grad = torch.tensor(0., device=device)
            if (gradient_loss is not None and args.lambda_grad > 0
                    and epoch >= args.grad_loss_start_epoch
                    and sname not in args._no_grad_set):
                try:
                    pts3d_s1 = out_S.get('pred1', {}).get('pts3d', None)
                    if pts3d_s1 is not None:
                        loss_grad = gradient_loss(pts3d_s1, gt1_t['pts3d'],
                                                  gt1_t.get('valid_mask')) * args.lambda_grad
                        pts3d_s2 = out_S.get('pred2', {}).get('pts3d', None)
                        if pts3d_s2 is not None:
                            loss_grad = loss_grad + gradient_loss(
                                pts3d_s2, gt2_t['pts3d'], gt2_t.get('valid_mask')) * args.lambda_grad
                except Exception:
                    loss_grad = torch.tensor(0., device=device)

            # RFD (optional)
            loss_rfd = torch.tensor(0., device=device)
            if rfd_module is not None and args.rfd_weight > 0 and sname not in args._no_rfd_set:
                try:
                    feat_s = ddp_getattr(s_wrapped, 'last_feat_encoder', None)
                    feat_t = ddp_getattr(teacher, 'last_feat_encoder', None)
                    if feat_s is not None and feat_t is not None:
                        if feat_t.shape[-2:] != feat_s.shape[-2:]:
                            feat_t = F.interpolate(feat_t.float(), size=feat_s.shape[-2:],
                                                   mode='bilinear', align_corners=False)
                        if feat_t.shape[0] > feat_s.shape[0]:
                            feat_t = feat_t[:feat_s.shape[0]]
                        mid = feat_s.shape[-1] // 2
                        loss_rfd = rfd_module(
                            feat_s[..., :mid], feat_s[..., mid:],
                            feat_t[..., :mid], feat_t[..., mid:],
                        ) * args.rfd_weight
                except Exception:
                    pass

            student_loss = sweight * (loss_s + loss_feat + loss_grad + loss_rfd)
            per_student.append((sname, s_wrapped, student_loss))
            loss_total = loss_total + student_loss

            details_float = {
                f'{sname}_{k}': (float(v.detach()) if torch.is_tensor(v) else float(v))
                for k, v in details_s.items()
            }
            log_kwargs[f'loss_{sname}'] = float(loss_s)
            log_kwargs[f'feat_{sname}'] = float(loss_feat)
            log_kwargs[f'grad_{sname}'] = float(loss_grad)
            log_kwargs.update(details_float)

        # Per-student NaN detection (DDP-safe: all ranks must agree)
        n_students = len(per_student)
        _nan_flags = torch.zeros(n_students, dtype=torch.int32, device=device)
        for si, (_, _, sloss) in enumerate(per_student):
            if not math.isfinite(float(sloss.detach())):
                _nan_flags[si] = 1
        if getattr(args, 'distributed', False):
            torch.distributed.all_reduce(_nan_flags, op=torch.distributed.ReduceOp.MAX)
        skip_student = [bool(_nan_flags[si].item()) for si in range(n_students)]

        if all(skip_student):
            for opt in (optimizers.values() if isinstance(optimizers, dict) else [optimizers]):
                if opt is not None:
                    opt.zero_grad()
            continue

        update_now = (data_iter_step + 1) % accum_iter == 0
        for si, (sname, s_wrapped, student_loss) in enumerate(per_student):
            opt = optimizers.get(sname) if isinstance(optimizers, dict) else optimizers
            if opt is None or skip_student[si]:
                if opt is not None:
                    opt.zero_grad()
                continue
            params = [p for pg in opt.param_groups for p in pg['params']]
            loss_scaler(student_loss / accum_iter, opt,
                        clip_grad=getattr(args, 'clip_grad', None) or None,
                        parameters=params, update_grad=update_now)

        if update_now:
            for opt in (optimizers.values() if isinstance(optimizers, dict) else [optimizers]):
                if opt is not None:
                    opt.zero_grad()

        if getattr(args, 'distributed', False):
            torch.distributed.barrier()

        loss_value = sum(
            float(sl.detach()) for si, (_, _, sl) in enumerate(per_student)
            if not skip_student[si]
        )
        log_kwargs['loss'] = loss_value
        metric_logger.update(**log_kwargs)

        epoch_1000x = int(epoch_f * 1000)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if ((data_iter_step + 1) % (accum_iter * args.print_freq)) == 0 and log_writers:
            for sname, _, _, _, _ in students_info:
                if sname in log_writers:
                    log_writers[sname].add_scalar('train_loss_total', loss_value_reduce, epoch_1000x)
                    opt = optimizers.get(sname) if isinstance(optimizers, dict) else optimizers
                    lr_val = opt.param_groups[0]['lr'] if opt is not None else 0.0
                    log_writers[sname].add_scalar('train_lr', lr_val, epoch_1000x)
                    for k, v in log_kwargs.items():
                        if k.startswith(f'{sname}_') and isinstance(v, (int, float)):
                            log_writers[sname].add_scalar(f'train_{k}', v, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_one_epoch(student, criterion, data_loader, device, epoch: int, args,
                   log_writer=None, prefix: str = 'test'):
    """Evaluate a student for one epoch."""
    student.eval()
    torch.manual_seed((args.seed + epoch * 1000 + misc.get_rank()) % (2 ** 31))
    np.random.seed(torch.randint(0, 2 ** 32, (1,)).item())

    metric_logger = misc.MetricLogger(delimiter='  ')
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9 ** 9))
    header = f'{prefix} Epoch: [{epoch}]'

    for _, batch in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        result = loss_of_one_batch(batch, student, criterion, device,
                                   symmetrize_batch=True, use_amp=bool(args.amp), ret=None)
        loss_result = result['loss']
        if isinstance(loss_result, tuple) and len(loss_result) == 2:
            loss_value, loss_details = loss_result
        else:
            loss_value, loss_details = loss_result, {}
        metric_logger.update(loss=float(loss_value), **loss_details)

    metric_logger.synchronize_between_processes()
    print(f'[{prefix}] Averaged stats:', metric_logger)

    aggs = [('avg', 'global_avg'), ('med', 'median')]
    results = {
        f'{k}_{tag}': getattr(meter, attr)
        for k, meter in metric_logger.meters.items()
        for tag, attr in aggs
    }
    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(prefix + '_' + name, val, 1000 * epoch)
    return results


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_student_checkpoint(model_without_ddp, optimizer, scaler, args, epoch,
                             output_dir: str, fname: str, best_so_far: float):
    """Save a student checkpoint to output_dir/checkpoint-{fname}.pth."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    to_save = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'scaler': scaler.state_dict() if scaler is not None else None,
        'args': args,
        'epoch': epoch,
        'best_so_far': best_so_far,
    }
    misc.save_on_master(to_save, output_dir / f'checkpoint-{fname}.pth')


def load_student_checkpoint(model, optimizer, scaler, output_dir: str,
                             device: str = 'cpu') -> Tuple[int, float]:
    """Load checkpoint-last.pth from output_dir if it exists.

    Returns:
        (epoch, best_so_far) or (0, inf) if no checkpoint found.
    """
    ckpt_path = Path(output_dir) / 'checkpoint-last.pth'
    if not ckpt_path.exists():
        print(f'  No checkpoint at {ckpt_path}')
        return 0, float('inf')
    print(f'  Loading checkpoint from {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('model', ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'    WARNING: missing keys: {missing[:5]}')
    if unexpected:
        print(f'    WARNING: unexpected keys: {unexpected[:5]}')
    if optimizer is not None and ckpt.get('optimizer') is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scaler is not None and ckpt.get('scaler') is not None:
        scaler.load_state_dict(ckpt['scaler'])
    epoch = ckpt.get('epoch', 0)
    best = ckpt.get('best_so_far', float('inf'))
    print(f'    Resumed from epoch {epoch}, best_so_far={best:.4f}')
    return epoch, best


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args_parser():
    p = argparse.ArgumentParser('MASt3R multi-student distillation', add_help=True)

    # Data
    p.add_argument('--train_dataset', required=True, type=str,
                   help='Training dataset spec (Python expression, e.g. '
                        '"LunarDataset(split=\'train\', ROOT=\'$DATA_ROOT/...\', ...)"). '
                        'Multiple datasets can be concatenated with "+".')
    p.add_argument('--test_dataset', default='[None]', type=str,
                   help='Test/validation dataset spec (same format as --train_dataset).')

    # Teacher
    p.add_argument('--teacher_ckpt', required=True, type=str,
                   help='Path to the teacher MASt3R checkpoint (.pth).')
    p.add_argument('--freeze_teacher', action='store_true', default=True,
                   help='Freeze all teacher parameters (default: True).')

    # Student 1 (MobileNetV3)
    p.add_argument('--s1_dec_embed_dim', type=int, default=512,
                   help='Decoder hidden dimension for student 1.')
    p.add_argument('--s1_dec_depth', type=int, default=6,
                   help='Number of decoder blocks for student 1.')
    p.add_argument('--s1_dec_heads', type=int, default=4,
                   help='Number of decoder attention heads for student 1.')
    p.add_argument('--s1_mlp_ratio', type=float, default=1.0,
                   help='MLP expansion ratio in decoder blocks for student 1.')
    p.add_argument('--s1_backbone_name', type=str, default='mobilenetv3_large_100',
                   help='timm model name for the MobileNet backbone of student 1.')
    p.add_argument('--s1_no_pretrain_backbone', action='store_true', default=False,
                   help='Disable ImageNet pretraining for student 1 backbone (ablation).')
    p.add_argument('--disable_s1', action='store_true', default=False,
                   help='Disable student 1 (MobileNet).')
    p.add_argument('--s1_weight', type=float, default=1.0,
                   help='Loss weight for student 1.')
    p.add_argument('--output_dir', default='./output_s1/', type=str,
                   help='Checkpoint output directory for student 1.')

    # Student 2 (ViT-Small / DINOv2)
    p.add_argument('--s2_dec_embed_dim', type=int, default=512)
    p.add_argument('--s2_dec_depth', type=int, default=6)
    p.add_argument('--s2_dec_heads', type=int, default=4)
    p.add_argument('--s2_mlp_ratio', type=float, default=1.0)
    p.add_argument('--s2_prefer_dinov2', action='store_true', default=True,
                   help='Use DINOv2 pretrained weights for student 2 ViT backbone.')
    p.add_argument('--s2_use_dinov3', action='store_true', default=False,
                   help='Use DINOv3 (HuggingFace) instead of DINOv2 for student 2.')
    p.add_argument('--s2_dinov3_model', type=str,
                   default='facebook/dinov3-vits16-pretrain-lvd1689m',
                   help='DINOv3 HuggingFace model name for student 2.')
    p.add_argument('--s2_freeze_backbone', action='store_true', default=False,
                   help='Freeze student 2 ViT backbone.')
    p.add_argument('--s2_no_pretrain_backbone', action='store_true', default=False,
                   help='Disable DINOv2 pretraining for student 2 (ablation).')
    p.add_argument('--disable_s2', action='store_true', default=False,
                   help='Disable student 2 (ViT-Small).')
    p.add_argument('--s2_weight', type=float, default=1.0)
    p.add_argument('--s2_output_dir', default='./output_s2/', type=str,
                   help='Checkpoint output directory for student 2.')

    # Students 3-11: optional, enabled by --enable_s{N}
    for N in range(3, 12):
        p.add_argument(f'--enable_s{N}', action='store_true', default=False,
                       help=f'Enable optional student {N}.')
        p.add_argument(f'--s{N}_dec_embed_dim', type=int, default=512 if N not in (4, 6, 7, 9, 11) else
                       (256 if N == 4 else 384))
        p.add_argument(f'--s{N}_dec_depth', type=int, default=6 if N != 4 else 4)
        p.add_argument(f'--s{N}_dec_heads', type=int, default=4 if N not in (6, 7, 11) else 6)
        p.add_argument(f'--s{N}_mlp_ratio', type=float, default=1.0)
        p.add_argument(f'--s{N}_freeze_backbone', action='store_true', default=(N == 3))
        p.add_argument(f'--s{N}_weight', type=float, default=1.0)
        p.add_argument(f'--s{N}_output_dir', default=f'./output_s{N}/', type=str,
                       help=f'Checkpoint output directory for student {N}.')
        if N in (2, 3, 4, 5, 8, 9, 11):
            p.add_argument(f'--s{N}_use_dinov3', action='store_true', default=False)
            p.add_argument(f'--s{N}_dinov3_model', type=str,
                           default='facebook/dinov3-vits16-pretrain-lvd1689m')
        if N in (5, 6, 7, 10):
            p.add_argument(f'--s{N}_backbone_type', type=str, default='dune',
                           choices=['dinov2', 'dune'],
                           help=f'Backbone type for student {N}.')
        if N == 5:
            p.add_argument('--s5_model_name', type=str, default='vit_tiny_patch16_224',
                           help='timm model name for ViT-Tiny backbone (student 5).')
        if N in (3, 4):
            p.add_argument(f'--s{N}_prefer_dinov2', action='store_true', default=True)

    # Loss weights and KD options
    p.add_argument('--train_criterion', type=str,
                   default="ConfLoss(Regr3D(L21, norm_mode='?avg_dis'), alpha=0.2) + "
                           "0.075*ConfMatchingLoss(MatchingLoss(InfoNCE(mode='proper', "
                           "temperature=0.05), negatives_padding=0, blocksize=8192), "
                           "alpha=10.0, confmode='mean')",
                   help='Training criterion (Python expression, eval\'d at runtime).')
    p.add_argument('--test_criterion', type=str, default=None,
                   help='Test criterion (defaults to --train_criterion).')
    p.add_argument('--teacher_conf_thresh', type=float, default=0.0,
                   help='Minimum teacher confidence to include a pseudo-GT pixel.')
    p.add_argument('--kd_feat_weight', type=float, default=0.0,
                   help='Weight for feature alignment loss (0 = disabled).')
    p.add_argument('--kd_feat_mode', type=str, default='cosine_margin',
                   choices=['cosine_margin', 'mixed'],
                   help="Feature alignment mode: 'cosine_margin' (DA-V2, recommended) "
                        "or 'mixed' (MSE + cosine + attention-transfer).")
    p.add_argument('--kd_feat_margin', type=float, default=0.9,
                   help='Tolerance margin alpha for cosine_margin mode. '
                        'Pixels with cos(f_s,f_t) >= alpha are excluded (DA-V2 default=0.9).')
    p.add_argument('--lambda_grad', type=float, default=0.0,
                   help='Weight for gradient smoothness loss (0 = disabled). '
                        'Recommended: 0.1-1.0.')
    p.add_argument('--grad_loss_start_epoch', type=int, default=0,
                   help='Epoch at which to start applying the gradient smoothness loss.')
    p.add_argument('--no_grad_students', type=str, default='',
                   help='Comma-separated student names to exclude from gradient loss.')
    p.add_argument('--no_feat_students', type=str, default='',
                   help='Comma-separated student names to exclude from feature alignment loss.')
    p.add_argument('--teacher_conf_regr3d', action='store_true', default=False,
                   help='Replace ConfLoss(Regr3D) with teacher-confidence-weighted Regr3D '
                        '(Distill3R-style).')
    p.add_argument('--conf_distill_gamma', type=float, default=0.2,
                   help='Weight for confidence distillation L1 term in TeacherConfLoss.')
    p.add_argument('--no_teacher_conf_students', type=str, default='',
                   help='Students to exclude from TeacherConfLoss (use standard ConfLoss).')
    p.add_argument('--rfd_weight', type=float, default=0.0,
                   help='Weight for Relational Feature Distillation loss (0 = disabled).')
    p.add_argument('--rfd_temperature', type=float, default=0.1,
                   help='Softmax temperature for RFD correlation matrix.')
    p.add_argument('--rfd_n_samples', type=int, default=512,
                   help='Number of spatial tokens to subsample for RFD.')
    p.add_argument('--no_rfd_students', type=str, default='',
                   help='Comma-separated student names to exclude from RFD.')

    # SVD initialization
    p.add_argument('--svd_init', action='store_true', default=False,
                   help='Warm-start student decoders from teacher via truncated SVD.')
    p.add_argument('--teacher_dec_depth', type=int, default=12,
                   help='Number of decoder blocks in the teacher (for SVD layer mapping).')
    p.add_argument('--svd_layer_mapping', type=str, default='uniform',
                   choices=['uniform', 'last_k', 'first_k'],
                   help="Layer mapping strategy for SVD init: 'uniform' (default), "
                        "'last_k' (last S teacher layers), 'first_k' (first S teacher layers).")
    p.add_argument('--svd_compression_mode', type=str, default='svd',
                   choices=['svd', 'direct_truncation'],
                   help="Compression method: 'svd' (truncated SVD, default) or "
                        "'direct_truncation' (top-left submatrix, ablation baseline).")
    p.add_argument('--no_svd_students', type=str, default='',
                   help='Comma-separated student names to skip SVD init.')

    # Training hyperparameters
    p.add_argument('--seed', default=0, type=int, help='Random seed.')
    p.add_argument('--batch_size', default=8, type=int, help='Per-GPU batch size.')
    p.add_argument('--accum_iter', default=4, type=int,
                   help='Gradient accumulation steps.')
    p.add_argument('--epochs', default=60, type=int, help='Total training epochs.')
    p.add_argument('--start_epoch', default=0, type=int)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--lr', type=float, default=None,
                   help='Absolute learning rate (overrides --blr if set).')
    p.add_argument('--blr', type=float, default=1.5e-4,
                   help='Base learning rate (scaled by effective batch size / 256).')
    p.add_argument('--min_lr', type=float, default=0.)
    p.add_argument('--warmup_epochs', type=int, default=10)
    p.add_argument('--clip_grad', type=float, default=1.0,
                   help='Max gradient norm for clipping (0 to disable).')
    p.add_argument('--amp', type=int, default=1, choices=[0, 1],
                   help='Use Automatic Mixed Precision (1 = enabled).')
    p.add_argument('--disable_cudnn_benchmark', action='store_true', default=False)
    p.add_argument('--num_workers', default=4, type=int)

    # Distributed training
    p.add_argument('--world_size', default=1, type=int)
    p.add_argument('--local_rank', default=-1, type=int)
    p.add_argument('--dist_url', default='env://')

    # Logging and checkpointing
    p.add_argument('--eval_freq', type=int, default=1,
                   help='Evaluate every N epochs.')
    p.add_argument('--save_freq', default=1, type=int,
                   help='Save checkpoint-last.pth every N epochs.')
    p.add_argument('--keep_freq', default=5, type=int,
                   help='Save numbered checkpoint every N epochs.')
    p.add_argument('--print_freq', default=20, type=int,
                   help='Log every N batches.')
    p.add_argument('--resume', action='store_true', default=False,
                   help='Auto-resume from checkpoint-last.pth in each output_dir.')

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    misc.init_distributed_mode(args)
    global_rank = misc.get_rank()
    args.distributed = args.world_size > 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Feature alignment loss
    feature_loss = build_feature_loss(
        mode=getattr(args, 'kd_feat_mode', 'cosine_margin'),
        alpha=getattr(args, 'kd_feat_margin', 0.9),
    ).to(device)

    # Gradient smoothness loss
    gradient_loss = None
    if getattr(args, 'lambda_grad', 0) > 0:
        gradient_loss = DepthGradientLoss(depth_weight=1.0, xy_weight=0.1).to(device)

    # Teacher-confidence Regr3D
    teacher_conf_criterion = None
    if getattr(args, 'teacher_conf_regr3d', False):
        gamma = getattr(args, 'conf_distill_gamma', 0.2)
        teacher_conf_criterion = (
            TeacherConfLoss(Regr3D(L21, norm_mode='?avg_dis'), gamma=gamma)
            + 0.075 * ConfMatchingLoss(
                MatchingLoss(InfoNCE(mode='proper', temperature=0.05),
                             negatives_padding=0, blocksize=8192),
                alpha=10.0, confmode='mean')
        ).to(device)

    # RFD
    rfd_module = None
    if getattr(args, 'rfd_weight', 0) > 0:
        rfd_module = RelationalFeatureDistillation(
            temperature=args.rfd_temperature,
            n_samples=args.rfd_n_samples,
        )

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = not args.disable_cudnn_benchmark

    # Data loaders
    train_loader = build_loader(args.train_dataset, args.batch_size, args.num_workers, test=False)

    def _test_loader_name(ds_str: str) -> str:
        import re
        m = re.search(r"ROOT='[^']*?/([^/']+)'", ds_str.strip())
        return m.group(1) if m else ds_str.strip().split('(')[0]

    test_loaders = {
        _test_loader_name(ds): build_loader(ds, args.batch_size, args.num_workers, test=True)
        for ds in args.test_dataset.split('+')
    }

    # Teacher
    print('Loading teacher from:', args.teacher_ckpt)
    teacher = load_mast3r_teacher(args.teacher_ckpt, device=device, verbose=True)
    if args.freeze_teacher:
        for p in teacher.parameters():
            p.requires_grad_(False)
    teacher.to(device)

    # Pre-cache hub models on rank 0 (prevents race conditions in DDP)
    if args.distributed and global_rank == 0:
        try:
            torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        except Exception:
            pass
        try:
            import timm
            timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0, img_size=512)
        except Exception:
            pass
    if args.distributed:
        torch.distributed.barrier()

    # ---- Build students ----
    students_info = []  # (name, model, dec_depth, output_dir, weight)

    if not getattr(args, 'disable_s1', False):
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        s1 = build_mobilenet_student(
            device=device,
            dec_embed_dim=args.s1_dec_embed_dim,
            dec_depth=args.s1_dec_depth,
            dec_num_heads=args.s1_dec_heads,
            mlp_ratio=args.s1_mlp_ratio,
            backbone_name=args.s1_backbone_name,
            pretrained_backbone=not getattr(args, 's1_no_pretrain_backbone', False),
        )
        students_info.append(('S1_MobileNet', s1, args.s1_dec_depth, args.output_dir, args.s1_weight))

    if not getattr(args, 'disable_s2', False):
        Path(args.s2_output_dir).mkdir(parents=True, exist_ok=True)
        if getattr(args, 's2_use_dinov3', False):
            s2 = build_dinov3_student(
                device=device, dec_embed_dim=args.s2_dec_embed_dim,
                dec_depth=args.s2_dec_depth, dec_num_heads=args.s2_dec_heads,
                mlp_ratio=args.s2_mlp_ratio, model_name=args.s2_dinov3_model,
                freeze_backbone=args.s2_freeze_backbone)
        else:
            s2 = build_vit_student(
                device=device, dec_embed_dim=args.s2_dec_embed_dim,
                dec_depth=args.s2_dec_depth, dec_num_heads=args.s2_dec_heads,
                mlp_ratio=args.s2_mlp_ratio, prefer_dinov2=args.s2_prefer_dinov2,
                freeze_backbone=args.s2_freeze_backbone,
                pretrained_backbone=not getattr(args, 's2_no_pretrain_backbone', False))
        students_info.append(('S2_ViT-Small', s2, args.s2_dec_depth, args.s2_output_dir, args.s2_weight))

    for N in range(3, 12):
        if not getattr(args, f'enable_s{N}', False):
            continue
        sdir = getattr(args, f's{N}_output_dir')
        Path(sdir).mkdir(parents=True, exist_ok=True)
        dec_dim = getattr(args, f's{N}_dec_embed_dim')
        dec_depth = getattr(args, f's{N}_dec_depth')
        dec_heads = getattr(args, f's{N}_dec_heads')
        mlp_ratio = getattr(args, f's{N}_mlp_ratio')
        freeze_bb = getattr(args, f's{N}_freeze_backbone', False)
        sweight = getattr(args, f's{N}_weight', 1.0)
        use_dinov3 = getattr(args, f's{N}_use_dinov3', False)
        bb_type = getattr(args, f's{N}_backbone_type', None)

        if use_dinov3:
            smodel = build_dinov3_student(
                device=device, dec_embed_dim=dec_dim, dec_depth=dec_depth,
                dec_num_heads=dec_heads, mlp_ratio=mlp_ratio,
                model_name=getattr(args, f's{N}_dinov3_model',
                                   'facebook/dinov3-vits16-pretrain-lvd1689m'),
                freeze_backbone=freeze_bb)
        elif bb_type in ('dinov2', 'dune'):
            smodel = build_vit_student(
                device=device, dec_embed_dim=dec_dim, dec_depth=dec_depth,
                dec_num_heads=dec_heads, mlp_ratio=mlp_ratio,
                backbone_type=bb_type, freeze_backbone=freeze_bb)
        elif N == 5:
            smodel = build_vit_tiny_student(
                device=device, dec_embed_dim=dec_dim, dec_depth=dec_depth,
                dec_num_heads=dec_heads, mlp_ratio=mlp_ratio,
                model_name=getattr(args, 's5_model_name', 'vit_tiny_patch16_224'),
                freeze_backbone=freeze_bb)
        else:
            smodel = build_vit_student(
                device=device, dec_embed_dim=dec_dim, dec_depth=dec_depth,
                dec_num_heads=dec_heads, mlp_ratio=mlp_ratio,
                prefer_dinov2=getattr(args, f's{N}_prefer_dinov2', True),
                freeze_backbone=freeze_bb)

        name_map = {3: 'ViT-Small-frozen', 4: 'ViT-Small-reduced',
                    5: 'ViT-Tiny', 6: 'Distill3R', 7: 'Hybrid',
                    8: 'DINOv3-full', 9: 'DINOv3-reduced',
                    10: 'DUNE-full', 11: 'DINOv3-Distill3R'}
        sname = f'S{N}_{name_map.get(N, f"custom{N}")}'
        students_info.append((sname, smodel, dec_depth, sdir, sweight))

    if not students_info:
        raise ValueError('No students enabled. Enable at least one with --enable_s{N} '
                         'or without --disable_s1 / --disable_s2.')

    # Print parameter counts
    for sname, smodel, _, _, _ in students_info:
        total = sum(p.numel() for p in smodel.parameters())
        trainable = sum(p.numel() for p in smodel.parameters() if p.requires_grad)
        print(f'{sname}: {total:,} total params, {trainable:,} trainable')

    # SVD initialization
    if getattr(args, 'svd_init', False) and not getattr(args, 'resume', False):
        no_svd = set(s.strip() for s in getattr(args, 'no_svd_students', '').split(',') if s.strip())
        print('SVD-initialising student decoders from teacher...')
        teacher_sd = {k: v.cpu() for k, v in teacher.state_dict().items()}
        for sname, smodel, sdec_depth, _, _ in students_info:
            if sname in no_svd:
                print(f'  Skipping SVD init for {sname}')
                continue
            print(f'  Initialising {sname} (dec_depth={sdec_depth})...')
            svd_init_student_from_teacher(
                smodel, teacher_sd,
                student_dec_depth=sdec_depth,
                teacher_dec_depth=args.teacher_dec_depth,
                verbose=True,
                layer_mapping_strategy=getattr(args, 'svd_layer_mapping', 'uniform'),
                compression_mode=getattr(args, 'svd_compression_mode', 'svd'),
            )
        del teacher_sd
        print('SVD init done.')
    elif getattr(args, 'svd_init', False) and getattr(args, 'resume', False):
        print('SVD init skipped (--resume is active; checkpoint will overwrite decoder weights).')

    # DDP wrapping
    models_without_ddp = {}
    students_wrapped = {}
    for sname, smodel, _, _, _ in students_info:
        if args.distributed:
            wrapped = torch.nn.parallel.DistributedDataParallel(
                smodel, device_ids=[args.gpu], find_unused_parameters=True, static_graph=True)
            models_without_ddp[sname] = wrapped.module
        else:
            wrapped = smodel
            models_without_ddp[sname] = smodel
        students_wrapped[sname] = wrapped

    students_info_wrapped = [
        (sname, students_wrapped[sname], models_without_ddp[sname], sdir, sweight)
        for sname, _, _, sdir, sweight in students_info
    ]

    # Criterion
    train_criterion = eval(args.train_criterion).to(device)
    test_criterion = eval(args.test_criterion or args.train_criterion).to(device)

    # Learning rate
    eff_bsz = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_bsz / 256
    print(f'Effective batch size: {eff_bsz},  lr: {args.lr:.2e}')

    # Per-student optimizers
    optimizers = {}
    for sname, _, m_no_ddp, _, _ in students_info_wrapped:
        param_groups = misc.get_parameter_groups(m_no_ddp, args.weight_decay)
        if param_groups:
            optimizers[sname] = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
        else:
            optimizers[sname] = None

    # Add per-student feature projector to optimizer param groups
    no_feat = set(s.strip() for s in getattr(args, 'no_feat_students', '').split(',') if s.strip())
    if isinstance(feature_loss, DepthAnythingFeatureAlignLoss) and args.kd_feat_weight > 0:
        teacher_enc_dim = 1024  # ViT-L/14 encoder dimension
        for sname, _, _, _, _ in students_info_wrapped:
            if sname in no_feat:
                continue
            # Determine student encoder dim from dec_embed_dim
            for prefix in [f's{i}' for i in range(1, 12)]:
                si = sname.split('_')[0]  # e.g. 'S2'
                if si.lower() == prefix:
                    sdim = getattr(args, f'{prefix}_dec_embed_dim', None)
                    if sdim is None:
                        break
                    proj = feature_loss.get_proj(sname, sdim, teacher_enc_dim, device)
                    if proj is not None and optimizers.get(sname) is not None:
                        optimizers[sname].add_param_group({
                            'params': list(proj.parameters()),
                            'lr': args.lr,
                            'weight_decay': 0.0,
                        })
                    break

    loss_scaler = NativeScaler()
    best = {sname: float('inf') for sname, _, _, _, _ in students_info_wrapped}
    log_writers = {}
    if global_rank == 0:
        for sname, _, _, sdir, _ in students_info_wrapped:
            log_writers[sname] = SummaryWriter(log_dir=sdir)

    # Resume
    if getattr(args, 'resume', False):
        resumed_epochs = []
        for sname, _, m_no_ddp, sdir, _ in students_info_wrapped:
            ep, best_val = load_student_checkpoint(
                m_no_ddp, optimizers[sname], loss_scaler, sdir, device='cpu')
            resumed_epochs.append(ep)
            best[sname] = best_val
        max_epoch = max(resumed_epochs) if resumed_epochs else 0
        if max_epoch > 0:
            args.start_epoch = max_epoch + 1
            print(f'Resuming training from epoch {args.start_epoch}')

    print(f'Starting distillation: {len(students_info_wrapped)} student(s), '
          f'{args.epochs} epochs')
    start_time = time.time()
    train_stats: dict = {}
    test_stats = {sname: {} for sname, _, _, _, _ in students_info_wrapped}

    for epoch in range(args.start_epoch, args.epochs + 1):
        # Save checkpoint-last before training (MASt3R convention)
        if epoch > args.start_epoch:
            if args.save_freq and epoch % args.save_freq == 0 or epoch == args.epochs:
                for sname, _, m_no_ddp, sdir, _ in students_info_wrapped:
                    save_student_checkpoint(
                        m_no_ddp, optimizers[sname], loss_scaler, args,
                        epoch - 1, sdir, 'last', best[sname])

        # Evaluation
        new_best = {sname: False for sname in best}
        if epoch > 0 and args.eval_freq > 0 and epoch % args.eval_freq == 0:
            for ds_name, loader in test_loaders.items():
                if args.distributed and hasattr(loader, 'sampler') and \
                        hasattr(loader.sampler, 'set_epoch'):
                    loader.sampler.set_epoch(epoch)
                for sname, s_wrapped, _, _, _ in students_info_wrapped:
                    stats = test_one_epoch(
                        s_wrapped, test_criterion, loader, device, epoch, args,
                        log_writers.get(sname), prefix=f'{ds_name}_{sname}')
                    test_stats[sname][ds_name] = stats
                    if stats['loss_med'] < best[sname]:
                        best[sname] = stats['loss_med']
                        new_best[sname] = True

        # Logging
        if misc.is_main_process():
            if log_writers:
                for writer in log_writers.values():
                    writer.flush()
            for sname, _, _, sdir, _ in students_info_wrapped:
                log_entry = dict(epoch=epoch,
                                 **{f'train_{k}': v for k, v in train_stats.items()})
                for ds_name in test_loaders:
                    if ds_name in test_stats.get(sname, {}):
                        log_entry.update({f'{ds_name}_{k}': v
                                          for k, v in test_stats[sname][ds_name].items()})
                with open(os.path.join(sdir, 'log.txt'), 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry) + '\n')

        # Keep / best checkpoints
        if epoch > args.start_epoch:
            if args.keep_freq and epoch % args.keep_freq == 0:
                for sname, _, m_no_ddp, sdir, _ in students_info_wrapped:
                    save_student_checkpoint(
                        m_no_ddp, optimizers[sname], loss_scaler, args,
                        epoch - 1, sdir, str(epoch), best[sname])
            for sname, _, m_no_ddp, sdir, _ in students_info_wrapped:
                if new_best[sname]:
                    save_student_checkpoint(
                        m_no_ddp, optimizers[sname], loss_scaler, args,
                        epoch - 1, sdir, 'best', best[sname])

        if epoch >= args.epochs:
            break

        if args.distributed and hasattr(train_loader, 'sampler') and \
                hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch_dual(
            students_info_wrapped, teacher, train_criterion, train_loader,
            optimizers, device, epoch, loss_scaler, args,
            log_writers if global_rank == 0 else None,
            feature_loss if args.kd_feat_weight > 0 else None,
            gradient_loss if getattr(args, 'lambda_grad', 0) > 0 else None,
            teacher_conf_criterion=teacher_conf_criterion,
            rfd_module=rfd_module,
        )

    total_time = time.time() - start_time
    print('Training time:', str(datetime.timedelta(seconds=int(total_time))))

    # Final save
    for sname, _, m_no_ddp, sdir, _ in students_info_wrapped:
        save_student_checkpoint(
            m_no_ddp, optimizers[sname], loss_scaler, args,
            args.epochs, sdir, 'final', best[sname])


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
