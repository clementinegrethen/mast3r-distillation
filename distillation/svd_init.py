"""
SVD-based decoder initialization for MASt3R student models.

Compresses teacher decoder weights into the student's smaller dimensional space
via truncated SVD (Eckart-Young theorem), providing a warm start that is
strictly better in the Frobenius-norm sense than random initialization.

Inspired by: Lillama (arXiv:2412.16719).
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Low-level compression primitives
# ---------------------------------------------------------------------------

def svd_compress_weight(W_teacher: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
    """Compress a 2-D teacher weight into target_shape via truncated SVD.

    W_T = U diag(sigma) V^T  (thin SVD)
    W_S = U[:d_out_S, :r] diag(sigma[:r]) V^T[:r, :d_in_S]
    where r = min(d_out_S, d_in_S, rank(W_T)).

    Args:
        W_teacher: shape (d_out_T, d_in_T)
        target_shape: (d_out_S, d_in_S)

    Returns:
        Compressed weight of shape target_shape, same dtype as W_teacher.
    """
    assert W_teacher.ndim == 2, f"Expected 2-D weight, got {W_teacher.shape}"
    d_out_S, d_in_S = target_shape

    U, S, Vt = torch.linalg.svd(W_teacher.float().cpu(), full_matrices=False)
    r = min(d_out_S, d_in_S, S.shape[0])

    W_student = (U[:d_out_S, :r] * S[:r].unsqueeze(0)) @ Vt[:r, :d_in_S]
    return W_student.to(W_teacher.dtype)


def direct_truncate_weight(W_teacher: torch.Tensor, target_shape: Tuple[int, ...]) -> torch.Tensor:
    """Compress a 2-D teacher weight by taking the top-left submatrix.

    Simpler than SVD but discards the global structure captured by singular
    vectors. Useful as an ablation baseline.

    Args:
        W_teacher: shape (d_out_T, d_in_T)
        target_shape: (d_out_S, d_in_S)

    Returns:
        Truncated weight of shape target_shape.
    """
    assert W_teacher.ndim == 2
    d_out_S, d_in_S = target_shape
    return W_teacher[:d_out_S, :d_in_S].clone()


def _truncate_vector(v: torch.Tensor, target_len: int) -> torch.Tensor:
    """Truncate a 1-D bias / norm vector to target_len."""
    return v[:target_len].clone()


# ---------------------------------------------------------------------------
# Layer mapping strategies
# ---------------------------------------------------------------------------

def layer_mapping(teacher_depth: int, student_depth: int, strategy: str = "uniform"):
    """Map student decoder layer indices to teacher decoder layer indices.

    Args:
        teacher_depth: number of decoder blocks in the teacher.
        student_depth: number of decoder blocks in the student.
        strategy:
            'uniform'  - evenly spaced (default): student i <- teacher floor(i*T/S)
            'last_k'   - take the last S teacher layers
            'first_k'  - take the first S teacher layers

    Returns:
        List of length student_depth with teacher layer indices.
    """
    if strategy == "last_k":
        start = teacher_depth - student_depth
        return list(range(start, teacher_depth))
    elif strategy == "first_k":
        return list(range(student_depth))
    else:  # uniform
        return [int(i * teacher_depth / student_depth) for i in range(student_depth)]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@torch.no_grad()
def svd_init_student_from_teacher(
    student: nn.Module,
    teacher_state: dict,
    student_dec_depth: int = 6,
    teacher_dec_depth: int = 12,
    verbose: bool = True,
    layer_mapping_strategy: str = "uniform",
    compression_mode: str = "svd",
) -> nn.Module:
    """Initialize a student model's decoder, heads, and shared params from the teacher.

    The student encoder (MobileNet / ViT) is NOT touched; only decoder blocks,
    decoder norms, and downstream heads are initialized.

    Args:
        student: student nn.Module, already on target device.
        teacher_state: teacher state_dict (CPU tensors are fine).
        student_dec_depth: number of decoder blocks in the student.
        teacher_dec_depth: number of decoder blocks in the teacher.
        verbose: print progress and statistics.
        layer_mapping_strategy: 'uniform' | 'last_k' | 'first_k'
        compression_mode: 'svd' (optimal low-rank, default) | 'direct_truncation'

    Returns:
        The student module with updated decoder weights (in-place).
    """
    sd_student = student.state_dict()
    lmap = layer_mapping(teacher_dec_depth, student_dec_depth, strategy=layer_mapping_strategy)
    if verbose:
        print(f"[SVD-init] strategy='{layer_mapping_strategy}', "
              f"mapping (student<-teacher): {list(enumerate(lmap))}")

    initialised = 0
    skipped = 0

    compress_2d = svd_compress_weight if compression_mode == "svd" else direct_truncate_weight

    def _try_init(student_key: str, teacher_key: str):
        nonlocal initialised, skipped
        if teacher_key not in teacher_state or student_key not in sd_student:
            skipped += 1
            return
        W_t = teacher_state[teacher_key]
        W_s = sd_student[student_key]
        if not (torch.is_tensor(W_t) and torch.is_tensor(W_s)):
            skipped += 1
            return
        if W_t.shape == W_s.shape:
            sd_student[student_key] = W_t.clone()
            initialised += 1
        elif W_t.ndim == 2 and W_s.ndim == 2:
            sd_student[student_key] = compress_2d(W_t, W_s.shape).to(W_s.device)
            initialised += 1
        elif W_t.ndim == 1 and W_s.ndim == 1 and W_t.shape[0] >= W_s.shape[0]:
            sd_student[student_key] = _truncate_vector(W_t, W_s.shape[0]).to(W_s.device)
            initialised += 1
        elif W_t.ndim >= 3 and W_s.ndim >= 3:
            shape_t, shape_s = W_t.shape, W_s.shape
            W_t_2d = W_t.reshape(shape_t[0], -1)
            W_s_2d_shape = (shape_s[0], int(np.prod(shape_s[1:])))
            if W_t_2d.shape[0] >= shape_s[0] and W_t_2d.shape[1] >= W_s_2d_shape[1]:
                compressed = compress_2d(W_t_2d, W_s_2d_shape)
                sd_student[student_key] = compressed.reshape(shape_s).to(W_s.device)
                initialised += 1
            else:
                skipped += 1
        else:
            skipped += 1

    # Shared params
    for suffix in ['mask_token', 'decoder_embed.weight', 'decoder_embed.bias',
                   'dec_norm.weight', 'dec_norm.bias']:
        _try_init(suffix, suffix)

    # Decoder blocks (both dec_blocks and dec_blocks2 for cross-attention)
    for prefix in ['dec_blocks', 'dec_blocks2']:
        for s_idx, t_idx in enumerate(lmap):
            s_pre = f'{prefix}.{s_idx}.'
            t_pre = f'{prefix}.{t_idx}.'
            for sk in [k for k in sd_student if k.startswith(s_pre)]:
                _try_init(sk, t_pre + sk[len(s_pre):])

    # Downstream heads
    for head in ['downstream_head1', 'downstream_head2']:
        for sk in [k for k in sd_student if k.startswith(head + '.')]:
            _try_init(sk, sk)

    result = student.load_state_dict(sd_student, strict=True)
    if verbose:
        print(f"[SVD-init] done: {initialised} params initialised, {skipped} skipped")
        if result.missing_keys:
            print(f"  missing: {result.missing_keys[:5]}")
        if result.unexpected_keys:
            print(f"  unexpected: {result.unexpected_keys[:5]}")

    return student
