#!/bin/bash
# Example: distill MASt3R into two students on BlendedMVS.
#
# Prerequisites:
#   - $TEACHER_CKPT  : path to the MASt3R teacher checkpoint
#   - $DATA_ROOT     : root of the blendedmvs_processed dataset
#   - $OUTPUT_DIR    : where checkpoints and logs will be written
#
# Adjust --nproc-per-node to the number of available GPUs.

set -euo pipefail

: "${TEACHER_CKPT:?Set TEACHER_CKPT to the path of the teacher .pth checkpoint}"
: "${DATA_ROOT:?Set DATA_ROOT to the root of the BlendedMVS processed dataset}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the desired checkpoint output directory}"

export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

N_GPUS=4  # set to match your allocation

torchrun --standalone --nproc-per-node=${N_GPUS} distillation_dual.py \
  --teacher_ckpt "${TEACHER_CKPT}" \
  --freeze_teacher \
  --disable_s1 \
  \
  --train_dataset \
    "30_000 @ BlendedMVS(split='train',ROOT='${DATA_ROOT}',resolution=(512,384),aug_crop='auto',n_corres=8192,nneg=0.5)" \
  --test_dataset \
    "1_000 @ BlendedMVS(split='val',ROOT='${DATA_ROOT}',resolution=(512,384),seed=777,n_corres=1024)" \
  \
  --lr 3e-5 --min_lr 2e-7 --warmup_epochs 5 --epochs 50 \
  --batch_size 1 --accum_iter 4 \
  --save_freq 1 --keep_freq 5 --eval_freq 5 --print_freq 50 \
  --amp 1 --disable_cudnn_benchmark \
  \
  --s2_dec_embed_dim 512 --s2_dec_depth 6 --s2_dec_heads 4 --s2_mlp_ratio 1.0 \
  --s2_prefer_dinov2 \
  --s2_output_dir "${OUTPUT_DIR}/s2_vit_small" \
  \
  --svd_init --teacher_dec_depth 12 \
  --kd_feat_weight 0.1 --kd_feat_mode cosine_margin --kd_feat_margin 0.9 \
  --lambda_grad 1.0 \
  --teacher_conf_thresh 0.2
