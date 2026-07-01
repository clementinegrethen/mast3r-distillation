#!/bin/bash
# Example: distill MASt3R into two students on StereoLunar (Moon) data.
#
# Prerequisites:
#   - $TEACHER_CKPT  : path to the MASt3R teacher checkpoint (e.g. MOONSt3R.pth)
#   - $DATA_ROOT     : root of the StereoLunar dataset
#                      expected sub-directories: nadir_clean_0_90, nadir_clean_90_180,
#                      nadir_clean_180_270, pitch_clean_0_90, pitch_clean_90_180,
#                      pitch_clean_180_270, pitch_clean_ajout, landingimages
#   - $OUTPUT_DIR    : where checkpoints and logs will be written
#
# Adjust --nproc-per-node to the number of available GPUs.

set -euo pipefail

: "${TEACHER_CKPT:?Set TEACHER_CKPT to the path of the teacher .pth checkpoint}"
: "${DATA_ROOT:?Set DATA_ROOT to the root of the StereoLunar dataset}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the desired checkpoint output directory}"

export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# NCCL tuning for multi-GPU (adjust interface if needed)
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

N_GPUS=4  # set to match your allocation

torchrun --standalone --nproc-per-node=${N_GPUS} distillation_dual.py \
  --teacher_ckpt "${TEACHER_CKPT}" \
  --freeze_teacher \
  \
  --train_dataset \
    "LunarDataset(split='train',ROOT='${DATA_ROOT}/nadir_clean_0_90',split_file='train.npz',resolution=512,n_corres=8192,nneg=0.5,transform=ColorJitter,aug_crop='auto') + \
     LunarDataset(split='train',ROOT='${DATA_ROOT}/nadir_clean_90_180',split_file='train.npz',resolution=512,n_corres=8192,nneg=0.5,transform=ColorJitter,aug_crop='auto') + \
     LunarDataset(split='train',ROOT='${DATA_ROOT}/nadir_clean_180_270',split_file='train.npz',resolution=512,n_corres=8192,nneg=0.5,transform=ColorJitter,aug_crop='auto') + \
     LunarDataset(split='train',ROOT='${DATA_ROOT}/pitch_clean_0_90',split_file='train.npz',resolution=512,n_corres=8192,nneg=0.5,transform=ColorJitter,aug_crop='auto') + \
     LunarDataset(split='train',ROOT='${DATA_ROOT}/pitch_clean_90_180',split_file='train.npz',resolution=512,n_corres=8192,nneg=0.5,transform=ColorJitter,aug_crop='auto') + \
     LunarDataset(split='train',ROOT='${DATA_ROOT}/pitch_clean_180_270',split_file='train.npz',resolution=512,n_corres=8192,nneg=0.5,transform=ColorJitter,aug_crop='auto') + \
     LunarDataset(split='train',ROOT='${DATA_ROOT}/pitch_clean_ajout',split_file='train.npz',resolution=512,n_corres=8192,nneg=0.5,transform=ColorJitter,aug_crop='auto') + \
     LunarDataset(split='train',ROOT='${DATA_ROOT}/landingimages',split_file='train.npz',resolution=512,n_corres=8192,nneg=0.5,transform=ColorJitter,aug_crop='auto')" \
  \
  --test_dataset \
    "LunarDataset(split='test',ROOT='${DATA_ROOT}/nadir_clean_0_90',split_file='test.npz',resolution=512,n_corres=8192,nneg=0.5) + \
     LunarDataset(split='test',ROOT='${DATA_ROOT}/pitch_clean_0_90',split_file='test.npz',resolution=512,n_corres=8192,nneg=0.5)" \
  \
  --lr 3e-5 --min_lr 2e-7 --warmup_epochs 5 --epochs 60 \
  --batch_size 1 --accum_iter 4 \
  --save_freq 5 --keep_freq 10 --eval_freq 5 --print_freq 50 \
  --amp 1 --disable_cudnn_benchmark \
  \
  --s1_dec_embed_dim 512 --s1_dec_depth 6 --s1_dec_heads 4 --s1_mlp_ratio 1.0 \
  --output_dir "${OUTPUT_DIR}/s1_mobilenet" \
  \
  --s2_dec_embed_dim 512 --s2_dec_depth 6 --s2_dec_heads 4 --s2_mlp_ratio 1.0 \
  --s2_prefer_dinov2 \
  --s2_output_dir "${OUTPUT_DIR}/s2_vit_small" \
  \
  --svd_init --teacher_dec_depth 12 \
  --kd_feat_weight 0.1 --kd_feat_mode cosine_margin --kd_feat_margin 0.9 \
  --lambda_grad 1.0 \
  --teacher_conf_thresh 0.2
