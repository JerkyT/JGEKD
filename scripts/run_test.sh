#!/usr/bin/env bash
set -v

python3 test_cls_scamobjectnn.py \
          --cuda_ops \
          --batch_size 32 \
          --model repsurf.scanobjectnn.repsurf_ssg_umb_2x \
          --epoch 250 \
          --log_dir test_2x \
          --gpus 2 \
          --n_workers 12 \
          --return_center \
          --return_dist \
          --return_polar \
          --group_size 8 \
          --umb_pool sum \
          --num_point 1024 \
          --dataset ScanObjectNN \
          --teacher_check ./log/PointAnalysis/log/ScanObjectNN/2x_JGE_TKD/checkpoints/best_model.pth \