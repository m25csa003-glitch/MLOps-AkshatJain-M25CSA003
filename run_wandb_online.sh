#!/bin/bash
cd ~/DLOps-Assignment-5/Q1
unset WANDB_API_KEY
unset WANDB_MODE

# Baseline
python train_baseline.py --epochs 1 --batch_size 64 --num_workers 1

# 9 LoRA combinations
for rank in 2 4 8; do
  for alpha in 2 4 8; do
    python lora_experiments.py \
      --rank $rank --alpha $alpha \
      --dropout 0.1 --epochs 1 \
      --batch_size 64 --num_workers 1
  done
done

# Optuna best config
python lora_experiments.py \
  --rank 4 --alpha 16 \
  --dropout 0.2 --epochs 1 \
  --lr 0.000504 --batch_size 64 --num_workers 1

echo "All runs synced to WandB!"
