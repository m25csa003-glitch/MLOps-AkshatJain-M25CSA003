#!/bin/bash
cd ~/DLOps-Assignment-5/Q1
export WANDB_MODE=offline

for rank in 2 4 8; do
  for alpha in 2 4 8; do
    echo "=============================="
    echo "Running: Rank=$rank, Alpha=$alpha"
    echo "=============================="
    python lora_experiments.py \
      --rank $rank \
      --alpha $alpha \
      --dropout 0.1 \
      --epochs 10 \
      --batch_size 64 \
      --lr 1e-4 \
      --num_workers 1
  done
done
echo "All 9 LoRA experiments done!"
