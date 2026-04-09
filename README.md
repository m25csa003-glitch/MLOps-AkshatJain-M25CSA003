## Environment Setup

## 🔗 Project Links
* **GitHub Repository:** [MLOps-AkshatJain-M25CSA003](https://github.com/m25csa003-glitch/MLOps-AkshatJain-M25CSA003)
* **Weights & Biases (WandB) Dashboard:** [DLOps-Ass5-Q1 Workspace](https://wandb.ai/akshst08092001-iit-jodhpur/DLOps-Ass5-Q1/workspace?nw=nwuserakshst08092001)
* **HuggingFace Model Weights:** [vit-lora-cifar100-best](https://huggingface.co/Despo08/vit-lora-cifar100-best)

### Option 1: Docker (Recommended)
```bash
# Build Docker image
docker build -t dlops-ass5 .

# Run container with GPU
docker run --gpus all -it \
  -v $(pwd):/workspace \
  dlops-ass5 /bin/bash
```

### Option 2: Singularity (HPC Server — Used for Training)

Training was performed inside a Singularity container built from a Docker image
(pytorch/pytorch:2.1.0-cuda11.8) on the college HPC cluster (cn02, NVIDIA A30 GPU),
as Docker was not directly available on the HPC nodes.
```bash
# Pull from DockerHub
singularity pull dlops-ass5.sif docker://pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Run with GPU
singularity exec --nv \
  --bind $(pwd):/workspace \
  dlops-ass5.sif /bin/bash

# Inside container, install requirements
pip install -r requirements.txt
```

### Option 3: Conda (Local)
```bash
conda create -n dlops5 python=3.9 -y
conda activate dlops5
pip install -r requirements.txt
```

---

## WandB + HuggingFace Login
```bash
wandb login
huggingface-cli login
```

---

## Q1: ViT-S + LoRA on CIFAR-100

### Step 1: Baseline (No LoRA)
```bash
cd Q1
python train_baseline.py \
  --epochs 10 \
  --batch_size 64 \
  --lr 1e-3 \
  --save_dir ../weights/Q1 \
  --wandb_project DLOps-Ass5-Q1
```

### Step 2: LoRA Experiments (all 9 combinations)
```bash
for rank in 2 4 8; do
  for alpha in 2 4 8; do
    python lora_experiments.py \
      --rank $rank \
      --alpha $alpha \
      --dropout 0.1 \
      --epochs 10 \
      --batch_size 64 \
      --lr 1e-4 \
      --save_dir ../weights/Q1 \
      --wandb_project DLOps-Ass5-Q1
  done
done
```

### Step 3: Optuna Hyperparameter Search
```bash
python optuna_search.py
```

Best config found: **Rank=4, Alpha=16, Dropout=0.2, LR=0.000504**

### Step 4: Train Best Config (Optuna)
```bash
python lora_experiments.py \
  --rank 4 --alpha 16 --dropout 0.2 \
  --epochs 10 --batch_size 64 \
  --lr 0.000504 \
  --save_dir ../weights/Q1
```

### Step 5: Test Model
```bash
# Test baseline
python test_vit.py --mode baseline \
  --baseline_ckpt ../weights/Q1/baseline_best.pth

# Test best LoRA model
python test_vit.py --mode lora \
  --lora_dir ../weights/Q1/lora_r4_a16
```

---

## Q1 Results

### Experiment 1: Baseline (No LoRA)

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|-----------|---------|
| 1 | 1.0826 | 0.7449 | 71.44% | 78.16% |
| 2 | 0.6454 | 0.6841 | 80.88% | 80.02% |
| 3 | 0.5684 | 0.6769 | 82.81% | 79.91% |
| 4 | 0.5211 | 0.6673 | 84.00% | 80.62% |
| 5 | 0.4768 | 0.6557 | 85.22% | 80.54% |
| 6 | 0.4458 | 0.6678 | 86.14% | 80.62% |
| 7 | 0.4179 | 0.6560 | 87.01% | 80.92% |
| 8 | 0.3976 | 0.6482 | 87.45% | 81.23% |
| 9 | 0.3800 | 0.6475 | 88.06% | 81.41% |
| 10 | 0.3715 | 0.6311 | 88.38% | 81.52% |

### Experiment 2: Rank=2, Alpha=2

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|-----------|---------|
| 1 | 2.4514 | 0.8823 | 46.80% | 77.63% |
| 2 | 0.6531 | 0.5619 | 82.44% | 83.79% |
| 3 | 0.4810 | 0.4831 | 85.97% | 85.28% |
| 4 | 0.4183 | 0.4444 | 87.45% | 86.17% |
| 5 | 0.3850 | 0.4290 | 88.33% | 86.72% |
| 6 | 0.3590 | 0.4166 | 89.00% | 87.14% |
| 7 | 0.3418 | 0.4067 | 89.45% | 87.28% |
| 8 | 0.3331 | 0.4097 | 89.65% | 87.21% |
| 9 | 0.3263 | 0.4089 | 89.95% | 87.15% |
| 10 | 0.3214 | 0.3980 | 89.96% | 87.64% |

### Experiment 3: Rank=2, Alpha=4

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|-----------|---------|
| 1 | 2.3071 | 0.7885 | 49.52% | 79.99% |
| 2 | 0.6045 | 0.5304 | 83.38% | 85.00% |
| 3 | 0.4637 | 0.4622 | 86.18% | 86.35% |
| 4 | 0.4072 | 0.4348 | 87.81% | 86.93% |
| 5 | 0.3723 | 0.4186 | 88.53% | 87.45% |
| 6 | 0.3489 | 0.4031 | 89.12% | 87.59% |
| 7 | 0.3309 | 0.3959 | 89.63% | 87.72% |
| 8 | 0.3228 | 0.3942 | 90.04% | 88.10% |
| 9 | 0.3171 | 0.4000 | 90.03% | 87.75% |
| 10 | 0.3138 | 0.3983 | 90.19% | 87.81% |

### Experiment 4: Rank=2, Alpha=8

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|-----------|---------|
| 1 | 2.2608 | 0.7609 | 50.99% | 80.62% |
| 2 | 0.5880 | 0.5283 | 83.61% | 84.83% |
| 3 | 0.4531 | 0.4625 | 86.58% | 86.28% |
| 4 | 0.4016 | 0.4328 | 87.66% | 86.92% |
| 5 | 0.3631 | 0.4136 | 88.93% | 87.62% |
| 6 | 0.3362 | 0.4031 | 89.55% | 87.78% |
| 7 | 0.3198 | 0.4021 | 90.00% | 87.57% |
| 8 | 0.3113 | 0.3972 | 90.36% | 88.20% |
| 9 | 0.3034 | 0.3939 | 90.51% | 88.16% |
| 10 | 0.2996 | 0.3981 | 90.51% | 87.92% |

### Experiment 5: Rank=4, Alpha=2

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|-----------|---------|
| 1 | 2.4669 | 0.8639 | 46.76% | 78.40% |
| 2 | 0.6345 | 0.5537 | 82.81% | 83.85% |
| 3 | 0.4721 | 0.4717 | 86.00% | 85.94% |
| 4 | 0.4112 | 0.4459 | 87.56% | 86.35% |
| 5 | 0.3761 | 0.4218 | 88.37% | 87.05% |
| 6 | 0.3536 | 0.4105 | 89.12% | 87.32% |
| 7 | 0.3386 | 0.4140 | 89.40% | 87.19% |
| 8 | 0.3295 | 0.4037 | 89.63% | 87.54% |
| 9 | 0.3207 | 0.3970 | 90.02% | 87.49% |
| 10 | 0.3165 | 0.4058 | 90.17% | 87.69% |

### Experiment 6: Rank=4, Alpha=4

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|-----------|---------|
| 1 | 2.3471 | 0.7819 | 49.02% | 80.04% |
| 2 | 0.5935 | 0.5267 | 83.66% | 84.83% |
| 3 | 0.4570 | 0.4630 | 86.45% | 86.15% |
| 4 | 0.4016 | 0.4470 | 87.66% | 86.05% |
| 5 | 0.3672 | 0.4174 | 88.73% | 87.31% |
| 6 | 0.3462 | 0.4105 | 89.10% | 87.23% |
| 7 | 0.3309 | 0.4038 | 89.42% | 87.50% |
| 8 | 0.3214 | 0.3956 | 89.78% | 87.75% |
| 9 | 0.3111 | 0.3941 | 90.36% | 88.00% |
| 10 | 0.3092 | 0.3988 | 90.39% | 87.73% |

### Experiment 7: Rank=4, Alpha=8

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|-----------|---------|
| 1 | 2.1792 | 0.7361 | 52.01% | 80.64% |
| 2 | 0.5747 | 0.5228 | 83.81% | 84.78% |
| 3 | 0.4429 | 0.4575 | 86.87% | 86.43% |
| 4 | 0.3906 | 0.4382 | 88.09% | 86.97% |
| 5 | 0.3604 | 0.4125 | 88.91% | 87.19% |
| 6 | 0.3335 | 0.3963 | 89.55% | 87.86% |
| 7 | 0.3170 | 0.3959 | 90.12% | 87.90% |
| 8 | 0.3063 | 0.3893 | 90.45% | 88.13% |
| 9 | 0.2985 | 0.3910 | 90.64% | 88.00% |
| 10 | 0.2925 | 0.3882 | 90.87% | 88.12% |

### Experiment 8: Rank=8, Alpha=2

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|-----------|---------|
| 1 | 2.4157 | 0.8471 | 47.96% | 78.76% |
| 2 | 0.6312 | 0.5438 | 82.49% | 84.10% |
| 3 | 0.4735 | 0.4738 | 85.92% | 85.48% |
| 4 | 0.4143 | 0.4362 | 87.48% | 86.57% |
| 5 | 0.3780 | 0.4314 | 88.39% | 86.58% |
| 6 | 0.3536 | 0.4185 | 88.92% | 86.98% |
| 7 | 0.3410 | 0.4089 | 89.35% | 87.35% |
| 8 | 0.3279 | 0.4028 | 89.89% | 87.26% |
| 9 | 0.3244 | 0.3959 | 89.77% | 87.60% |
| 10 | 0.3227 | 0.4007 | 90.08% | 87.33% |

### Experiment 9: Rank=8, Alpha=4

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|-----------|---------|
| 1 | 2.3654 | 0.8024 | 49.21% | 79.61% |
| 2 | 0.6071 | 0.5331 | 83.30% | 84.23% |
| 3 | 0.4578 | 0.4660 | 86.46% | 85.84% |
| 4 | 0.4013 | 0.4338 | 87.66% | 86.57% |
| 5 | 0.3675 | 0.4154 | 88.74% | 87.13% |
| 6 | 0.3448 | 0.4072 | 89.38% | 87.30% |
| 7 | 0.3267 | 0.3983 | 89.83% | 87.54% |
| 8 | 0.3156 | 0.3985 | 90.30% | 87.71% |
| 9 | 0.3101 | 0.3952 | 90.39% | 87.66% |
| 10 | 0.3077 | 0.3958 | 90.47% | 87.82% |

### Experiment 10: Rank=8, Alpha=8

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|-----------|---------|
| 1 | 2.2525 | 0.7477 | 50.83% | 80.54% |
| 2 | 0.5732 | 0.5140 | 84.02% | 84.85% |
| 3 | 0.4410 | 0.4504 | 86.85% | 86.68% |
| 4 | 0.3871 | 0.4267 | 88.25% | 87.26% |
| 5 | 0.3524 | 0.4045 | 89.17% | 87.87% |
| 6 | 0.3304 | 0.4016 | 89.77% | 87.70% |
| 7 | 0.3116 | 0.3901 | 90.24% | 88.17% |
| 8 | 0.3068 | 0.3904 | 90.39% | 88.04% |
| 9 | 0.2927 | 0.3886 | 90.68% | 87.82% |
| 10 | 0.2925 | 0.3861 | 90.83% | 88.15% |

### Optuna Best Config: Rank=4, Alpha=16, Dropout=0.2, LR=0.000504

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|----------|-----------|---------|
| 1 | 0.8650 | 0.4408 | 77.39% | 86.69% |
| 2 | 0.3514 | 0.4059 | 89.02% | 87.41% |
| 3 | 0.2799 | 0.3921 | 91.07% | 88.31% |
| 4 | 0.2269 | 0.3950 | 92.70% | 88.09% |
| 5 | 0.1880 | 0.3904 | 93.90% | 88.59% |
| 6 | 0.1565 | 0.3818 | 94.96% | 88.65% |
| 7 | 0.1341 | 0.3792 | 95.80% | 88.84% |
| 8 | 0.1155 | 0.3804 | 96.44% | 88.99% |
| 9 | 0.1033 | 0.3731 | 96.97% | 89.12% |
| 10 | 0.0965 | 0.3763 | 97.21% | 89.03% |

### Test Results Summary

| LoRA | Rank | Alpha | Dropout | Test Acc | Trainable Params |
|------|------|-------|---------|----------|-----------------|
| No | - | - | - | 80.92% | 38,500 |
| Yes | 2 | 2 | 0.1 | 87.99% | 75,364 |
| Yes | 2 | 4 | 0.1 | 88.19% | 75,364 |
| Yes | 2 | 8 | 0.1 | 88.28% | 75,364 |
| Yes | 4 | 2 | 0.1 | 88.30% | 112,228 |
| Yes | 4 | 4 | 0.1 | 88.05% | 112,228 |
| Yes | 4 | 8 | 0.1 | 88.61% | 112,228 |
| Yes | 8 | 2 | 0.1 | 88.07% | 185,956 |
| Yes | 8 | 4 | 0.1 | 88.39% | 185,956 |
| Yes | 8 | 8 | 0.1 | 88.70% | 185,956 |
| Yes (Optuna Best) | 4 | 16 | 0.2 | 89.96% | 112,228 |

---

## Q2: Adversarial Attacks (IBM ART)

### Step 1: Train ResNet18 from Scratch
```bash
cd Q2
python train_resnet18.py \
  --epochs 50 \
  --batch_size 128 \
  --lr 0.1 \
  --save_dir ../weights/Q2 \
  --wandb_project DLOps-Ass5-Q2
```

### Step 2: FGSM from Scratch
```bash
python fgsm_scratch.py \
  --ckpt ../weights/Q2/resnet18_best.pth \
  --epsilons 0.01 0.02 0.05 0.1 0.15 0.2 0.3 \
  --save_dir ../weights/Q2
```

### Step 3: FGSM using IBM ART
```bash
python fgsm_art.py \
  --ckpt ../weights/Q2/resnet18_best.pth \
  --epsilons 0.01 0.02 0.05 0.1 0.15 0.2 0.3 \
  --save_dir ../weights/Q2
```

### Step 4: PGD Detector
```bash
python detector_pgd.py \
  --victim_ckpt ../weights/Q2/resnet18_best.pth \
  --epochs 20 --eps 0.05 \
  --save_dir ../weights/Q2
```

### Step 5: BIM Detector
```bash
python detector_bim.py \
  --victim_ckpt ../weights/Q2/resnet18_best.pth \
  --epochs 20 --eps 0.05 \
  --save_dir ../weights/Q2
```

---

## Q2 Results

### ResNet18 Training: Test Accuracy = 93.41% ✅ (Target ≥ 72%)

### FGSM Comparison

| Epsilon | Clean Acc | FGSM Scratch | FGSM ART |
|---------|-----------|-------------|----------|
| 0.01 | 93.41% | 39.89% | 44.40% |
| 0.02 | 93.41% | 32.18% | 37.70% |
| 0.05 | 93.41% | 23.84% | 25.80% |
| 0.10 | 93.41% | 12.49% | 14.10% |
| 0.15 | 93.41% | 10.62% | ~11.30% |
| 0.20 | 93.41% | 10.10% | - |
| 0.30 | 93.41% | 10.00% | - |

### Detector Results

| Attack | Detection Acc | Target | Status |
|--------|--------------|--------|--------|
| PGD | 99.33% | ≥ 70% | ✅ PASSED |
| BIM | 99.53% | ≥ 70% | ✅ PASSED |

---

## Hardware Used

- **Training:** NVIDIA A30 (24GB VRAM), CUDA 12.8
- **Development:** Apple MacBook Air M4 (16GB RAM)
- **Container:** Singularity 4.3.1 (Docker image: pytorch/pytorch:2.1.0-cuda11.8)
