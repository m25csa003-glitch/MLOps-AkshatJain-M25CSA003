#Assignment 4 — Optimizing Transformer Translation with Ray Tune & Optuna

**Student:** M25CSA003  
**Course:** CSL7120  
**Task:** English → Hindi Neural Machine Translation with Hyperparameter Tuning

---

## Results Summary

| Metric | Baseline | Best Tuned Model |
|--------|----------|-----------------|
| Training Time | 56.55 min | **10.53 min** |
| Epochs | 100 | **20** |
| Final Loss | 0.0974 | 0.1367 |
| BLEU Score | 0.7234 | **0.7698** |
| Hardware | CPU | GPU (NVIDIA A30) |

> **Baseline BEAT in just 20 epochs — 5.4x faster with +6.4% better BLEU score!**

---

## Repository Structure

```
├── M25CSA003_ass_4_tuned_en_to_hi.py   # Ray Tune + Optuna sweep code
├── retrain_best.py                      # Retrain with best config found
├── M25CSA003_ass_4_best_model.pth       # Best model weights (BLEU 0.7698)
├── M25CSA003_ass_4_report.pdf           # Assignment report
├── en_to_hi.py                          # Original baseline code
├── baseline_metrics.csv                 # Baseline metrics
└── README.md
```

---

## Approach

### Part 1 — Baseline
Ran the original `en_to_hi.py` without any changes for 100 epochs on CPU.
- **BLEU:** 0.7234 | **Loss:** 0.0974 | **Time:** 56.55 min

### Part 2 — Ray Tune + Optuna Sweep
Refactored training loop into `train_tune(config)` and ran 20 trials with:

| Hyperparameter | Search Space |
|----------------|-------------|
| Learning Rate | loguniform(1e-5, 1e-3) |
| Batch Size | choice([32, 64]) |
| Attention Heads | choice([4, 8]) |
| FeedForward Dim | choice([1024, 2048, 4096]) |
| Dropout Rate | uniform(0.1, 0.4) |
| Num Layers | choice([3, 4, 6]) |

**ASHA Scheduler** (grace_period=5, max_t=40) used alongside Optuna to prune bad trials early.

### Part 3 — Best Configuration
```
Learning Rate  : 0.000480
Batch Size     : 64
Num Heads      : 4
d_ff           : 1024
Dropout        : 0.1874
Num Layers     : 3
```

Using the optimized learning rate and batch size from the sweep — applied to the full baseline architecture — the model beat the baseline in **20 epochs**.

---

## How to Run

### Install Dependencies
```bash
pip install torch ray[tune] optuna nltk pandas
```

### Run Ray Tune Sweep
```bash
python M25CSA003_ass_4_tuned_en_to_hi.py
```

### Retrain Best Model
```bash
python retrain_best.py
```

---

## Key Takeaway
Ray Tune + Optuna identified that a **higher learning rate (0.00048)** with **larger batch size (64)** allows the same baseline architecture to converge **5x faster** while achieving a **higher BLEU score**.
