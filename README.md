# Assignment 1 – ML / DL Ops

**Name:** Akshat Jain  
**Roll Number:** M25CSA003  
**Institute:** Indian Institute of Technology Jodhpur  
**Course:** ML-DL-Ops  

---

## 📌 Overview

This repository contains the implementation and experimental analysis for **Assignment 1** of the ML-DL-Ops course.  
The assignment focuses on evaluating deep learning and classical machine learning models on the **MNIST** and **FashionMNIST** datasets.

The following experiments were conducted:

- Deep Learning models: **ResNet-18** and **ResNet-50** (trained from scratch)
- Classical ML baseline: **Support Vector Machines (SVM)**
- Hyperparameter tuning (batch size, optimizer, learning rate)
- CPU vs GPU performance comparison
- Compute analysis using FLOPs

---

## 📂 Files in this Branch (Assignment 1)

- `DLOPs_Assignment1_AkshatJain.ipynb` – Complete Colab notebook
- `M25CSA003_Akshat_Jain_Ass1.pdf` – Final report
- `results.csv` – Experiment results (best runs)
- `loss_plot.png` – Training vs validation loss curves
- `best_model.pth` – Best-performing trained model

---

## 🧠 Q1(a): Deep Learning Experiments

### MNIST – Test Classification Accuracy (%)

| Batch Size | Optimizer | Learning Rate | ResNet-18 | ResNet-50 |
|-----------|----------|---------------|-----------|-----------|
| 16 | SGD | 0.001 | 98.91 | 98.60 |
| 16 | SGD | 0.0001 | 96.56 | 94.72 |
| 16 | Adam | 0.001 | 98.60 | 98.47 |
| 16 | Adam | 0.0001 | 99.37 | 98.78 |
| 32 | SGD | 0.001 | 98.65 | 98.48 |
| 32 | SGD | 0.0001 | 92.54 | 82.73 |
| 32 | Adam | 0.001 | 99.04 | 98.21 |
| 32 | Adam | 0.0001 | 98.84 | 98.55 |

---

### FashionMNIST – Test Classification Accuracy (%)

| Batch Size | Optimizer | Learning Rate | ResNet-18 | ResNet-50 |
|-----------|----------|---------------|-----------|-----------|
| 16 | SGD | 0.001 | 90.29 | 86.77 |
| 16 | SGD | 0.0001 | 81.01 | 84.80 |
| 16 | Adam | 0.001 | 87.97 | 89.35 |
| 16 | Adam | 0.0001 | 91.61 | 90.00 |
| 32 | SGD | 0.001 | 89.13 | 87.00 |
| 32 | SGD | 0.0001 | 87.14 | 88.43 |
| 32 | Adam | 0.001 | 89.57 | 87.29 |
| 32 | Adam | 0.0001 | 90.70 | 89.96 |

---

## 🧪 Q1(b): SVM Experiments

| Dataset | Kernel | C | Gamma | Degree | Accuracy (%) | Train Time (ms) |
|--------|--------|---|-------|--------|---------------|-----------------|
| MNIST | RBF | 1.0 | 0.05 | – | 21.00 | 56173.62 |
| MNIST | RBF | 10.0 | 0.01 | – | 78.55 | 48376.34 |
| MNIST | Poly | 1.0 | – | 2 | 93.10 | 14446.60 |
| MNIST | Poly | 10.0 | – | 3 | 94.10 | 15599.56 |
| FashionMNIST | RBF | 1.0 | 0.05 | – | 30.15 | 55457.67 |
| FashionMNIST | RBF | 10.0 | 0.01 | – | 75.10 | 46412.19 |
| FashionMNIST | Poly | 1.0 | – | 2 | 85.50 | 12954.26 |
| FashionMNIST | Poly | 10.0 | – | 3 | 86.25 | 9698.05 |

---

## ⚙️ Q2: CPU vs GPU Performance (FashionMNIST)

| Compute | Batch | Optimizer | LR | Model | Test Acc (%) | Train Time (ms) | FLOPs (GFLOPs) |
|--------|-------|----------|----|-------|--------------|------------------|---------------|
| CPU | 16 | SGD | 0.001 | ResNet-18 | 90.3 | 2400 | 0.15 |
| CPU | 16 | Adam | 0.001 | ResNet-18 | 88.9 | 2600 | 0.15 |
| GPU | 16 | SGD | 0.001 | ResNet-18 | 90.3 | 300 | 0.15 |
| GPU | 16 | Adam | 0.001 | ResNet-18 | 88.9 | 340 | 0.15 |
| CPU | 16 | SGD | 0.001 | ResNet-50 | 86.8 | 5200 | 0.35 |
| GPU | 16 | Adam | 0.001 | ResNet-50 | 89.3 | 640 | 0.35 |

---

## 📊 Training vs Validation Loss

![Loss Curve](loss_plot.png)

---

## 📌 Conclusion

- ResNet architectures achieve near-saturation accuracy on MNIST.
- FashionMNIST is a more challenging dataset requiring careful tuning.
- Adam optimizer provides more stable convergence than SGD.
- Polynomial SVM kernels outperform RBF kernels on both datasets.
- GPU acceleration significantly reduces training time, especially for deeper models, while FLOPs remain constant across compute platforms.

---

## 🔗 Colab Notebook

- https://colab.research.google.com/drive/1bMCtfQMrX_nlLa-5PSJdox8T6pGFMVjD?usp=sharing
