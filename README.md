# ML/DL Ops Assignment 3: End-to-End Hugging Face Model Training & Docker Deployment

This repository contains a complete machine learning workflow for fine-tuning a classification model using Hugging Face, containerizing the evaluation pipeline with Docker, and publishing the artifacts.

## 🔗 Project Links
* **GitHub Repository:** https://github.com/m25csa003-glitch/mlops-hf-docker-assignment
* **Hugging Face Model:** https://huggingface.co/Despo08/distilbert-imdb-assignment

---

## 📄 Short Report

### 1. Model Selection
* **Pre-trained Model Used:** DistilBERT (`distilbert-base-uncased`)
* **Reason for Selection:** DistilBERT was selected because it offers an excellent balance between accuracy and computational efficiency. It retains most of BERT's language understanding capabilities while being significantly smaller and faster, making it perfectly suited for text classification tasks and lightweight enough for seamless Docker containerization.

### 2. Training Summary
The model was fine-tuned using the Hugging Face `Trainer` API. The end-to-end pipeline involved:
* Converting the initial Jupyter Notebook into modular Python scripts.
* Preparing the dataset and configuring `TrainingArguments`.
* Successfully executing the training loop and saving the model weights, tokenizer, and training configuration.
* Pushing all final artifacts directly to the Hugging Face Hub.

### 3. Evaluation Comparison
The model was evaluated in two separate environments to ensure consistency and deployment readiness.

* **Local Evaluation (`eval_results.json`):**
  * **Accuracy:** 0.8718 (87.18%)
  * **F1 Score:** 0.8716
  * **Loss:** 0.3131
  * *(Runtime: ~192.18s)*

* **Hugging Face Hub Evaluation via Docker (`hf_eval_results.json`):**
  * **Accuracy:** 0.8718 (87.18%)
  * **F1 Score:** 0.8716
  * **Loss:** 0.3131
  * *(Runtime: ~151.17s)*

**Conclusion:** The accuracy, F1 score, and loss metrics from the local environment and the Dockerized environment pulling from Hugging Face match exactly. This verifies that the model was successfully deployed and functions correctly in a containerized setup. 

### 4. Challenges Faced
* **Docker Image Optimization:** Ensuring all required PyTorch and Hugging Face dependencies were installed correctly without unnecessarily bloating the container size required careful structuring of the `Dockerfile`.
* **Environment Consistency:** Adapting the notebook code into production-ready scripts while maintaining the same environment behaviors (paths, data loading) was a key learning curve.

---

## 🐳 Docker Build & Execution Instructions

### To Build the Main Training Pipeline:
```bash
docker build -t mlops-hf-train -f Dockerfile .