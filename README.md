# ML/DL Ops Assignment 3: End-to-End Hugging Face Model Training & Docker Deployment

This repository contains a complete machine learning workflow for fine-tuning a classification model using Hugging Face, containerizing the evaluation pipeline with Docker, and publishing the artifacts.

## 🔗 Project Links
* **GitHub Repository:** https://github.com/m25csa003-glitch/mlops-hf-docker-assignment
* **Hugging Face Model:** https://huggingface.co/Despo08/distilbert-goodreads-assignment

---

## 📄 Short Report

### 1. Model Selection
* **Pre-trained Model Used:** DistilBERT (`distilbert-base-uncased`)
* **Dataset:** Goodreads Book Reviews (Sampled for efficiency)
* **Reason for Selection:** DistilBERT was selected because it offers an excellent balance between accuracy and computational efficiency. It is highly capable of understanding the linguistic nuances in book reviews for sentiment classification, while being lightweight enough for seamless Docker containerization.

### 2. Training Summary
The model was fine-tuned using the Hugging Face `Trainer` API. The end-to-end pipeline involved:
* Processing the large `.json.gz` Goodreads dataset, handling missing values, and mapping 1-5 star ratings to binary sentiment labels.
* Tokenizing the data with a max length of 128 for faster processing.
* Training the model using local GPU acceleration (MPS on Mac).
* Pushing the final trained model weights, tokenizer, and config directly to the Hugging Face Hub.

### 3. Evaluation Comparison
The model was evaluated in two separate environments to ensure deployment readiness.

* **Local Evaluation (`eval_results.json`):**
  * **Accuracy:** 78.17% (0.7816)
  * **F1 Score:** 0.8457
  * **Loss:** 0.4521

* **Hugging Face Hub Evaluation via Docker (`hf_eval_results.json`):**
  * **Accuracy:** 78.17% (0.7816)
  * **F1 Score:** 0.8457
  * **Loss:** 0.4521

**Conclusion:** The metrics from the local environment and the Dockerized container (which pulled the model directly from Hugging Face) match exactly. This verifies successful deployment.

### 4. Challenges Faced
* **Data Handling in Docker:** The Goodreads dataset is massive. Copying it directly into the Docker image would have resulted in an excessively large build. This was resolved by using **Docker Volume Mounting** (`-v`), allowing the container to read the dataset directly from the local host without inflating the image size.
* **Library Deprecations:** Addressed recent updates in the Hugging Face `transformers` library, such as replacing the deprecated `evaluation_strategy` with `eval_strategy`.

---

## 🐳 Docker Build & Execution Instructions

### 1. Build the Evaluation Image:
```bash
docker build -t hf-eval-pipeline -f Dockerfile.eval .