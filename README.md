# ML/DL Ops Assignment 3: End-to-End Hugging Face Model Training & Docker Deployment

## 🔗 Links
* **Hugging Face Model:** [https://huggingface.co/Despo08/distilbert-imdb-assignment](https://huggingface.co/Despo08/distilbert-imdb-assignment)

## 🐳 Docker Image Build Instructions
To build and run the final evaluation Docker image, execute the following commands in the root directory:

```bash
# Build the production image
docker build -f Dockerfile.eval -t hf-eval-prod .

# Run the container (Evaluates automatically on startup)
docker run hf-eval-prod