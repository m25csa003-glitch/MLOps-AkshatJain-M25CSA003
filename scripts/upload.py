from huggingface_hub import HfApi
import os


repo_id = "Despo08/distilbert-imdb-assignment" 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "distilbert-imdb-final")

api = HfApi()

print(f"Creating repository: {repo_id}")

api.create_repo(repo_id=repo_id, exist_ok=True)

print("Uploading model, tokenizer, and config to Hugging Face...")

api.upload_folder(
    folder_path=model_path,
    repo_id=repo_id,
    repo_type="model",
)
print(f"Model successfully uploaded to: https://huggingface.co/{repo_id}")