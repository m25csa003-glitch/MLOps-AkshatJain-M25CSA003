import os
import json
import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from data import load_and_prepare_data
from utils import compute_metrics

def re_evaluate_hub_model():
    
    repo_id = "Despo08/distilbert-imdb-assignment"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"Downloading and loading model from Hugging Face Hub: {repo_id}...")
    
    
    _, test_dataset, tokenizer = load_and_prepare_data(repo_id)
    
    
    model = AutoModelForSequenceClassification.from_pretrained(repo_id)
    model.to(device)
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    EVAL_DIR = os.path.join(BASE_DIR, "hf_eval_results")
    RESULTS_FILE = os.path.join(BASE_DIR, "hf_eval_results.json")
    
    eval_args = TrainingArguments(
        output_dir=EVAL_DIR,
        per_device_eval_batch_size=16,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("Running evaluation on the downloaded model...")
    results = trainer.evaluate()
    print("\nRe-Evaluation Results from Hub Model:", results)
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results successfully saved to {RESULTS_FILE}")

if __name__ == "__main__":
    re_evaluate_hub_model()