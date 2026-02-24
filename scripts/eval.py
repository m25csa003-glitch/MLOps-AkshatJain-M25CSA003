import os
import json
import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from data import load_and_prepare_data
from utils import compute_metrics

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "distilbert-imdb-final")
EVAL_DIR = os.path.join(BASE_DIR, "eval_results")
RESULTS_FILE = os.path.join(BASE_DIR, "eval_results.json")

def evaluate_model():
    model_name = "distilbert-base-uncased"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    _, test_dataset, tokenizer = load_and_prepare_data(model_name)
    
    print(f"Loading trained model from {MODEL_PATH}...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    
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
    
    print("Running full evaluation...")
    results = trainer.evaluate()
    print("\nEvaluation Results:", results)
    
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results successfully saved to {RESULTS_FILE}")

if __name__ == "__main__":
    evaluate_model()