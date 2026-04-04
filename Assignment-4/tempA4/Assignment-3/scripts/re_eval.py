import json
from transformers import AutoModelForSequenceClassification, Trainer
from data import load_and_prepare_data
from utils import compute_metrics

def main():
    repo_name = "Despo08/distilbert-goodreads-assignment"
    print(f"Loading model directly from Hugging Face Hub: {repo_name}")
    
    # Load model directly from HF
    model = AutoModelForSequenceClassification.from_pretrained(repo_name)
    
    print("Preparing test data...")
    tokenized_datasets, _ = load_and_prepare_data(sample_size=15000)
    test_dataset = tokenized_datasets["test"]
    
    trainer = Trainer(
        model=model,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("Running evaluation from Hub model...")
    hf_eval_results = trainer.evaluate()
    
    print("Saving results to hf_eval_results.json...")
    with open("hf_eval_results.json", "w") as f:
        json.dump(hf_eval_results, f, indent=4)
        
    print("Hugging Face Hub Evaluation Results:", hf_eval_results)

if __name__ == "__main__":
    main()