import json
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from data import load_and_prepare_data
from utils import compute_metrics

def main():
    print("Loading test data...")
    tokenized_datasets, tokenizer = load_and_prepare_data(sample_size=15000) # Same sample size to maintain split
    test_dataset = tokenized_datasets["test"]
    
    print("Loading trained model from ./models/final_model...")
    model = AutoModelForSequenceClassification.from_pretrained("./models/final_model")
    
    trainer = Trainer(
        model=model,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("Running evaluation...")
    eval_results = trainer.evaluate()
    
    print("Saving results to eval_results.json...")
    with open("eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=4)
        
    print("Local Evaluation Results:", eval_results)

if __name__ == "__main__":
    main()