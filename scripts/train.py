import os
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from data import load_and_prepare_data
from utils import compute_metrics

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "distilbert-imdb")
FINAL_MODEL_DIR = os.path.join(BASE_DIR, "models", "distilbert-imdb-final")
LOG_DIR = os.path.join(BASE_DIR, "logs")

def train():
    model_name = "distilbert-base-uncased"
    
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n--- Hardware Setup: Using device {device.type.upper()} for fast training! ---\n")

    train_dataset, test_dataset, tokenizer = load_and_prepare_data(model_name)
    
    print("Loading pre-trained model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=LOG_DIR,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    print("Starting full training on 25,000 samples...")
    trainer.train()
    
    print(f"Saving the final model to {FINAL_MODEL_DIR}...")
    trainer.save_model(FINAL_MODEL_DIR)
    print("Training complete!")

if __name__ == "__main__":
    train()