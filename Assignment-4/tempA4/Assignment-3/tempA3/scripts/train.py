import os
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from data import load_and_prepare_data
from utils import compute_metrics

def main():
    print("Preparing data...")
    tokenized_datasets, tokenizer = load_and_prepare_data(sample_size=15000)
    
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    
    training_args = TrainingArguments(
        output_dir="./models/checkpoints",
        eval_strategy="epoch",  # <--- YAHAN CHANGE KIYA HAI
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2, 
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_dir='./logs',
        logging_steps=100,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving final model locally...")
    model.save_pretrained("./models/final_model")
    tokenizer.save_pretrained("./models/final_model")
    print("Training complete and model saved to ./models/final_model")

if __name__ == "__main__":
    main()