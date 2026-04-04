from transformers import AutoModelForSequenceClassification, AutoTokenizer

def main():
    repo_name = "Despo08/distilbert-goodreads-assignment"
    print(f"Uploading model and tokenizer to {repo_name}...")
    
    model = AutoModelForSequenceClassification.from_pretrained("./models/final_model")
    tokenizer = AutoTokenizer.from_pretrained("./models/final_model")
    
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)
    
    print("Upload complete! Model is now live on Hugging Face.")

if __name__ == "__main__":
    main()