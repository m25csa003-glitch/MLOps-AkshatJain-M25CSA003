from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_prepare_data(model_name="distilbert-base-uncased"):
    print("Loading full IMDB dataset (25k train / 25k test)...")
    dataset = load_dataset("imdb")

    train_dataset = dataset["train"].shuffle(seed=42)
    test_dataset = dataset["test"].shuffle(seed=42)

    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    print("Tokenizing data (this will take a minute)...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    return tokenized_train, tokenized_test, tokenizer