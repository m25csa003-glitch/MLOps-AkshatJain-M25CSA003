import os
from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_prepare_data(sample_size=10000, test_size=0.2):
    print("Loading Goodreads dataset...")
    # Loading all json.gz files from the data directory
    data_files = "data/goodreads_data/*.json.gz"
    
    # Load dataset using Hugging Face datasets library
    dataset = load_dataset('json', data_files=data_files, split='train')
    
    print(f"Original dataset size: {len(dataset)}")
    
    # Shuffle and sample to reduce training time
    dataset = dataset.shuffle(seed=42).select(range(min(sample_size, len(dataset))))
    
    # Filter out empty texts or missing ratings
    dataset = dataset.filter(lambda x: x.get('review_text') is not None and x.get('rating') is not None)
    
    # Convert ratings (0-5) to binary labels (1 for >3, 0 for <=3)
    def map_labels(example):
        example['label'] = 1 if example['rating'] > 3 else 0
        return example
        
    dataset = dataset.map(map_labels)
    
    # Split into train and test sets
    split_dataset = dataset.train_test_split(test_size=test_size, seed=42)
    
    print("Tokenizing data...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    def tokenize_function(examples):
        # max_length 128 rakha hai fast training ke liye
        return tokenizer(examples["review_text"], padding="max_length", truncation=True, max_length=128)
        
    tokenized_datasets = split_dataset.map(tokenize_function, batched=True)
    
    return tokenized_datasets, tokenizer