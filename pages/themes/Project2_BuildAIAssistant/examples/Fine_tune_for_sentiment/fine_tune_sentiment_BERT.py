import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)

def prepare_data(model_name, train_size=1000, test_size=500):
    """Downloads the IMDb dataset and tokenizes the text."""
    print("--- Loading and Tokenizing Data ---")
    dataset = load_dataset("imdb")

    # Selecting subsets for quick demonstration
    train_subset = dataset["train"].shuffle(seed=42).select(range(train_size))
    test_subset = dataset["test"].shuffle(seed=42).select(range(test_size))

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length")

    tokenized_train = train_subset.map(tokenize_fn, batched=True)
    tokenized_test = test_subset.map(tokenize_fn, batched=True)

    return tokenized_train, tokenized_test, tokenizer

def initialize_model(model_name):
    """Loads the pre-trained model with a classification head."""
    print("--- Initializing Model ---")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1}
    )
    return model

def get_compute_metrics():
    """Returns a function to calculate accuracy during training."""
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    return compute_metrics

def train_model(model, training_args, train_data, test_data, compute_metrics):
    """Executes the training loop."""
    print("--- Starting Training ---")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer

def run_inference(model, tokenizer, test_reviews):
    """Tests the model on custom text strings."""
    print("\n--- Running Inference ---")
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    for review in test_reviews:
        result = classifier(review)
        print(f"Review: {review}\nResult: {result}\n")

def main():
    # Configuration
    MODEL_NAME = "distilbert-base-uncased"

    # 1. Prepare Data
    train_data, test_data, tokenizer = prepare_data(MODEL_NAME)

    # 2. Initialize Model
    model = initialize_model(MODEL_NAME)

    # 3. Setup Training Config
    training_args = TrainingArguments(
        output_dir="./movie_sentiment_model",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=10,
        save_total_limit=1,
    )

    # 4. Train
    trainer = train_model(
        model,
        training_args,
        train_data,
        test_data,
        get_compute_metrics()
    )

    # 5. Inference
    sample_reviews = [
        "The plot was predictable, but the acting was phenomenal.",
        "I wouldn't recommend this to my worst enemy. Boring!"
    ]
    run_inference(model, tokenizer, sample_reviews)

if __name__ == "__main__":
    main()