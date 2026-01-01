"""
Using a Saved BERT Model for Sentiment Analysis
Demonstrates how to save a Hugging Face model locally and load it for inference.
"""

import os
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

def load_or_download_model(model_name="distilbert-base-uncased-finetuned-sst-2-english", save_dir="models/bert_sentiment"):
    """
    Load the model from local directory if it exists, otherwise download and save it.
    """
    if os.path.exists(save_dir):
        print(f"Loading model from local directory: {save_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(save_dir)
        model = AutoModelForSequenceClassification.from_pretrained(save_dir)
    else:
        print(f"Downloading model '{model_name}' and saving to '{save_dir}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Save locally
        os.makedirs(save_dir, exist_ok=True)
        tokenizer.save_pretrained(save_dir)
        model.save_pretrained(save_dir)
        print("Model saved successfully!")

    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def main():
    print("=" * 60)
    print("BERT SENTIMENT ANALYSIS - LOCAL MODEL USAGE")
    print("=" * 60)

    try:
        # Initialize analyzer
        # Note: You can change the save_dir to any path you prefer
        analyzer = load_or_download_model()

        # Examples
        test_reviews = [
            "This script makes it so easy to use BERT models!",
            "I'm not sure if I like the complexity of transformers, but the results are great.",
            "The loading time is a bit long on my old computer."
        ]

        print("\nRunning test predictions...")
        for review in test_reviews:
            result = analyzer(review)[0]
            label = result['label']
            score = result['score']
            icon = "😊" if label == "POSITIVE" else "😞"
            print(f"\nReview: {review}")
            print(f"Result: {label} {icon} ({score:.2%})")

        # Interactive Mode
        print("\n" + "-" * 60)
        print("Interactive Mode: Enter your own review (or 'q' to quit)")
        while True:
            user_input = input("\nReview > ").strip()
            if user_input.lower() in ['q', 'quit', 'exit']:
                break
            if not user_input:
                continue
                
            result = analyzer(user_input)[0]
            label = result['label']
            score = result['score']
            icon = "😊" if label == "POSITIVE" else "😞"
            print(f"Prediction: {label} {icon} ({score:.2%})")

    except Exception as e:
        print(f"\nError: {e}")
        print("Tip: Ensure you have 'transformers' and 'torch' installed.")

if __name__ == "__main__":
    main()
