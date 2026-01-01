"""
Sentiment Analysis using Hugging Face Transformers (BERT-based model)
Demonstrates using pre-trained BERT models for high-accuracy sentiment analysis.
"""

import os
from transformers import pipeline

# Note: Install required packages first:
# pip install transformers torch

def run_sentiment_demo():
    print("=" * 80)
    print("SENTIMENT ANALYSIS WITH HUGGING FACE (BERT/DistilBERT)")
    print("=" * 80)

    # 1. Initialize the sentiment analysis pipeline
    # By default, this uses distilbert-base-uncased-finetuned-sst-2-english
    print("\n[1/3] Loading pre-trained model and tokenizer...")
    try:
        sentiment_analyzer = pipeline("sentiment-analysis")
        print("   Model loaded successfully!")
    except Exception as e:
        print(f"   Error loading model: {e}")
        print("   Make sure you have 'transformers' and 'torch' (or 'tensorflow') installed.")
        return

    # 2. Define test reviews
    print("\n[2/3] Preparing test reviews...")
    new_reviews = [
        "This movie was absolutely brilliant! I loved every second!",
        "Waste of time. Terrible acting and boring plot.",
        "Not bad, but could have been better. Average movie.",
        "Fantastic cinematography and great performances!",
        "I regret watching this. Complete disaster.",
        "The plot was complex and intriguing, though the pacing was a bit slow at times.",
        "I've seen better, but it wasn't the worst thing I've watched this year."
    ]

    # 3. Perform Inference
    print("\n" + "=" * 80)
    print("[3/3] Testing with new reviews...")
    print("=" * 80)

    results = sentiment_analyzer(new_reviews)

    for i, (review, result) in enumerate(zip(new_reviews, results), 1):
        label = result['label']
        score = result['score']
        
        # Format the display
        sentiment_icon = "😊" if label == "POSITIVE" else "😞"
        print(f"\nReview {i}:")
        print(f'  Text: "{review}"')
        print(f"  Prediction: {label} {sentiment_icon}")
        print(f"  Confidence: {score:.2%}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    run_sentiment_demo()
