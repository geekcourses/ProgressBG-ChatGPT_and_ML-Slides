"""
Comparing BERT-based Sentiment Analysis with IMDB Dataset
This script demonstrates how to use a pre-trained transformer model 
to evaluate reviews from the IMDB dataset.
"""

import os
import pandas as pd
import kagglehub
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ============================================================================
# DATA LOADING
# ============================================================================
def load_imdb_dataset(sample_size=100):
    """Download and load a sample of the IMDB 50K movie reviews dataset"""
    print(f"\n[1/4] Loading IMDB dataset (sampling {sample_size} reviews for speed)...")
    
    # Download latest version from Kaggle
    path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    df = pd.read_csv(f"{path}/IMDB Dataset.csv")
    
    # Take a small sample for demonstration (BERT inference can be slow on CPU)
    sample_df = df.sample(sample_size, random_state=42).reset_index(drop=True)
    return sample_df

# ============================================================================
# EVALUATION & VISUALIZATION
# ============================================================================
def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix using seaborn"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )
    plt.title("Confusion Matrix (BERT Model)", fontsize=16, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    # 1. Load Data
    df = load_imdb_dataset(sample_size=50) # Small sample for quick demo
    
    # 2. Load Model
    print("\n[2/4] Initializing Hugging Face pipeline...")
    # This automatically handles tokenization and model loading
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # 3. Running Inference
    print(f"\n[3/4] Running inference on {len(df)} reviews...")
    start_time = time.time()
    
    # Get reviews as a list
    texts = df['review'].tolist()
    
    # BERT has a maximum token limit (usually 512). We should truncate long reviews.
    # The pipeline can handle truncation if configured, or we can slice the text.
    results = classifier(texts, truncation=True)
    
    end_time = time.time()
    print(f"   Inference completed in {end_time - start_time:.2f} seconds.")

    # 4. Processing Results
    print("\n[4/4] Evaluating results...")
    
    # Map model labels (POSITIVE/NEGATIVE) to dataset labels (positive/negative)
    predictions = [res['label'].lower() for res in results]
    actuals = df['sentiment'].tolist()
    
    # Calculate Metrics
    accuracy = accuracy_score(actuals, predictions)
    print(f"\n   Accuracy: {accuracy:.2%}")
    print("\n   Classification Report:")
    print(classification_report(actuals, predictions))
    
    # Visualize
    # plot_confusion_matrix(actuals, predictions) # Uncomment if running in an interactive env

    # Show a few examples
    print("\n" + "=" * 60)
    print("Sample Individual Predictions:")
    print("=" * 60)
    for i in range(5):
        print(f"\nReview: {texts[i][:100]}...")
        print(f"Actual: {actuals[i]} | Predicted: {predictions[i]} (Conf: {results[i]['score']:.2%})")

    # 5. Save the Model
    print("\n[5/5] Saving model and tokenizer for later use...")
    save_dir = "models/distilbert_sentiment"
    os.makedirs(save_dir, exist_ok=True)
    classifier.model.save_pretrained(save_dir)
    classifier.tokenizer.save_pretrained(save_dir)
    print(f"   Model and tokenizer saved to: {save_dir}/")

if __name__ == "__main__":
    main()
