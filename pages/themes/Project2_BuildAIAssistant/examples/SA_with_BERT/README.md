# Sentiment Analysis with BERT (Hugging Face)

This directory contains examples of performing Sentiment Analysis using State-of-the-Art Transformer models (BERT/DistilBERT) via the Hugging Face `transformers` library.

## Comparison: Traditional ML vs. BERT

| Feature | Traditional (NLTK + Sklearn) | Transformers (BERT) |
|---------|-----------------------------|---------------------|
| **Approach** | Feature Engineering (TF-IDF, Bag of Words) | Pre-trained Contextual Embeddings |
| **Context** | Limited (mostly word frequency) | High (understands sequence and context) |
| **Preprocessing** | Heavy (Stopwords, Lemmatization, etc.) | Minimal (built-in Tokenization) |
| **Performance** | Good for simple tasks, very fast | Excellent accuracy, computationally heavy |
| **Hardware** | Works great on any CPU | Recommended GPU for large datasets |

## Files in this Demo

1.  **`SA_with_BERT_demo.py`**: A quick-start script using the `pipeline` API for instant sentiment analysis on sample text.
2.  **`SA_BERT_on_IMDB.py`**: Loads a sample from the IMDB dataset and evaluates the model's accuracy (parity with the sklearn example).
3.  **`use_BERT_model.py`**: Demonstrates how to save the model locally and load it for offline use or deployment.

## Getting Started

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Quick Demo**:
    ```bash
    python SA_with_BERT_demo.py
    ```

3.  **Run Dataset Evaluation**:
    ```bash
    python SA_BERT_on_IMDB.py
    ```
