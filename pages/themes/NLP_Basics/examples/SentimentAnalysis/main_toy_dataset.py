"""
Complete Sentiment Analysis System
Demonstrates the full NLP pipeline: preprocessing -> vectorization -> training -> evaluation
"""

import numpy as np
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Note: Install required packages first:
# pip install numpy pandas scikit-learn matplotlib seaborn nltk

import nltk

# Download required NLTK data (run once)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# ============================================================================
# STEP 1: CREATE SAMPLE DATASET
# ============================================================================
def create_dataset():
    """Create a sample movie review dataset"""
    sample_reviews = [
        # Positive reviews (label = 1)
        ("This movie was excellent! I loved every minute of it.", 1),
        ("Amazing film with great acting and storyline.", 1),
        ("Absolutely fantastic! Best movie I've seen this year.", 1),
        ("Wonderful performances and beautiful cinematography.", 1),
        ("I really enjoyed this movie. Highly recommend!", 1),
        ("Great entertainment! The plot was engaging.", 1),
        ("Superb direction and outstanding cast.", 1),
        ("This film exceeded all my expectations.", 1),
        ("Brilliant! A must-watch for everyone.", 1),
        ("Incredible movie with a powerful message.", 1),
        ("Loved the story and the characters!", 1),
        ("Perfect movie night! So entertaining!", 1),
        ("Exceptional work by the entire team.", 1),
        ("Highly satisfying and well-made film.", 1),
        ("One of the best movies ever made!", 1),
        # Negative reviews (label = 0)
        ("Terrible movie. Complete waste of time.", 0),
        ("I hated it. The plot made no sense.", 0),
        ("Boring and poorly acted. Don't watch it.", 0),
        ("Awful film. I couldn't finish it.", 0),
        ("The worst movie I've ever seen.", 0),
        ("Disappointing and dull throughout.", 0),
        ("Poor storyline and bad acting.", 0),
        ("I strongly dislike this movie.", 0),
        ("Horrible! Not worth your money.", 0),
        ("Extremely boring and predictable.", 0),
        ("Waste of money and time. Terrible!", 0),
        ("I want my money back. So bad!", 0),
        ("Poorly made with no redeeming qualities.", 0),
        ("Absolutely horrible film. Avoid it!", 0),
        ("Unwatchable. Complete disaster.", 0),
    ]

    df = pd.DataFrame(sample_reviews, columns=["review", "sentiment"])
    return df


# ============================================================================
# STEP 2: TEXT PREPROCESSING
# ============================================================================
def preprocess_text(text):
    """
    Complete text preprocessing pipeline

    Args:
        text (str): Raw text input

    Returns:
        str: Cleaned and processed text
    """
    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 3. Tokenize into words
    tokens = word_tokenize(text)

    # 4. Remove stopwords (but keep negations!)
    stop_words = set(stopwords.words("english")) - {
        "not",
        "no",
        "nor",
        "neither",
        "never",
    }
    tokens = [word for word in tokens if word not in stop_words]

    # 5. Lemmatization (reduce words to base form)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # 6. Join tokens back into string
    return " ".join(tokens)


# ============================================================================
# STEP 3: FEATURE IMPORTANCE ANALYSIS
# ============================================================================
def show_feature_importance(vectorizer, classifier, n=10):
    """Display most important features for each class"""
    feature_names = vectorizer.get_feature_names_out()
    coefs = classifier.coef_[0]

    # Most positive features
    top_positive_indices = np.argsort(coefs)[-n:]
    print(f"\n{'=' * 60}")
    print(f"Top {n} words indicating POSITIVE sentiment:")
    print("=" * 60)
    for idx in reversed(top_positive_indices):
        print(f"  {feature_names[idx]:20s} : {coefs[idx]:6.3f}")

    # Most negative features
    top_negative_indices = np.argsort(coefs)[:n]
    print(f"\n{'=' * 60}")
    print(f"Top {n} words indicating NEGATIVE sentiment:")
    print("=" * 60)
    for idx in top_negative_indices:
        print(f"  {feature_names[idx]:20s} : {coefs[idx]:6.3f}")


# ============================================================================
# STEP 4: PREDICTION FUNCTION
# ============================================================================
def predict_sentiment(text, vectorizer, model):
    """
    Predict sentiment of new text

    Args:
        text (str): Input text
        vectorizer: Trained TF-IDF vectorizer
        model: Trained classifier

    Returns:
        tuple: (sentiment, confidence)
    """
    # Preprocess the text
    cleaned = preprocess_text(text)

    # Vectorize
    vectorized = vectorizer.transform([cleaned])

    # Predict
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]

    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    confidence = probabilities[prediction] * 100

    return sentiment, confidence


# ============================================================================
# STEP 5: VISUALIZATION
# ============================================================================
def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    print("=" * 80)
    print("SENTIMENT ANALYSIS PIPELINE - COMPLETE DEMONSTRATION")
    print("=" * 80)

    # STEP 1: Load and explore data
    print("\n[1/7] Loading dataset...")
    df = create_dataset()
    print(f"   Dataset size: {len(df)} reviews")
    print(f"   Positive reviews: {sum(df['sentiment'] == 1)}")
    print(f"   Negative reviews: {sum(df['sentiment'] == 0)}")

    # Show example
    print("\n   Example review:")
    print(f"   Original: {df['review'].iloc[0]}")

    # STEP 2: Preprocess text
    print("\n[2/7] Preprocessing text...")
    df["cleaned_review"] = df["review"].apply(preprocess_text)
    print(f"   Cleaned:  {df['cleaned_review'].iloc[0]}")

    # STEP 3: Split data
    print("\n[3/7] Splitting data into train/test sets...")
    X = df["cleaned_review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training set: {len(X_train)} reviews")
    print(f"   Test set: {len(X_test)} reviews")

    # STEP 4: Vectorization (TF-IDF)
    print("\n[4/7] Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=100,  # Limit to top 100 features
        ngram_range=(1, 2),  # Use unigrams and bigrams
        min_df=1,  # Minimum document frequency
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"   Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    print(f"   Training matrix shape: {X_train_tfidf.shape}")
    print(f"   Test matrix shape: {X_test_tfidf.shape}")

    # STEP 5: Train model
    print("\n[5/7] Training Logistic Regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_tfidf, y_train)
    print("   Model training complete!")

    # STEP 6: Evaluate model
    print("\n[6/7] Evaluating model performance...")
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n   Accuracy: {accuracy:.2%}")

    print("\n   Classification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=["Negative", "Positive"], digits=3
        )
    )

    # Show confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    # Show feature importance
    show_feature_importance(vectorizer, model, n=10)

    # STEP 7: Test with new reviews
    print("\n" + "=" * 80)
    print("[7/7] Testing with new reviews...")
    print("=" * 80)

    new_reviews = [
        "This movie was absolutely brilliant! I loved every second!",
        "Waste of time. Terrible acting and boring plot.",
        "Not bad, but could have been better. Average movie.",
        "Fantastic cinematography and great performances!",
        "I regret watching this. Complete disaster.",
    ]

    for i, review in enumerate(new_reviews, 1):
        sentiment, confidence = predict_sentiment(review, vectorizer, model)
        print(f"\nReview {i}:")
        print(f'  Text: "{review}"')
        print(f"  Prediction: {sentiment}")
        print(f"  Confidence: {confidence:.1f}%")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)

    return vectorizer, model


# ============================================================================
# RUN THE PIPELINE
# ============================================================================
if __name__ == "__main__":
    vectorizer, model = main()

    # Interactive testing
    print("\n" + "=" * 80)
    print("Interactive Mode - Test your own reviews!")
    print("=" * 80)
    print("Enter 'quit' to exit\n")

    while True:
        user_input = input("Enter a movie review: ").strip()

        if user_input.lower() in ["quit", "exit", "q"]:
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        sentiment, confidence = predict_sentiment(user_input, vectorizer, model)
        print(f"→ Sentiment: {sentiment} (Confidence: {confidence:.1f}%)\n")
