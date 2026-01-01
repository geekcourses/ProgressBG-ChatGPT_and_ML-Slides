import os
import string
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data (run once)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

def preprocess_text(text):
    """
    Complete text preprocessing pipeline (must match training preprocessing)
    """
    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 3. Tokenize into words
    tokens = word_tokenize(text)

    # 4. Remove stopwords (keeping negations)
    stop_words = set(stopwords.words("english")) - {
        "not", "no", "nor", "neither", "never",
    }
    tokens = [word for word in tokens if word not in stop_words]

    # 5. Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # 6. Join tokens back into string
    return " ".join(tokens)

def load_resources(folder="models"):
    """Load the trained vectorizer and classifier"""
    v_path = os.path.join(folder, "vectorizer.joblib")
    m_path = os.path.join(folder, "classifier.joblib")

    if not os.path.exists(v_path) or not os.path.exists(m_path):
        raise FileNotFoundError(
            f"Model files not found in '{folder}/'. "
            "Please run the training script (SA_on_IMDB_50K_dataset.py) first."
        )

    vectorizer = joblib.load(v_path)
    model = joblib.load(m_path)
    return vectorizer, model

def predict(text, vectorizer, model):
    """Predict sentiment for a single piece of text"""
    # 1. Preprocess using the shared function logic
    cleaned = preprocess_text(text)
    
    # 2. Vectorize
    vectorized = vectorizer.transform([cleaned])
    
    # 3. Predict
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
    confidence = probabilities[int(prediction)] * 100
    
    return sentiment, confidence

def main():
    print("=" * 60)
    print("SENTIMENT ANALYSIS - MODEL INFERENCE DEMO")
    print("=" * 60)

    try:
        # Load the model and vectorizer
        print("Loading model and vectorizer...")
        vectorizer, model = load_resources()
        print("Resources loaded successfully!\n")

        # Example reviews to test
        test_reviews = [
            "The movie was an absolute masterpiece. I loved the character development!",
            "I found it quite boring and the plot was predictable.",
            "It was okay, not great but worth watching once.",
        ]

        for review in test_reviews:
            sentiment, confidence = predict(review, vectorizer, model)
            print(f"Review: {review}")
            print(f"Result: {sentiment} ({confidence:.2f}% confidence)\n")

        # Interactive loop
        print("-" * 60)
        print("Interactive Mode: Enter your own review (or 'q' to quit)")
        while True:
            user_input = input("\nReview > ").strip()
            if user_input.lower() in ['q', 'quit', 'exit']:
                break
            if not user_input:
                continue
                
            sentiment, confidence = predict(user_input, vectorizer, model)
            print(f"Prediction: {sentiment} ({confidence:.2f}% confidence)")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
