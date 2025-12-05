from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Define the corpus
documents = [
    "I love machine learning",
    "I love coding",
    "Machine learning is amazing",
    "Deep learning is a subset of machine learning",
]

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit the vectorizer to the documents and transform the documents into the TF-IDF matrix
# The result is a sparse matrix (efficient for large datasets)
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Get the feature names (vocabulary)
vocabulary = tfidf_vectorizer.get_feature_names_out()

# Convert the sparse matrix to a dense NumPy array and round the scores for readability
tfidf_array = tfidf_matrix.toarray().round(4)

# Define descriptive indices for the documents (rows)
document_indices = [f"Doc {i + 1}" for i in range(len(documents))]

df_tfidf = pd.DataFrame(tfidf_array, columns=vocabulary, index=document_indices)

print("## 📊 TF-IDF Matrix as DataFrame")
print("---------------------------------")
print(df_tfidf)

# Optional: Print the original documents for easy reference
print("\n## 📄 Original Documents")
print("---------------------------------")
for i, doc in enumerate(documents):
    print(f"Doc {i + 1}: {doc}")
