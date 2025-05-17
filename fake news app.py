import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# --- Model Training and Saving ---

# Load dataset (make sure file exists in your directory)
try:
    df = pd.read_csv("fakenews_dataset_500.csv")
except FileNotFoundError:
    st.error("Error: 'fakenews_dataset_500.csv' not found. Please upload the dataset.")
    st.stop() # Stop execution if dataset is not found

# Data preparation
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Reset index to align with sparse matrix
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Save the vectorizer and model
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(model, 'logistic_regression_model.pkl')

# --- Streamlit App ---

st.title("Fake News Classifier")

st.write("Enter a news article to classify:")

# Get user input
news_article = st.text_area("News Article", "")

if st.button("Classify"):
    if news_article:
        # Load the saved vectorizer and model within the button click to ensure they are available
        try:
            loaded_vectorizer = joblib.load('tfidf_vectorizer.pkl')
            loaded_model = joblib.load('logistic_regression_model.pkl')
        except FileNotFoundError:
            st.error("Error: Model or vectorizer files not found. Please run the training part of the code first.")
            st.stop()

        # Preprocess the input
        input_vec = loaded_vectorizer.transform([news_article])

        # Make a prediction
        prediction = loaded_model.predict(input_vec)

        # Display the result
        if prediction[0] == 0:
            st.write("Prediction: **Fake News**")
        else:
            st.write("Prediction: **True News**")
    else:
        st.write("Please enter a news article to classify.")

# --- Optional: Display Training Results (outside of the main app flow) ---
# You can uncomment this section if you want to see the training results
# when you run this script directly (not through streamlit run)

# if __name__ == '__main__':
#     # Accuracy
#     y_pred = model.predict(X_test_vec)
#     accuracy = accuracy_score(y_test, y_pred)
#     print("=== Model Accuracy ===")
#     print(f"Accuracy: {accuracy * 100:.2f}%")

#     # Confusion matrix
#     print("\n=== Confusion Matrix ===")
#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(6, 4))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['Fake', 'True'],
#                 yticklabels=['Fake', 'True'])
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.title("Confusion Matrix")
#     plt.tight_layout()
#     plt.show()

#     # Function to show top words
#     def show_top_words(vectorizer, X_vec, y_vec, label, top_n=10):
#         class_indices = [i for i, y in enumerate(y_vec) if y == label]
#         class_vec = X_vec[class_indices]
#         mean_tfidf = class_vec.mean(axis=0).A1
#         sorted_indices = mean_tfidf.argsort()[-top_n:]
#         feature_names = vectorizer.get_feature_names_out()
#         top_words = [feature_names[i] for i in sorted_indices]
#         return top_words

#     # Show top fake and true news words
#     top_fake_words = show_top_words(vectorizer, X_train_vec, y_train, label=0)
#     top_true_words = show_top_words(vectorizer, X_train_vec, y_train, label=1)

#     print("\n=== Top Words for Fake News ===")
#     print(top_fake_words)

#     print("\n=== Top Words for True News ===")
#     print(top_true_words)
