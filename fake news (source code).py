import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset (make sure file exists in your directory)
df = pd.read_csv("fakenews_dataset_500.csv")

# Plot class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=df, palette='Set2')
plt.title("Class Distribution (0 = Fake, 1 = True)")
plt.xlabel("News Type")
plt.ylabel("Count")
plt.xticks([0, 1], ['Fake News', 'True News'])
plt.tight_layout()
plt.show()

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

# Accuracy
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print("=== Model Accuracy ===")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix
print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Fake', 'True'],
            yticklabels=['Fake', 'True'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Function to show top words
def show_top_words(vectorizer, X_vec, y_vec, label, top_n=10):
    class_indices = [i for i, y in enumerate(y_vec) if y == label]
    class_vec = X_vec[class_indices]
    mean_tfidf = class_vec.mean(axis=0).A1
    top_indices = mean_tfidf.argsort()[-top_n:][::-1]
    features = vectorizer.get_feature_names_out()
    top_words = [features[i] for i in top_indices]

    label_name = "True News" if label == 1 else "Fake News"
    print(f"\nTop {top_n} words in {label_name}:")
    for word in top_words:
        print("-", word)

# Show TF-IDF top words
show_top_words(vectorizer, X_train_vec, y_train, label=0, top_n=10)
show_top_words(vectorizer, X_train_vec, y_train, label=1, top_n=10)

# User input prediction
print("\n=== Test Your Own News ===")
user_input = input("Enter a news statement or headline: ")
user_input_vec = vectorizer.transform([user_input])
user_prediction = model.predict(user_input_vec)[0]
user_label = "True News" if user_prediction == 1 else "Fake News"

print("\n=== Prediction Result ===")
print("Your Input:", user_input)
print("Prediction:", user_label)

# Visualization of input prediction
plt.figure(figsize=(5, 3))
sns.barplot(x=["Your News"], y=[1], color="green" if user_prediction == 1 else "red")
plt.title("Prediction: " + user_label)
plt.ylim(0, 1.5)
plt.ylabel("")
plt.xticks(fontsize=12)
plt.yticks([])
plt.tight_layout()
plt.show()
