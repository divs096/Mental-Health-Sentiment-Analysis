import joblib
import os
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

nltk.download("stopwords")

# Load the trained model and vectorizer
model_path = os.path.join("backend", "model.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError("🔍 Model not found. Please train it using train_model.py")

model_data = joblib.load(model_path)
model = model_data["model"]
vectorizer = model_data["vectorizer"]

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def predict_sentiment(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return prediction

# 🧪 Testing this module standalone
if __name__ == "__main__":
    while True:
        user_input = input("\n🗣️ Enter a message to analyze (or type 'exit'): ")
        if user_input.lower() == "exit":
            print("👋 Exiting sentiment analyzer.")
            break
        result = predict_sentiment(user_input)
        print(f"💬 Predicted Sentiment: {result}")
