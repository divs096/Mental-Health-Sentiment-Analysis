import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv('data/mental_health.csv')

X = df['text']
y = df['label']  # Make sure these are multiclass labels

# Define the pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs'))
])

# Train the model
model.fit(X, y)

# Save the model
joblib.dump(model, 'model/model.pkl')

print("✅ Multiclass model trained and saved as model/model.pkl")
