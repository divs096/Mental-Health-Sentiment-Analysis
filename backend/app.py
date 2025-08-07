from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# Build correct path to the saved model
base_dir = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(base_dir, '..', 'model', 'model.pkl')

# Load the model pipeline
try:
    model = joblib.load(model_path)
    print("✅ Model loaded successfully")
    print("📚 Classes:", model.named_steps['clf'].classes_)  # Optional for debugging
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return "🧠 Mental Health Sentiment API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    input_data = request.get_json()
    if not input_data or 'text' not in input_data:
        return jsonify({'error': 'Missing text field'}), 400

    input_text = input_data['text']
    prediction = model.predict([input_text])[0]

    return jsonify({
        'input': input_text,
        'prediction': prediction
    })

if __name__ == '__main__':
    app.run(debug=True)
