# 🧠 Mental Health Sentiment Analyzer

A simple logistic regression-powered app to detect sentiment in user text, potentially aiding in identifying signs of mental stress.

## 🚀 Setup Instructions

1. Clone the repo:
    ```
    git clone https://github.com/yourusername/mental-health-analyzer.git
    ```

2. Navigate to the project:
    ```
    cd mental-health-analyzer
    ```

3. (Optional but recommended) Create a virtual environment:
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

5. Add your `mental_health.csv` file inside the `data/` folder. The CSV should have two columns: `text` and `label` (0 or 1).

6. Train the model:
    ```
    python backend/train_model.py
    ```

7. Launch the web app:
    ```
    streamlit run frontend/app.py
    ```

---


