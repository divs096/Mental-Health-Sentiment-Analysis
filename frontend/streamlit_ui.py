import streamlit as st
import requests

# Page setup
st.set_page_config(page_title="Mental Health Sentiment Analyzer", layout="centered")
st.title("🧠 Mental Health Sentiment Analyzer")
st.markdown("Enter a message below to check its mental health sentiment.")

# Input box
user_input = st.text_area("💬 Your Message:", height=150)

# Backend API endpoint
API_URL = "http://localhost:5000/predict"

# On button click
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a message to analyze.")
    else:
        try:
            # Send to backend
            response = requests.post(API_URL, json={"text": user_input})
            if response.status_code == 200:
                result = response.json()
                label = result['prediction']
                st.success(f"🔍 Prediction: **{label}**")
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"❌ Could not connect to the API. Make sure the backend is running.\n\n{e}")
