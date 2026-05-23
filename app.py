import streamlit as st
import numpy as np
import pickle

# Page config
st.set_page_config(
    page_title="Multi Disease Prediction",
    page_icon="🩺",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.stApp {
    background: linear-gradient(to right, #74ebd5, #ACB6E5);
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #ffffff;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #f0f0f0;
    font-size: 18px;
    margin-bottom: 30px;
}

div.stButton > button {
    background-color: #ff4b4b;
    color: white;
    font-size: 18px;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    border: none;
}

div.stButton > button:hover {
    background-color: #ff1e1e;
    color: white;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    color: black;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# Load model
model, encoder = pickle.load(open("model.pkl", "rb"))

# Title
st.markdown('<div class="title">🩺 Multi Disease Prediction System</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Predict diseases using Machine Learning and medical parameters</div>',
    unsafe_allow_html=True
)

# Input section
st.subheader("📋 Enter Patient Details")

age = st.slider("Age", 1, 100, 25)

glucose = st.slider("Glucose Level", 50, 300, 100)

blood_pressure = st.slider("Blood Pressure", 40, 200, 80)

bmi = st.slider("BMI", 10.0, 50.0, 22.0)

cholesterol = st.slider("Cholesterol", 100, 400, 180)

# Prediction
if st.button("🔍 Predict Disease"):

    features = np.array([[
        age,
        glucose,
        blood_pressure,
        bmi,
        cholesterol
    ]])

    prediction = model.predict(features)

    disease = encoder.inverse_transform(prediction)

    st.markdown(
        f'<div class="result-box">✅ Predicted Disease: {disease[0]}</div>',
        unsafe_allow_html=True
    )

# Footer
st.markdown("---")

st.markdown(
    "<center>Developed using ❤️ with Streamlit & Machine Learning</center>",
    unsafe_allow_html=True
)