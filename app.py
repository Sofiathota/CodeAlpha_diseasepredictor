import streamlit as st
import numpy as np
import pickle

# Load model and encoder
model, encoder = pickle.load(open("model.pkl", "rb"))

# Title
st.title("🩺 Multi Disease Prediction System")

st.write("Enter patient details below:")

# Inputs
age = st.number_input("Age", min_value=1, max_value=100)

glucose = st.number_input("Glucose Level", min_value=50, max_value=300)

blood_pressure = st.number_input("Blood Pressure", min_value=40, max_value=200)

bmi = st.number_input("BMI", min_value=10.0, max_value=50.0)

cholesterol = st.number_input("Cholesterol", min_value=100, max_value=400)

# Prediction
if st.button("Predict Disease"):

    features = np.array([[
        age,
        glucose,
        blood_pressure,
        bmi,
        cholesterol
    ]])

    prediction = model.predict(features)

    disease = encoder.inverse_transform(prediction)

    st.success(f"Predicted Disease: {disease[0]}")