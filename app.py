import streamlit as st
import numpy as np
import joblib

# ==========================
# Load saved model objects
# ==========================

model = joblib.load("model/titanic_survival_model.pkl")
scaler = joblib.load("model/scaler.pkl")
sex_encoder = joblib.load("model/sex_encoder.pkl")
embarked_encoder = joblib.load("model/embarked_encoder.pkl")


# ==========================
# App UI
# ==========================

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

st.title("🚢 Titanic Survival Prediction System")
st.write("Enter passenger details to predict survival outcome.")

# ==========================
# User Inputs
# ==========================

pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=25.0)
fare = st.number_input("Fare", min_value=0.0, value=30.0)
embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])

# ==========================
# Predict Button
# ==========================

if st.button("Predict Survival"):

    # Encode categorical inputs
    sex_encoded = sex_encoder.transform([sex])[0]
    embarked_encoded = embarked_encoder.transform([embarked])[0]

    # Create input array
    input_data = np.array([[pclass, sex_encoded, age, fare, embarked_encoded]])

    # Scale features
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    # Display result
    if prediction == 1:
        st.success("✅ Prediction: Survived")
    else:
        st.error("❌ Prediction: Did Not Survive")

