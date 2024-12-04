import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the trained model
model = load_model('heart_failure_cnn_model.h5')  # Simpan model ke file terlebih dahulu setelah training
scaler = StandardScaler()

# Define the Streamlit app
st.title("Heart Failure Prediction")

# Input form
st.subheader("Masukkan Data Pasien:")
age = st.number_input("Age", min_value=0, step=1)
anaemia = st.selectbox("Anaemia (1: Yes, 0: No)", [0, 1])
creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase (mcg/L)", min_value=0)
diabetes = st.selectbox("Diabetes (1: Yes, 0: No)", [0, 1])
ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=0, max_value=100, step=1)
high_blood_pressure = st.selectbox("High Blood Pressure (1: Yes, 0: No)", [0, 1])
platelets = st.number_input("Platelets (kiloplatelets/mL)", min_value=0.0, step=0.1)
serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.0, step=0.1)
serum_sodium = st.number_input("Serum Sodium (mEq/L)", min_value=0, step=1)
sex = st.selectbox("Sex (1: Male, 0: Female)", [0, 1])
smoking = st.selectbox("Smoking (1: Yes, 0: No)", [0, 1])
time = st.number_input("Follow-up Period (days)", min_value=0, step=1)

# Predict button
if st.button("Predict"):
    # Preprocess the input
    input_data = np.array([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                            high_blood_pressure, platelets, serum_creatinine, serum_sodium,
                            sex, smoking, time]])
    
    # Standardize the data
    input_data_scaled = scaler.fit_transform(input_data)
    input_data_reshaped = np.expand_dims(input_data_scaled, axis=-1)
    
    # Make prediction
    prediction_probs = model.predict(input_data_reshaped)
    prediction = np.argmax(prediction_probs)
    confidence = prediction_probs[0][prediction] * 100

    # Display result
    result = "DEATH_EVENT: Yes" if prediction == 1 else "DEATH_EVENT: No"
    st.subheader("Hasil Prediksi:")
    st.write(f"{result} dengan tingkat keyakinan {confidence:.2f}%")
