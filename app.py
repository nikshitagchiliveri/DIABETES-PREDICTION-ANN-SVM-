import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model, scaler, and RFE selector
ensemble_model = joblib.load('ensemble_model.pkl')
scaler = joblib.load('scaler.pkl')
rfe_selector = joblib.load('rfe_selector.pkl')

# Define the feature names (including engineered features, for the model)
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 
                 'Glucose_BMI', 'Age_Glucose', 'Insulin_Glucose']

# Streamlit app
st.title("Diabetes Prediction App")
st.write("Enter the following details to predict if a person has diabetes.")

# Create input fields for the raw features (user inputs these)
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20)
glucose = st.number_input("Glucose", min_value=0.0, max_value=200.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=150.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0)
insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0)
age = st.number_input("Age", min_value=0, max_value=120)

# Compute the engineered features based on user input
glucose_bmi = glucose * bmi
age_glucose = age * glucose
insulin_glucose = insulin * glucose

# Create a DataFrame with the input data (including engineered features)
input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, 
                            insulin, bmi, diabetes_pedigree, age, 
                            glucose_bmi, age_glucose, insulin_glucose]], 
                          columns=feature_names)

# Scale the input data
input_scaled = scaler.transform(input_data)

# Apply RFE feature selection
input_selected = rfe_selector.transform(input_scaled)

# Make prediction
if st.button("Predict"):
    prediction = ensemble_model.predict(input_selected)[0]
    result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    st.write(f"Prediction: **{result}**")
