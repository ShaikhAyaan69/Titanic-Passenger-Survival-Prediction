# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 18:19:52 2024

@author: ASUS
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('logistic_regression_model.pkl')

# Set up the title and description
st.title("Titanic Passenger Survival Prediction :ship:")
st.write("Enter the passenger details below to predict survival probability.")

# Input fields for user data
Pclass = st.selectbox("Passenger Class", [1, 2, 3])
Sex = st.selectbox("Sex", ["Male", "Female"])
Age = st.number_input("Age", min_value=0, value=30)  # Default age can be set

Fare = st.slider("Fare in British Pounds (£)", min_value=100.0, max_value=1000.0, value=300.0, step=10.0)


Embarked = st.selectbox("Boarding Point", ["Cherbourg, France", "Queenstown, Ireland", "Southampton, England"])


# Convert categorical variables to numeric values
sex_numeric = 1 if Sex == 'Male' else 0


# Create dummy variables for Embarked
embarked_dummy = [0, 0]  # Default for Q and S
if Embarked == 'Q':
    embarked_dummy = [1, 0]  # Embarked_Q = 1, Embarked_S = 0
elif Embarked == 'S':
    embarked_dummy = [0, 1]  # Embarked_Q = 0, Embarked_S = 1

# Create a DataFrame for the input data
input_data = pd.DataFrame([[Pclass, sex_numeric, Age, Fare] + embarked_dummy],
                          columns=['Pclass', 'Sex', 'Age', 'Fare', 'Embarked_Q', 'Embarked_S'])


# Make predictions
try:
    prediction = model.predict(input_data)
    survival_probability = model.predict_proba(input_data)[0][1]  # Probability of survival

    # Display the prediction result
    if prediction == 1:
        survival_percentage = survival_probability * 100  # Convert to percentage
        st.markdown(f"<h4>✨ Great news! The passenger is predicted to <strong>SURVIVED</strong> with a probability of {survival_percentage:.0f}%.</h4>", unsafe_allow_html=True)
    else:
        not_survived_percentage = (1 - survival_probability) * 100  # Convert to percentage
        st.markdown(f"<h4>⚠️ Unfortunately, the passenger is predicted to <strong>NOT SURVIVED</strong> with a probability of {not_survived_percentage:.0f}%.</h4>", unsafe_allow_html=True)
except ValueError as e:
    st.error(f"Error in prediction: {str(e)}")
    







