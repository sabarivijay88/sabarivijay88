import streamlit as st
import pandas as pd
import joblib

model = joblib.load("tourism_project/deployment/best_model.pkl")

st.title("Tourism Product Prediction App")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Monthly Income", min_value=0, value=50000)
trips = st.number_input("Number of Trips", min_value=0, value=1)
passport = st.selectbox("Has Passport?", ["Yes", "No"])

input_data = pd.DataFrame({
    "Age": [age],
    "MonthlyIncome": [income],
    "NumberOfTrips": [trips],
    "Passport": [1 if passport == "Yes" else 0]
})

prediction = model.predict(input_data)[0]

if prediction == 1:
    st.success("✅ Customer is likely to take the product!")
else:
    st.warning("❌ Customer is unlikely to take the product.")
