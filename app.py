import streamlit as st
import pandas as pd
import joblib

# Load all trained models
data = joblib.load("all_regression_models.pkl")
models = data["models"]
features = data["features"]

st.title("📊 Regression Model Prediction App")

# Algorithm selector
algorithm = st.selectbox("Choose a Regression Algorithm", list(models.keys()))

st.write("### Enter Feature Values:")
input_data = {}

# Dynamically create input fields for each feature
for feature in features:
    input_data[feature] = st.number_input(f"{feature}", value=0.0)

if st.button("Predict"):
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Get selected model
    model = models[algorithm]
    
    # Prediction
    prediction = model.predict(input_df)[0]
    
    st.success(f"🔮 Predicted Value using **{algorithm}**: {prediction:.2f}")
