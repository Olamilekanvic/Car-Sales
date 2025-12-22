# Custom log1p transformer function
def log1p_transform(x):
    import numpy as np
    return np.log1p(x)

# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# App configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Car Price Prediction App",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Car Price Prediction")
st.write("Predict the estimated market price of a car using machine learning.")

# --------------------------------------------------
# Load trained pipeline and feature contract
# --------------------------------------------------
@st.cache_resource
def load_model():
    bundle = joblib.load("car_price_xgb_pipeline.pkl")
    return bundle["pipeline"], bundle["features"]

pipeline, EXPECTED_FEATURES = load_model()

# --------------------------------------------------
# Input validation helper
# --------------------------------------------------
def validate_input(df):
    missing = set(EXPECTED_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")

# --------------------------------------------------
# User inputs
# --------------------------------------------------
st.subheader("Enter car details")

model_name = st.text_input("Car Model", placeholder="e.g. Corolla")

engine_size = st.number_input(
    "Engine Size (Litres)",
    min_value=0.5,
    max_value=8.0,
    step=0.1
)

age = st.number_input(
    "Car Age (Years)",
    min_value=0,
    max_value=50,
    step=1
)

mileage = st.number_input(
    "Mileage (km)",
    min_value=0,
    step=1000
)

brand_popularity = st.slider(
    "Brand Popularity Score",
    min_value=0.0,
    max_value=1.0,
    step=0.01
)

st.markdown("### Manufacturer")
manufacturer_ford = st.checkbox("Ford")
manufacturer_porsche = st.checkbox("Porsche")
manufacturer_toyota = st.checkbox("Toyota")
manufacturer_vw = st.checkbox("Volkswagen")

st.markdown("### Fuel Type")
fuel_hybrid = st.checkbox("Hybrid")
fuel_petrol = st.checkbox("Petrol")

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict Price"):
    try:
        if model_name.strip() == "":
            st.error("Please enter a car model.")
            st.stop()

        # Build input DataFrame (RAW features only)
        input_df = pd.DataFrame({
            'Model': [model_name],
            'Engine size': [engine_size],
            'Age': [age],
            'Mileage': [mileage],
            'Brand_popularity': [brand_popularity],
            'Manufacturer_Ford': [manufacturer_ford],
            'Manufacturer_Porsche': [manufacturer_porsche],
            'Manufacturer_Toyota': [manufacturer_toyota],
            'Manufacturer_VW': [manufacturer_vw],
            'Fuel type_Hybrid': [fuel_hybrid],
            'Fuel type_Petrol': [fuel_petrol]
        })

        # Validate feature contract
        validate_input(input_df)

        # Predict (returns REAL price because of TransformedTargetRegressor)
        prediction = pipeline.predict(input_df).item()
        #prediction = pipeline.predict(input_df)[0]

        st.success(f"💰 Estimated Car Price: ${prediction:,.2f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
