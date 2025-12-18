import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Load serialized pipeline and feature contract
# This runs ONCE when the app starts
@st.cache_resource
def load_model_bundle():
    bundle = joblib.load("car_price_xgb_pipeline.pkl")
    return bundle["pipeline"], bundle["features"]

pipeline, EXPECTED_FEATURES = load_model_bundle()


# Input validation function; Ensures incoming data matches training features
def validate_input(df):
    missing = set(EXPECTED_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")



# App UI
st.title("🚗 Car Price Prediction App")
st.write("Predict car prices using a trained XGBoost model")



# User input widgets

model = st.text_input("Car Model")

engine_size = st.number_input(
    "Engine Size (L)", min_value=0.5, max_value=8.0, step=0.1
)

age = st.number_input(
    "Car Age (years)", min_value=0, step=1
)

mileage = st.number_input(
    "Mileage (km)", min_value=0, step=1000
)

brand_popularity = st.slider(
    "Brand Popularity", 0.0, 1.0
)

manufacturer_ford = st.checkbox("Ford")
manufacturer_porsche = st.checkbox("Porsche")
manufacturer_toyota = st.checkbox("Toyota")
manufacturer_vw = st.checkbox("Volkswagen")

fuel_petrol = st.checkbox("Petrol")
fuel_hybrid = st.checkbox("Hybrid")


# =====================================
# Prediction logic
# Everything model-related happens HERE
# =====================================
if st.button("Predict Price"):

    # ---- basic sanity check ----
    if model.strip() == "":
        st.error("Please enter a car model.")
        st.stop()

    # ---- feature engineering (must match training) ----
    mileage_log = np.log(mileage + 1)

    # ---- build input dataframe (RAW → MODEL READY) ----
    input_df = pd.DataFrame({
        'Model': [model],
        'Engine size': [engine_size],
        'Age': [age],
        'Brand_popularity': [brand_popularity],
        'Manufacturer_Ford': [manufacturer_ford],
        'Manufacturer_Porsche': [manufacturer_porsche],
        'Manufacturer_Toyota': [manufacturer_toyota],
        'Manufacturer_VW': [manufacturer_vw],
        'Fuel type_Hybrid': [fuel_hybrid],
        'Fuel type_Petrol': [fuel_petrol],
        'Mileage_log': [mileage_log]
    })

    # ---- validate feature completeness ----
    try:
        validate_input(input_df)

        # ---- enforce column order ----
        input_df = input_df[EXPECTED_FEATURES]

        # ---- predict (pipeline handles encoding + exp) ----
        prediction = pipeline.predict(input_df)[0]

        st.success(f"💰 Estimated Price: ₦{prediction:,.2f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
