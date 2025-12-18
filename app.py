import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load pipeline
# -------------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load("car_price_xgb_pipeline.pkl")

pipeline = load_pipeline()

# -------------------------------
# App UI
# -------------------------------
st.title("🚗 Car Price Prediction App")
st.write("Predict car prices using a trained XGBoost model")

# -------------------------------
# User Inputs
# -------------------------------
model = st.text_input("Car Model (e.g. Corolla, Camry, Golf)")

engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=8.0, step=0.1)
year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, step=1)
mileage = st.number_input("Mileage (km)", min_value=0, step=1000)

age = st.number_input("Car Age (years)", min_value=0, step=1)
mileage_per_year = st.number_input("Mileage per Year", min_value=0.0)

brand_popularity = st.slider("Brand Popularity Score", 0.0, 1.0)

fuel_petrol = st.checkbox("Petrol")
fuel_hybrid = st.checkbox("Hybrid")

manufacturer_ford = st.checkbox("Ford")
manufacturer_toyota = st.checkbox("Toyota")
manufacturer_vw = st.checkbox("Volkswagen")
manufacturer_porsche = st.checkbox("Porsche")

# -------------------------------
# Predict button
# -------------------------------
if st.button("Predict Price"):
    input_df = pd.DataFrame({
        "Model": [model],
        "Engine size": [engine_size],
        "Year of manufacture": [year],
        "Mileage": [mileage],
        "Age": [age],
        "Mileage_per_year": [mileage_per_year],
        "Brand_popularity": [brand_popularity],
        "Manufacturer_Ford": [manufacturer_ford],
        "Manufacturer_Toyota": [manufacturer_toyota],
        "Manufacturer_VW": [manufacturer_vw],
        "Manufacturer_Porsche": [manufacturer_porsche],
        "Fuel type_Petrol": [fuel_petrol],
        "Fuel type_Hybrid": [fuel_hybrid],
    })

    try:
        prediction = pipeline.predict(input_df)[0]
        st.success(f"💰 Estimated Price: ₦{prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
