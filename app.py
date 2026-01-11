import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ======================================================
# Page Config
# ======================================================

st.set_page_config(
    page_title="California Housing Predictor",
    page_icon="🏠",
    layout="wide"
)

# ======================================================
# Load Models
# ======================================================

model = joblib.load("linear_model.pkl")
pca = joblib.load("pca_model.pkl")
scaler = joblib.load("scaler.pkl")

# ======================================================
# Header
# ======================================================

st.title("🏠 California Housing Price Prediction")
st.markdown(
    "Predict **median house value** using **PCA + Linear Regression**"
)

# ======================================================
# Layout
# ======================================================

col1, col2 = st.columns(2)

with col1:
    MedInc = st.slider("Median Income", 0.5, 15.0, 5.0)
    HouseAge = st.slider("House Age", 1, 52, 20)
    AveRooms = st.slider("Average Rooms", 1.0, 10.0, 5.0)
    AveBedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.0)

with col2:
    Population = st.slider("Population", 100, 50000, 1000)
    AveOccup = st.slider("Average Occupancy", 1.0, 10.0, 3.0)
    Latitude = st.slider("Latitude", 32.0, 42.0, 36.0)
    Longitude = st.slider("Longitude", -124.0, -114.0, -119.0)

# ======================================================
# Prediction
# ======================================================

input_data = np.array([[
    MedInc, HouseAge, AveRooms, AveBedrms,
    Population, AveOccup, Latitude, Longitude
]])

scaled = scaler.transform(input_data)
pca_input = pca.transform(scaled)

if st.button("Predict House Value"):
    prediction = model.predict(pca_input)[0]
    st.success(f"Estimated Median House Value: ${prediction * 100000:,.2f}")

# ======================================================
# PCA Info
# ======================================================

st.subheader("📊 PCA Information")
st.write(f"Number of PCA Components Used: **{pca.n_components_}**")

# ======================================================
# Footer
# ======================================================

st.markdown("---")
st.markdown("📌 *PCA-based Regression Model deployed using Streamlit*")
