import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, scaler, and feature names
model = joblib.load("xgb.pkl")
scaler = joblib.load("scaler (3).pkl")
feature_names = joblib.load("feature_names.pkl")

# Page configuration and styling
st.set_page_config(page_title="Solar Power Prediction", page_icon="🔆", 
                   layout="wide")
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(to right, #2b5876, #4e4376);
            color: white;
        }
        .stSidebar {
            background-color: #333;
        }
        .stButton>button {
            background-color: #ff7f50;
            color: white;
            border-radius: 10px;
            font-size: 18px;
        }
    </style>
    """, unsafe_allow_html=True
)

st.title("🔆 Solar Power Generation Prediction")
st.markdown("Provide feature values below to predict power generated from the solar panel.")

st.sidebar.header("Input Solar Features")

# Create input form
def get_user_input():
    input_df = pd.DataFrame(columns=feature_names)
    input_df.loc[0] = 0  # Initialize with zeros

    for feature in feature_names:
        if "time" in feature or "hour" in feature:
            input_df[feature] = st.sidebar.slider(f"{feature}", 0, 24, 12)
        elif "temperature" in feature or "humidity" in feature:
            input_df[feature] = st.sidebar.number_input(f"{feature}", min_value=0.0, max_value=100.0, value=25.0)
        else:
            input_df[feature] = st.sidebar.number_input(f"{feature}", value=1.0)

    return input_df

input_data = get_user_input()

# Scale the data
scaled_input = scaler.transform(input_data)

# Prediction section
st.subheader("Prediction Result")
if st.button("Predict Power"):
    prediction = model.predict(scaled_input)[0]
    st.success(f"Predicted Power Generated: **{np.expm1(prediction):.2f} units**")
