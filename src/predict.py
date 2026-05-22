import joblib
import pandas as pd
import numpy as np

from src.config import (
    MODEL_OUTPUT_PATH,
    SCALER_OUTPUT_PATH,
    LABEL_ENCODER_OUTPUT_PATH
)

# Load saved artifacts
model = joblib.load(MODEL_OUTPUT_PATH)

scaler = joblib.load(SCALER_OUTPUT_PATH)

label_encoder = joblib.load(
    LABEL_ENCODER_OUTPUT_PATH
)

# Example input:
# N, P, K, temperature, humidity, ph, rainfall

sample_data = pd.DataFrame([{
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.87,
    "humidity": 82.00,
    "ph": 6.5,
    "rainfall": 202.93
}])

# Scale input
scaled_data = scaler.transform(sample_data)

# Predict
prediction = model.predict(scaled_data)

# Decode prediction
crop_name = label_encoder.inverse_transform(
    prediction
)

print(f"Recommended Crop: {crop_name[0]}")