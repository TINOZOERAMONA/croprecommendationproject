import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import numpy as np
from src.config import (
    MODEL_OUTPUT_PATH,
    SCALER_OUTPUT_PATH,
    LABEL_ENCODER_OUTPUT_PATH
)
from src.nlp_engine import (
    extract_features_bert,
    generate_feature_vector
)

# Load saved artifacts
model = joblib.load(MODEL_OUTPUT_PATH)

scaler = joblib.load(SCALER_OUTPUT_PATH)

label_encoder = joblib.load(
    LABEL_ENCODER_OUTPUT_PATH
)

def run_prediction():
    print("\n🌾 Crop Recommendation System (CLI Mode)")
    print("Type farming conditions in natural language\n")

    query = input("Farmer Input: ")

    # STEP 1: NLP → structured features
    extracted, scores = extract_features_bert(query)

    # STEP 2: feature vector
    X = generate_feature_vector(extracted)

    # STEP 3: scale (VERY IMPORTANT — must match training)
    X_scaled = scaler.transform(X)

    # STEP 4: predict
    pred = model.predict(X_scaled)[0]
    crop = label_encoder.inverse_transform([pred])[0]

    print("\n==============================")
    print("🌱 Recommended Crop:", crop)
    print("==============================\n")


if __name__ == "__main__":
    run_prediction()
