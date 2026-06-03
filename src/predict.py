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
    recommend_crops_bert
)

# Load saved artifacts
model = joblib.load(MODEL_OUTPUT_PATH)

scaler = joblib.load(SCALER_OUTPUT_PATH)

label_encoder = joblib.load(
    LABEL_ENCODER_OUTPUT_PATH
)

def run_prediction():
    print("The Crop Recommendation System (CLI Mode)")
    print("Type farming conditions in natural language\n")

    query = input("Farmer Input: ")

    recommendations, extracted, explanations = recommend_crops_bert(
    query,
    model,
    label_encoder,
    scaler
    )

    print("\n==============================")
    print("Top Recommended Crops")
    print("==============================\n")

    for i, rec in enumerate(recommendations, 1):
        print(
            f"{i}. {rec['crop']} "
            f"({rec['confidence']}%)"
        )

    print("\nReasoning:")

    for reason in explanations:
        print(f"- {reason}")

    print()

if __name__ == "__main__":
    run_prediction()