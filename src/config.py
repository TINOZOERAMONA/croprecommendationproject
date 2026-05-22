"""
Central configuration for the Crop Recommendation project
"""

DATA_PATH = "data/raw/Crop_recommendation.csv"

CLEAN_DATA_PATH = "data/processed/cleaned_crop_data.csv"

MODEL_OUTPUT_PATH = "models/random_forest.pkl"
SCALER_OUTPUT_PATH = "models/scaler.pkl"
LABEL_ENCODER_OUTPUT_PATH = "models/label_encoder.pkl"

REPORT_OUTPUT_PATH = "reports/model_comparison.csv"

TEST_SIZE = 0.2
RANDOM_STATE = 42