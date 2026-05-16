from preprocessing import CropPreprocessor

preprocessor = CropPreprocessor()

preprocessor.preprocess(
    "data/raw/Crop_recommendation.csv"
)

print("Preprocessing completed successfully.")