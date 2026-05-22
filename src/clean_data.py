from src.preprocessing import CropPreprocessor
from src.config import DATA_PATH

preprocessor = CropPreprocessor()

preprocessor.preprocess(DATA_PATH)

print("Preprocessing completed successfully.")