from preprocessing import CropPreprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

preprocessor = CropPreprocessor()

data = preprocessor.preprocess("data/raw/Crop_recommendation.csv")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

model = RandomForestClassifier()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy}")

joblib.dump(model, "models/random_forest.pkl")
joblib.dump(preprocessor.scaler, "models/scaler.pkl")
joblib.dump(preprocessor.label_encoder, "models/label_encoder.pkl")