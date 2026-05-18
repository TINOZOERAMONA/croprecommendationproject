from preprocessing import CropPreprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
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
rf_accuracy = accuracy
rf_precision = precision_score(y_test, predictions, average="weighted", zero_division=0)
rf_recall = recall_score(y_test, predictions, average="weighted", zero_division=0)
rf_f1 = f1_score(y_test, predictions, average="weighted", zero_division=0)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_precision = precision_score(y_test, knn_predictions, average="weighted", zero_division=0)
knn_recall = recall_score(y_test, knn_predictions, average="weighted", zero_division=0)
knn_f1 = f1_score(y_test, knn_predictions, average="weighted", zero_division=0)

xgb_model = XGBClassifier(eval_metric="logloss")
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
xgb_precision = precision_score(y_test, xgb_predictions, average="weighted", zero_division=0)
xgb_recall = recall_score(y_test, xgb_predictions, average="weighted", zero_division=0)
xgb_f1 = f1_score(y_test, xgb_predictions, average="weighted", zero_division=0)

print(f"Accuracy: {accuracy}")

joblib.dump(model, "models/random_forest.pkl")
joblib.dump(preprocessor.scaler, "models/scaler.pkl")
joblib.dump(preprocessor.label_encoder, "models/label_encoder.pkl")


import pandas as pd

results = [
    {
        "Model": "KNN",
        "Accuracy": knn_accuracy,
        "Precision": knn_precision,
        "Recall": knn_recall,
        "F1_Score": knn_f1
    },
    {
        "Model": "RandomForest",
        "Accuracy": rf_accuracy,
        "Precision": rf_precision,
        "Recall": rf_recall,
        "F1_Score": rf_f1
    },
    {
        "Model": "XGBoost",
        "Accuracy": xgb_accuracy,
        "Precision": xgb_precision,
        "Recall": xgb_recall,
        "F1_Score": xgb_f1
    }
]

df = pd.DataFrame(results)

df.to_csv("reports/model_comparison.csv", index=False)