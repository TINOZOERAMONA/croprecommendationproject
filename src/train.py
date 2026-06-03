import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from src.preprocessing import CropPreprocessor
from src.utils import set_seed
from src.config import (
    DATA_PATH,
    MODEL_OUTPUT_PATH,
    SCALER_OUTPUT_PATH,
    LABEL_ENCODER_OUTPUT_PATH,
    REPORT_OUTPUT_PATH,
    RANDOM_STATE
)

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        ),
        "F1_Score": f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0
        )
    }

def train_models():

    set_seed(RANDOM_STATE)

    preprocessor = CropPreprocessor()

    data = preprocessor.preprocess(DATA_PATH)

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    results = []

    # Random Forest
    rf_model = RandomForestClassifier(
        random_state=RANDOM_STATE
    )
    rf_model.fit(X_train, y_train)

    rf_metrics = evaluate_model(
        rf_model,
        X_test,
        y_test
    )

    rf_metrics["Model"] = "RandomForest"

    results.append(rf_metrics)

    # KNN
    knn_model = KNeighborsClassifier(n_neighbors=5)

    knn_model.fit(X_train, y_train)

    knn_metrics = evaluate_model(
        knn_model,
        X_test,
        y_test
    )

    knn_metrics["Model"] = "KNN"

    results.append(knn_metrics)

    # XGBoost
    xgb_model = XGBClassifier(
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )

    xgb_model.fit(X_train, y_train)

    xgb_metrics = evaluate_model(
        xgb_model,
        X_test,
        y_test
    )

    xgb_metrics["Model"] = "XGBoost"

    results.append(xgb_metrics)

    models_dict = {
    "RandomForest": rf_model,
    "KNN": knn_model,
    "XGBoost": xgb_model
    }

    # Save report
    df = pd.DataFrame(results)

    best_model_name = df.loc[
    df["F1_Score"].idxmax()
    ]["Model"]

    best_model_object = models_dict[
        best_model_name
    ]

    print(
        f"The winning model is "
        f"{best_model_name}! Saving..."
    )
    joblib.dump(
    best_model_object,
    MODEL_OUTPUT_PATH
    )

    joblib.dump(
        preprocessor.scaler,
        SCALER_OUTPUT_PATH
    )

    joblib.dump(
        preprocessor.label_encoder,
        LABEL_ENCODER_OUTPUT_PATH
    )
    
    df = df[[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score"
    ]]

    df.to_csv(REPORT_OUTPUT_PATH, index=False)

    print("Training completed successfully.")
    print(df)


if __name__ == "__main__":
    train_models()