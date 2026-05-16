"""
Data preprocessing module for Crop Recommendation System
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class CropPreprocessor:

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    
    #loading dataset
    def load_dataset(self, file_path):
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            raise Exception(f"Error loading dataset: {e}")

    #validating the dataset
    def validate_dataset(self, df):

        required_columns = [
            "N", "P", "K",
            "temperature", "humidity",
            "ph", "rainfall", "label"
        ]

        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            raise ValueError(f"Missing columns: {missing}")

        return True

    #cleaning data
    def clean_data(self, df):
        df = df.drop_duplicates()
        df = df.dropna()
        return df

    #saving cleaned dataser
    def save_cleaned_data(self, df, output_path):
        df.to_csv(output_path, index=False)
        print(f"Cleaned dataset saved to: {output_path}")

    #splitting features and target
    def split_features_target(self, df):
        X = df.drop(columns=["label"])
        y = df["label"]
        self.feature_names = X.columns
        return X, y

    #encoding target variable
    def encode_target(self, y):
        return self.label_encoder.fit_transform(y)

    #scaling features
    def scale_features(self, X):
        return self.scaler.fit_transform(X)

    #train/test split
    def split_dataset(self, X, y):
        return train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

    
    
    def preprocess(self, file_path):

        # Load
        df = self.load_dataset(file_path)

        # Validate
        self.validate_dataset(df)

        # Clean
        df = self.clean_data(df)

        # Save cleaned dataset
        self.save_cleaned_data(
            df,
            "data/processed/cleaned_crop_data.csv"
        )

        # Split
        X, y = self.split_features_target(df)

        # Encode
        y = self.encode_target(y)

        # Train/test split
        X_train, X_test, y_train, y_test = self.split_dataset(X, y)

        # Scale
        X = self.scale_features(X)

        

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names 
        }