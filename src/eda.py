# Exploratory Data Analysis Module

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("reports", exist_ok=True)

print("EDA file is running...")


def perform_eda(file_path):
    df = pd.read_csv(file_path)

    print("\nDataset Shape:")
    print(df.shape)

    print("\nSummary Statistics:")
    print(df.describe())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicates:")
    print(df.duplicated().sum())

    # Crop distribution
    plt.figure(figsize=(10, 6))
    df['label'].value_counts().plot(kind='bar')
    plt.title("Crop Distribution")
    plt.savefig("reports/crop_distribution.png")

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df.drop('label', axis=1).corr(),
        annot=True
    )
    plt.title("Feature Correlation")
    plt.savefig("reports/correlation_heatmap.png")

    print("\nEDA completed successfully.")
    print("Figures saved in the reports folder.")



if __name__ == "__main__":
    perform_eda("data/processed/cleaned_crop_data.csv")