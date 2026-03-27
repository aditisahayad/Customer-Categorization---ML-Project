"""
save_models.py
==============
Script to train and save the ML pipeline models for the Customer Categorization system.

This script:
1. Reads the marketing_campaign.csv dataset
2. Performs feature engineering (matching the notebook pipeline)
3. Fits a preprocessing pipeline (StandardScaler + PowerTransformer)
4. Fits a PCA model for dimensionality reduction
5. Performs KMeans clustering to generate cluster labels
6. Trains a Logistic Regression classifier on 5 user-facing features
7. Saves preprocessor, PCA, and classifier as pickle files

Usage:
    python save_models.py
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")


def load_and_preprocess_data():
    """
    Load the marketing_campaign.csv dataset and perform feature engineering
    exactly as done in the notebooks.
    """
    # -------------------------------------------------------------------
    # Step 1: Load raw data
    # -------------------------------------------------------------------
    data_path = os.path.join("notebooks", "marketing_campaign.csv")
    df = pd.read_csv(data_path, sep="\t")
    print(f"[INFO] Loaded dataset with shape: {df.shape}")

    # -------------------------------------------------------------------
    # Step 2: Handle missing values — fill Income NaN with median
    # -------------------------------------------------------------------
    df["Income"] = df["Income"].fillna(df["Income"].median())

    # -------------------------------------------------------------------
    # Step 3: Drop constant / irrelevant columns
    # -------------------------------------------------------------------
    columns_to_drop = ["ID", "Z_CostContact", "Z_Revenue"]
    df = df.drop(columns=columns_to_drop)

    # -------------------------------------------------------------------
    # Step 4: Feature Engineering (matching the notebook)
    # -------------------------------------------------------------------
    # Age
    df["Age"] = 2022 - df["Year_Birth"]

    # Education encoding: Basic→0, 2n Cycle→1, Graduation→2, Master→3, PhD→4
    df["Education"] = df["Education"].replace(
        {"Basic": 0, "2n Cycle": 1, "Graduation": 2, "Master": 3, "PhD": 4}
    )

    # Marital Status encoding: partner→1, no partner→0
    df["Marital_Status"] = df["Marital_Status"].replace(
        {
            "Married": 1, "Together": 1,
            "Absurd": 0, "Widow": 0, "YOLO": 0,
            "Divorced": 0, "Single": 0, "Alone": 0,
        }
    )

    # Children
    df["Children"] = df["Kidhome"] + df["Teenhome"]

    # Family Size
    df["Family_Size"] = df["Marital_Status"] + df["Children"] + 1

    # Total Spending
    df["Total_Spending"] = (
        df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"]
        + df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
    )

    # Total Promo
    df["Total Promo"] = (
        df["AcceptedCmp1"] + df["AcceptedCmp2"] + df["AcceptedCmp3"]
        + df["AcceptedCmp4"] + df["AcceptedCmp5"]
    )

    # Days as Customer
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
    today = datetime(2022, 1, 1)  # Use fixed date for reproducibility
    df["Days_as_Customer"] = (today - df["Dt_Customer"]).dt.days

    # Offers Responded To
    df["Offers_Responded_To"] = (
        df["AcceptedCmp1"] + df["AcceptedCmp2"] + df["AcceptedCmp3"]
        + df["AcceptedCmp4"] + df["AcceptedCmp5"] + df["Response"]
    )

    # Parental Status
    df["Parental Status"] = np.where(df["Children"] > 0, 1, 0)

    # -------------------------------------------------------------------
    # Step 5: Drop columns used for feature creation & rename
    # -------------------------------------------------------------------
    df = df.drop(columns=["Year_Birth", "Kidhome", "Teenhome"])
    df = df.rename(
        columns={
            "Marital_Status": "Marital Status",
            "MntWines": "Wines", "MntFruits": "Fruits",
            "MntMeatProducts": "Meat", "MntFishProducts": "Fish",
            "MntSweetProducts": "Sweets", "MntGoldProds": "Gold",
            "NumWebPurchases": "Web", "NumCatalogPurchases": "Catalog",
            "NumStorePurchases": "Store", "NumDealsPurchases": "Discount Purchases",
        }
    )

    # Select the 21 features used for clustering
    df = df[[
        "Age", "Education", "Marital Status", "Parental Status", "Children",
        "Income", "Total_Spending", "Days_as_Customer", "Recency",
        "Wines", "Fruits", "Meat", "Fish", "Sweets", "Gold",
        "Web", "Catalog", "Store", "Discount Purchases",
        "Total Promo", "NumWebVisitsMonth",
    ]]

    print(f"[INFO] Feature-engineered dataset shape: {df.shape}")
    return df


def build_preprocessor(df):
    """
    Build and fit the preprocessing pipeline using ColumnTransformer with:
    - StandardScaler for normal numeric features
    - PowerTransformer for outlier-prone features
    """
    outlier_features = [
        "Wines", "Fruits", "Meat", "Fish", "Sweets", "Gold", "Age", "Total_Spending"
    ]
    all_numeric = [f for f in df.columns if df[f].dtype != "O"]
    numeric_features = [f for f in all_numeric if f not in outlier_features]

    numeric_pipeline = Pipeline(steps=[
        ("Imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("StandardScaler", StandardScaler()),
    ])

    outlier_pipeline = Pipeline(steps=[
        ("Imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("Transformer", PowerTransformer(standardize=True)),
    ])

    preprocessor = ColumnTransformer([
        ("numeric_pipeline", numeric_pipeline, numeric_features),
        ("outlier_pipeline", outlier_pipeline, outlier_features),
    ])

    # Fit and transform
    scaled_data = preprocessor.fit_transform(df)
    # ColumnTransformer reorders: numeric_features first, then outlier_features
    output_columns = numeric_features + outlier_features
    scaled_df = pd.DataFrame(scaled_data, columns=output_columns)

    print(f"[INFO] Preprocessor fitted. Scaled data shape: {scaled_df.shape}")
    return preprocessor, scaled_df


def fit_pca(scaled_df, n_components=3):
    """Fit PCA for dimensionality reduction."""
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_df)
    print(f"[INFO] PCA fitted. Explained variance ratio: {pca.explained_variance_ratio_}")
    return pca, pca_data


def perform_clustering(pca_data, n_clusters=3):
    """Perform KMeans clustering and return cluster labels."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(pca_data)
    print(f"[INFO] KMeans clustering done. Cluster distribution:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"       Cluster {u}: {c} samples")
    return cluster_labels


def train_classifier(df, cluster_labels):
    """
    Train a Logistic Regression classifier using the 5 user-facing features
    to predict the cluster label.
    """
    # Use the 5 features that the API will accept
    feature_cols = ["Age", "Income", "Total_Spending", "Children", "Education"]
    X = df[feature_cols].copy()
    y = cluster_labels

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Build a small pipeline: scale the 5 features, then classify
    classifier_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=1000.0, max_iter=200,
            penalty="l2", solver="lbfgs",
        )),
    ])

    classifier_pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = classifier_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n[INFO] Classifier Training Results:")
    print(f"       Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    return classifier_pipeline


def save_pickle(obj, filepath):
    """Save a Python object to a pickle file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    print(f"[INFO] Saved: {filepath}")


def main():
    print("=" * 60)
    print("  Customer Categorization — Model Training & Saving")
    print("=" * 60)

    # 1. Load and preprocess data
    df = load_and_preprocess_data()

    # 2. Build full preprocessor (for reference / future use)
    preprocessor, scaled_df = build_preprocessor(df)

    # 3. Fit PCA
    pca, pca_data = fit_pca(scaled_df)

    # 4. Cluster
    cluster_labels = perform_clustering(pca_data)

    # 5. Train classifier on the 5 user-facing features
    classifier_pipeline = train_classifier(df, cluster_labels)

    # 6. Save models
    save_pickle(preprocessor, os.path.join("models", "preprocessor.pkl"))
    save_pickle(pca, os.path.join("models", "pca.pkl"))
    save_pickle(classifier_pipeline, os.path.join("models", "model.pkl"))

    print("\n" + "=" * 60)
    print("  All models saved successfully to models/ directory!")
    print("=" * 60)


if __name__ == "__main__":
    main()
