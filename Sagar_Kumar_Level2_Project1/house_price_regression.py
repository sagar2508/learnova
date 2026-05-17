"""
Project 5: Predicting House Prices with Linear Regression
Dataset: Housing.csv (price as target variable)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).resolve().parent / "dataset" / "Housing.csv"
FIG_DIR = Path(__file__).resolve().parent / "outputs" / "figures"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

TARGET = "price"
NUMERIC_FEATURES = ["area", "bedrooms", "bathrooms", "stories", "parking"]
CATEGORICAL_FEATURES = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus",
]
YES_NO_COLS = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def explore_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Data Exploration ---")
    print(f"Shape: {df.shape}")
    print(f"Target variable: {TARGET}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    df = df.drop_duplicates().reset_index(drop=True)

    for col in YES_NO_COLS:
        df[col] = df[col].astype(str).str.strip().str.lower()
    df["furnishingstatus"] = df["furnishingstatus"].astype(str).str.strip().str.lower()

    valid_furnishing = {"furnished", "semi-furnished", "unfurnished"}
    df = df[df["furnishingstatus"].isin(valid_furnishing)].reset_index(drop=True)

    for col in YES_NO_COLS:
        df = df[df[col].isin(["yes", "no"])].reset_index(drop=True)

    print(f"Price range: {df[TARGET].min():,.0f} - {df[TARGET].max():,.0f}")
    print(f"Rows after cleaning: {len(df)}")
    return df


def select_features(df: pd.DataFrame, min_abs_corr: float = 0.15) -> tuple[list[str], list[str], pd.Series]:
    numeric_df = df[NUMERIC_FEATURES + [TARGET]]
    corr_with_price = numeric_df.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)

    print("\n--- Feature Selection (numeric correlation with price) ---")
    print(corr_with_price.round(3))

    selected_numeric = corr_with_price[corr_with_price.abs() >= min_abs_corr].index.tolist()
    if not selected_numeric:
        selected_numeric = NUMERIC_FEATURES.copy()

    print(f"\nSelected numeric features (|r| >= {min_abs_corr}): {selected_numeric}")
    print(f"Selected categorical features: {CATEGORICAL_FEATURES}")
    return selected_numeric, CATEGORICAL_FEATURES.copy(), corr_with_price


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )


def train_model(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = df[numeric_features + categorical_features]
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_features, categorical_features)),
            ("regressor", LinearRegression()),
        ]
    )

    print("\n--- Model Training ---")
    model.fit(X_train, y_train)
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    return model, y_train, y_test, y_pred_train, y_pred_test


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    print(f"\n--- {label} Evaluation ---")
    print(f"MSE:       {mse:,.2f}")
    print(f"RMSE:      {rmse:,.2f}")
    print(f"R-squared: {r2:.4f}")
    return {"mse": mse, "rmse": rmse, "r2": r2}


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df[NUMERIC_FEATURES + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap (Numeric Features vs Price)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_price_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[TARGET], kde=True, ax=ax, color="#2c3e50")
    ax.set_title("Distribution of House Prices (Target Variable)")
    ax.set_xlabel("Price")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_price_distribution.png", dpi=150)
    plt.close(fig)


def plot_area_vs_price(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="area", y=TARGET, alpha=0.6, ax=ax)
    ax.set_title("Area vs Price")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_area_vs_price.png", dpi=150)
    plt.close(fig)


def plot_actual_vs_predicted(
    y_true: np.ndarray, y_pred: np.ndarray, label: str, filename: str
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.3)
    lo = min(float(y_true.min()), float(y_pred.min()))
    hi = max(float(y_true.max()), float(y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="Perfect prediction")
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title(f"Actual vs Predicted Prices ({label})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150)
    plt.close(fig)


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, ax=axes[0])
    axes[0].axhline(0, color="r", linestyle="--")
    axes[0].set_xlabel("Predicted Price")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residual Plot (Test Set)")
    sns.histplot(residuals, kde=True, ax=axes[1], color="#8e44ad")
    axes[1].set_title("Residual Distribution")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_residual_analysis.png", dpi=150)
    plt.close(fig)


def save_coefficients(
    model: Pipeline, numeric_features: list[str], categorical_features: list[str]
) -> pd.DataFrame:
    preprocess = model.named_steps["preprocess"]
    regressor = model.named_steps["regressor"]
    cat_encoder: OneHotEncoder = preprocess.named_transformers_["cat"]
    cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    feature_names = numeric_features + cat_names

    coef_df = pd.DataFrame(
        {"feature": feature_names, "coefficient": regressor.coef_}
    ).sort_values("coefficient", key=abs, ascending=False)
    coef_df.to_csv(OUTPUT_DIR / "model_coefficients.csv", index=False)
    print("\nTop 10 coefficients by magnitude:")
    print(coef_df.head(10).to_string(index=False))
    return coef_df


def main() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.05)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("House Price Prediction - Linear Regression")
    print("=" * 72)

    df = load_data()
    df = explore_and_clean(df)
    numeric_features, categorical_features, corr = select_features(df)

    plot_correlation_heatmap(df)
    plot_price_distribution(df)
    plot_area_vs_price(df)

    model, y_train, y_test, y_pred_train, y_pred_test = train_model(
        df, numeric_features, categorical_features
    )

    train_metrics = evaluate_model(y_train, y_pred_train, "Training")
    test_metrics = evaluate_model(y_test, y_pred_test, "Test")

    plot_actual_vs_predicted(y_train, y_pred_train, "Training", "04_actual_vs_predicted_train.png")
    plot_actual_vs_predicted(y_test, y_pred_test, "Test", "04_actual_vs_predicted_test.png")
    plot_residuals(y_test, y_pred_test)

    save_coefficients(model, numeric_features, categorical_features)

    pd.DataFrame([{"split": "train", **train_metrics}, {"split": "test", **test_metrics}]).to_csv(
        OUTPUT_DIR / "model_metrics.csv", index=False
    )
    corr.to_csv(OUTPUT_DIR / "feature_correlations.csv")

    print(f"\nFigures saved to: {FIG_DIR.resolve()}")
    print(f"Metrics saved to: {OUTPUT_DIR / 'model_metrics.csv'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
