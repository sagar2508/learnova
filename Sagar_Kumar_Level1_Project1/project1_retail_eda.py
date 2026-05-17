"""
Project 1: Exploratory Data Analysis on Retail Menu Data
Dataset: McDonald's menu nutrition (menu.csv) — retail product catalog analysis.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA_PATH = Path(__file__).resolve().parent / "dataset" / "menu.csv"
FIG_DIR = Path(__file__).resolve().parent / "outputs" / "figures"

NUMERIC_COLS = [
    "Calories",
    "Calories from Fat",
    "Total Fat",
    "Saturated Fat",
    "Trans Fat",
    "Cholesterol",
    "Sodium",
    "Carbohydrates",
    "Dietary Fiber",
    "Sugars",
    "Protein",
]

DAYPART_ORDER = [
    "Breakfast",
    "Beef & Pork",
    "Chicken & Fish",
    "Salads",
    "Snacks & Sides",
    "Desserts",
    "Beverages",
    "Coffee & Tea",
    "Smoothies & Shakes",
]


def parse_serving_oz(value: str) -> float | np.nan:
    if pd.isna(value):
        return np.nan
    match = re.search(r"([\d.]+)\s*oz", str(value), re.I)
    return float(match.group(1)) if match else np.nan


def parse_size_tier(item: str) -> str:
    text = str(item).lower()
    for tier in ("snack", "small", "medium", "large"):
        if tier in text:
            return tier.title()
    return "Standard"


def load_and_clean(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["Item"] = df["Item"].astype(str).str.strip()
    df["Category"] = df["Category"].astype(str).str.strip()
    df["serving_oz"] = df["Serving Size"].apply(parse_serving_oz)
    df["calories_per_oz"] = np.where(
        df["serving_oz"] > 0, df["Calories"] / df["serving_oz"], np.nan
    )
    df["size_tier"] = df["Item"].apply(parse_size_tier)
    df["health_segment"] = pd.cut(
        df["Calories"],
        bins=[-np.inf, 300, 500, 700, np.inf],
        labels=["Light (<=300)", "Moderate (301-500)", "Heavy (501-700)", "Indulgent (>700)"],
    )
    return df


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    stats = {}
    for col in NUMERIC_COLS:
        series = df[col].dropna()
        mode_vals = series.mode()
        stats[col] = {
            "count": int(series.count()),
            "mean": round(series.mean(), 2),
            "median": round(series.median(), 2),
            "mode": round(mode_vals.iloc[0], 2) if len(mode_vals) else np.nan,
            "std": round(series.std(ddof=1), 2),
            "min": round(series.min(), 2),
            "max": round(series.max(), 2),
        }
    return pd.DataFrame(stats).T


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("Category", observed=True)
        .agg(
            items=("Item", "count"),
            avg_calories=("Calories", "mean"),
            avg_protein=("Protein", "mean"),
            avg_sodium=("Sodium", "mean"),
            avg_sugars=("Sugars", "mean"),
        )
        .round(2)
    )
    ordered = [c for c in DAYPART_ORDER if c in agg.index]
    extra = [c for c in agg.index if c not in ordered]
    return agg.reindex(ordered + extra)


def plot_category_counts(df: pd.DataFrame) -> None:
    counts = df["Category"].value_counts().reindex(
        [c for c in DAYPART_ORDER if c in df["Category"].unique()],
        fill_value=0,
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=counts.values, y=counts.index, hue=counts.index, palette="viridis", legend=False, ax=ax)
    ax.set_title("Product Mix by Menu Category")
    ax.set_xlabel("Number of SKUs")
    ax.set_ylabel("Category")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_category_product_mix.png", dpi=150)
    plt.close(fig)


def plot_daypart_calorie_trend(df: pd.DataFrame) -> None:
    trend = (
        df.groupby("Category", observed=True)["Calories"]
        .mean()
        .reindex([c for c in DAYPART_ORDER if c in df["Category"].unique()])
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(trend.index, trend.values, marker="o", linewidth=2, color="#c0392b")
    ax.fill_between(range(len(trend)), trend.values, alpha=0.15, color="#c0392b")
    ax.set_xticks(range(len(trend)))
    ax.set_xticklabels(trend.index, rotation=35, ha="right")
    ax.set_title("Average Calories Across Menu Dayparts (Category Trend)")
    ax.set_ylabel("Mean Calories")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_daypart_calorie_trend.png", dpi=150)
    plt.close(fig)


def plot_size_tier_trend(df: pd.DataFrame) -> None:
    sized = df[df["size_tier"].isin(["Snack", "Small", "Medium", "Large"])]
    if sized.empty:
        return
    tier_order = ["Snack", "Small", "Medium", "Large"]
    trend = (
        sized.groupby("size_tier", observed=True)["Calories"]
        .mean()
        .reindex(tier_order)
        .dropna()
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(trend.index, trend.values, marker="s", linewidth=2, color="#2980b9")
    ax.set_title("Calorie Trend by Beverage/Dessert Size Tier")
    ax.set_ylabel("Mean Calories")
    ax.set_xlabel("Size Tier")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_size_tier_calorie_trend.png", dpi=150)
    plt.close(fig)


def plot_health_segment_distribution(df: pd.DataFrame) -> None:
    seg = df["health_segment"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=seg.index.astype(str), y=seg.values, hue=seg.index.astype(str), palette="Set2", legend=False, ax=ax)
    ax.set_title("Customer-Oriented Segments by Calorie Band")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Number of Menu Items")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_health_segment_distribution.png", dpi=150)
    plt.close(fig)


def plot_top_products(df: pd.DataFrame) -> None:
    top = df.nlargest(15, "Calories")[["Item", "Category", "Calories"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [f"{row.Item[:40]}…" if len(row.Item) > 40 else row.Item for row in top.itertuples()]
    sns.barplot(y=labels, x=top["Calories"], hue=top["Calories"], palette="Reds_r", legend=False, ax=ax)
    ax.set_title("Top 15 Highest-Calorie Products")
    ax.set_xlabel("Calories")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_top_calorie_products.png", dpi=150)
    plt.close(fig)


def plot_category_nutrition_heatmap(df: pd.DataFrame) -> None:
    heat_cols = ["Calories", "Total Fat", "Sodium", "Sugars", "Protein"]
    matrix = df.groupby("Category", observed=True)[heat_cols].mean()
    matrix = matrix.reindex([c for c in DAYPART_ORDER if c in matrix.index])
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(matrix, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
    ax.set_title("Average Nutrition Profile by Category (Heatmap)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_category_nutrition_heatmap.png", dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df[NUMERIC_COLS].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax)
    ax.set_title("Nutrient Correlation Matrix")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_nutrient_correlation_heatmap.png", dpi=150)
    plt.close(fig)


def plot_sodium_vs_calories(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=df,
        x="Calories",
        y="Sodium",
        hue="Category",
        size="Protein",
        sizes=(30, 200),
        alpha=0.75,
        ax=ax,
    )
    ax.set_title("Product Positioning: Calories vs Sodium (bubble size = Protein)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_calories_vs_sodium_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_recommendations(df: pd.DataFrame, cat_summary: pd.DataFrame) -> None:
    highest_cal_cat = cat_summary["avg_calories"].idxmax()
    lowest_cal_cat = cat_summary["avg_calories"].idxmin()
    high_sodium = df.nlargest(5, "Sodium")[["Item", "Category", "Sodium"]]
    high_sugar = df.nlargest(5, "Sugars")[["Item", "Category", "Sugars"]]

    print("\n" + "=" * 72)
    print("ACTIONABLE RECOMMENDATIONS (based on EDA)")
    print("=" * 72)
    print(
        f"1. Menu engineering: '{highest_cal_cat}' has the highest average calories "
        f"({cat_summary.loc[highest_cal_cat, 'avg_calories']:.0f}). Introduce lighter options "
        f"or bundle with salads/beverages to balance basket health scores."
    )
    print(
        f"2. Promote high-margin dayparts: '{lowest_cal_cat}' averages the fewest calories - "
        "use as an entry point for upselling combos (sides, beverages, desserts)."
    )
    print(
        f"3. Sodium reduction priority: Top sodium items include "
        f"{', '.join(high_sodium['Item'].head(3).tolist())}. Reformulate or flag for sodium-conscious diners."
    )
    print(
        f"4. Sugar-heavy SKUs: Focus reformulation on "
        f"{', '.join(high_sugar['Item'].head(3).tolist())} - critical for smoothie/shake and dessert lines."
    )
    light_share = (df["health_segment"] == "Light (<=300)").mean() * 100
    print(
        f"5. Assortment balance: Only {light_share:.1f}% of items fall in the light (<=300 cal) segment. "
        "Expand light options in Breakfast and Chicken & Fish to capture health-conscious segments."
    )
    print(
        "6. Size-tier pricing: Calories rise sharply from Small to Large on shakes - "
        "use tiered pricing and default Medium to improve margin without surprising customers."
    )
    print("=" * 72 + "\n")


def main() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.05)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and cleaning data...")
    df = load_and_clean()
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"Categories: {df['Category'].nunique()}")
    print(f"Missing values after cleaning: {df.isnull().sum().sum()}")

    print("\n--- Descriptive Statistics (key nutrients) ---")
    stats = descriptive_statistics(df)
    print(stats.loc[["Calories", "Protein", "Sodium", "Sugars", "Total Fat"]])

    print("\n--- Category / Product Summary ---")
    cat_summary = category_summary(df)
    print(cat_summary)

    print("\n--- Health segment counts ---")
    print(df["health_segment"].value_counts())

    print("\nGenerating visualizations...")
    plot_category_counts(df)
    plot_daypart_calorie_trend(df)
    plot_size_tier_trend(df)
    plot_health_segment_distribution(df)
    plot_top_products(df)
    plot_category_nutrition_heatmap(df)
    plot_correlation_heatmap(df)
    plot_sodium_vs_calories(df)
    print(f"Figures saved to: {FIG_DIR.resolve()}")

    stats.to_csv(FIG_DIR.parent / "descriptive_statistics.csv")
    cat_summary.to_csv(FIG_DIR.parent / "category_summary.csv")

    print_recommendations(df, cat_summary)


if __name__ == "__main__":
    main()
