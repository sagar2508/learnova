"""
Project 8: Google Play Store Analysis
Datasets: apps.csv, user_reviews.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

APPS_PATH = Path(__file__).resolve().parent / "dataset" / "apps.csv"
REVIEWS_PATH = Path(__file__).resolve().parent / "dataset" / "user_reviews.csv"
FIG_DIR = Path(__file__).resolve().parent / "outputs" / "figures"
INTERACTIVE_DIR = Path(__file__).resolve().parent / "outputs" / "interactive"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def parse_installs(value: str) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).replace(",", "").replace("+", "").strip()
    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_price(value: str) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).replace("$", "").strip()
    try:
        return float(text)
    except ValueError:
        return np.nan


def load_apps(path: Path = APPS_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_reviews(path: Path = REVIEWS_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_apps(df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Apps: Data Preparation ---")
    print(f"Raw shape: {df.shape}")

    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")], errors="ignore")
    df["App"] = df["App"].astype(str).str.strip()
    df["Category"] = df["Category"].astype(str).str.strip().str.upper()
    df["Genres"] = df["Genres"].astype(str).str.strip()
    df["Type"] = df["Type"].astype(str).str.strip()

    df["installs_numeric"] = df["Installs"].apply(parse_installs)
    df["price_numeric"] = df["Price"].apply(parse_price)
    df["is_free"] = df["price_numeric"].fillna(0) == 0
    df["Last Updated"] = pd.to_datetime(df["Last Updated"], errors="coerce")

    df.loc[df["Rating"] > 5, "Rating"] = np.nan
    df.loc[df["Reviews"] < 0, "Reviews"] = np.nan

    before = len(df)
    df = df.drop_duplicates(subset=["App"], keep="first").reset_index(drop=True)
    print(f"Removed duplicate apps: {before - len(df)}")

    print(f"Missing Rating: {df['Rating'].isna().sum()}")
    print(f"Missing Size: {df['Size'].isna().sum()}")
    print(f"Clean shape: {df.shape}")
    return df


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Reviews: Data Preparation ---")
    print(f"Raw shape: {df.shape}")

    df["App"] = df["App"].astype(str).str.strip()
    df["Translated_Review"] = df["Translated_Review"].replace({"nan": np.nan})
    df = df[df["Translated_Review"].notna()].copy()
    df["Translated_Review"] = df["Translated_Review"].astype(str).str.strip()
    df = df[df["Translated_Review"].str.len() > 0]

    for col in ("Sentiment",):
        df[col] = df[col].astype(str).str.strip().str.title()
        df.loc[~df[col].isin(["Positive", "Negative", "Neutral"]), col] = np.nan

    df["Sentiment_Polarity"] = pd.to_numeric(df["Sentiment_Polarity"], errors="coerce")
    df["Sentiment_Subjectivity"] = pd.to_numeric(df["Sentiment_Subjectivity"], errors="coerce")

    print(f"Reviews with text: {len(df)}")
    print(f"Reviews with sentiment label: {df['Sentiment'].notna().sum()}")
    return df.reset_index(drop=True)


def merge_apps_reviews(apps: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    merged = reviews.merge(
        apps[["App", "Category", "Rating", "price_numeric", "installs_numeric", "Type"]],
        on="App",
        how="left",
    )
    return merged


def category_exploration(apps: pd.DataFrame) -> pd.DataFrame:
    summary = (
        apps.groupby("Category", observed=True)
        .agg(
            app_count=("App", "count"),
            avg_rating=("Rating", "mean"),
            median_installs=("installs_numeric", "median"),
            avg_price=("price_numeric", "mean"),
            paid_share=("is_free", lambda s: (~s).mean()),
        )
        .round(3)
        .sort_values("app_count", ascending=False)
    )
    print("\n--- Category Exploration (top 10 by app count) ---")
    print(summary.head(10))
    return summary


def metrics_analysis(apps: pd.DataFrame) -> dict:
    rated = apps.dropna(subset=["Rating"])
    sized = apps.dropna(subset=["Size"])
    paid = apps[apps["price_numeric"] > 0]

    stats = {
        "avg_rating": round(rated["Rating"].mean(), 3),
        "median_rating": round(rated["Rating"].median(), 3),
        "avg_reviews": round(apps["Reviews"].mean(), 1),
        "median_installs": round(apps["installs_numeric"].median(), 0),
        "free_apps_pct": round(apps["is_free"].mean() * 100, 1),
        "avg_paid_price": round(paid["price_numeric"].mean(), 2) if len(paid) else 0,
        "avg_size_mb": round(sized["Size"].mean(), 2),
    }
    print("\n--- Metrics Analysis ---")
    for key, val in stats.items():
        print(f"{key}: {val}")
    return stats


def sentiment_analysis(reviews: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    labeled = reviews.dropna(subset=["Sentiment"])
    dist = labeled["Sentiment"].value_counts(normalize=True).mul(100).round(2)
    print("\n--- Sentiment Analysis (overall) ---")
    print(dist)

    by_category = (
        merged.dropna(subset=["Sentiment", "Category"])
        .groupby(["Category", "Sentiment"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    by_category_pct = by_category.div(by_category.sum(axis=1), axis=0).mul(100).round(2)
    by_category_pct.to_csv(OUTPUT_DIR / "sentiment_by_category.csv")
    print("\nSentiment % by category saved.")
    return by_category_pct


# --- Static plots ---


def plot_category_distribution(apps: pd.DataFrame) -> None:
    counts = apps["Category"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=counts.values, y=counts.index, hue=counts.index, palette="mako", legend=False, ax=ax)
    ax.set_title("Top 15 App Categories by Count")
    ax.set_xlabel("Number of Apps")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_category_distribution.png", dpi=150)
    plt.close(fig)


def plot_rating_distribution(apps: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(apps["Rating"].dropna(), bins=20, kde=True, ax=ax, color="#3498db")
    ax.set_title("App Rating Distribution")
    ax.set_xlabel("Rating")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_rating_distribution.png", dpi=150)
    plt.close(fig)


def plot_price_analysis(apps: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(data=apps, x="Type", hue="Type", ax=axes[0], palette="pastel", legend=False)
    axes[0].set_title("Free vs Paid Apps")
    paid = apps[apps["price_numeric"] > 0]
    sns.histplot(paid["price_numeric"], bins=30, ax=axes[1], color="#e74c3c")
    axes[1].set_title("Paid App Price Distribution ($)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_pricing_analysis.png", dpi=150)
    plt.close(fig)


def plot_size_vs_rating(apps: pd.DataFrame) -> None:
    subset = apps.dropna(subset=["Size", "Rating"])
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=subset, x="Size", y="Rating", alpha=0.35, ax=ax)
    ax.set_title("App Size (MB) vs Rating")
    ax.set_xlabel("Size (MB)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_size_vs_rating.png", dpi=150)
    plt.close(fig)


def plot_installs_vs_rating(apps: pd.DataFrame) -> None:
    subset = apps.dropna(subset=["installs_numeric", "Rating"]).copy()
    subset = subset[subset["installs_numeric"] > 0]
    subset["log_installs"] = np.log10(subset["installs_numeric"] + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=subset, x="log_installs", y="Rating", alpha=0.3, ax=ax)
    ax.set_title("Popularity (log Installs) vs Rating")
    ax.set_xlabel("log10(Installs)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_installs_vs_rating.png", dpi=150)
    plt.close(fig)


def plot_sentiment_distribution(reviews: pd.DataFrame) -> None:
    labeled = reviews.dropna(subset=["Sentiment"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sentiment_counts = labeled["Sentiment"].value_counts()
    axes[0].pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", startangle=90)
    axes[0].set_title("Review Sentiment Distribution")
    sns.histplot(labeled["Sentiment_Polarity"].dropna(), bins=30, kde=True, ax=axes[1], color="#9b59b6")
    axes[1].set_title("Sentiment Polarity Distribution")
    axes[1].set_xlabel("Polarity")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_sentiment_analysis.png", dpi=150)
    plt.close(fig)


def plot_sentiment_by_top_categories(merged: pd.DataFrame) -> None:
    top_cats = merged["Category"].value_counts().head(8).index
    subset = merged[merged["Category"].isin(top_cats)].dropna(subset=["Sentiment"])
    ct = pd.crosstab(subset["Category"], subset["Sentiment"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    ct.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_title("Sentiment Mix by Top Categories (%)")
    ax.set_ylabel("Percentage")
    ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1))
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_sentiment_by_category.png", dpi=150)
    plt.close(fig)


# --- Interactive Plotly charts ---


def save_interactive_category_chart(apps: pd.DataFrame) -> None:
    counts = apps["Category"].value_counts().reset_index()
    counts.columns = ["Category", "App Count"]
    fig = px.bar(
        counts.head(20),
        x="App Count",
        y="Category",
        orientation="h",
        title="Google Play Store - App Count by Category (Interactive)",
        color="App Count",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    fig.write_html(str(INTERACTIVE_DIR / "01_category_distribution.html"))


def save_interactive_rating_installs(apps: pd.DataFrame) -> None:
    subset = apps.dropna(subset=["Rating", "installs_numeric"]).copy()
    subset = subset[subset["installs_numeric"] > 0]
    fig = px.scatter(
        subset,
        x="installs_numeric",
        y="Rating",
        color="Category",
        hover_name="App",
        log_x=True,
        title="Rating vs Installs (Interactive)",
        opacity=0.6,
    )
    fig.write_html(str(INTERACTIVE_DIR / "02_rating_vs_installs.html"))


def save_interactive_price_by_category(apps: pd.DataFrame) -> None:
    top = apps["Category"].value_counts().head(12).index
    subset = apps[apps["Category"].isin(top)]
    fig = px.box(
        subset,
        x="Category",
        y="price_numeric",
        color="Category",
        title="App Price by Category (Interactive)",
        points="outliers",
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-35)
    fig.write_html(str(INTERACTIVE_DIR / "03_price_by_category.html"))


def save_interactive_sentiment_sunburst(merged: pd.DataFrame) -> None:
    labeled = merged.dropna(subset=["Sentiment", "Category"])
    top_cats = labeled["Category"].value_counts().head(10).index
    subset = labeled[labeled["Category"].isin(top_cats)]
    counts = subset.groupby(["Category", "Sentiment"], observed=True).size().reset_index(name="count")
    fig = px.sunburst(
        counts,
        path=["Category", "Sentiment"],
        values="count",
        title="Review Sentiment by Category (Interactive Sunburst)",
    )
    fig.write_html(str(INTERACTIVE_DIR / "04_sentiment_sunburst.html"))


def save_interactive_size_rating(apps: pd.DataFrame) -> None:
    subset = apps.dropna(subset=["Size", "Rating", "Category"])
    fig = px.density_heatmap(
        subset,
        x="Size",
        y="Rating",
        title="App Size vs Rating Density (Interactive)",
        nbinsx=30,
        nbinsy=20,
    )
    fig.write_html(str(INTERACTIVE_DIR / "05_size_rating_heatmap.html"))


def print_recommendations(apps: pd.DataFrame, cat_summary: pd.DataFrame, stats: dict) -> None:
    top_cat = cat_summary.index[0]
    best_rated_cat = cat_summary["avg_rating"].idxmax()
    print("\n" + "=" * 72)
    print("INSIGHTS & RECOMMENDATIONS")
    print("=" * 72)
    print(
        f"1. Market focus: '{top_cat}' dominates with {int(cat_summary.loc[top_cat, 'app_count'])} apps - "
        "high competition; differentiation and niche sub-genres matter."
    )
    print(
        f"2. Quality benchmark: '{best_rated_cat}' leads average rating "
        f"({cat_summary.loc[best_rated_cat, 'avg_rating']:.2f}) - study top apps there for UX patterns."
    )
    print(
        f"3. Monetization: {stats['free_apps_pct']}% apps are free; paid apps average ${stats['avg_paid_price']}. "
        "Freemium + in-app purchases likely dominate revenue."
    )
    print(
        "4. Size vs satisfaction: Very large apps do not always rate higher - optimize APK size "
        "for emerging markets with storage constraints."
    )
    print(
        "5. Sentiment: Positive reviews dominate; monitor Negative spikes after releases "
        "using category-level sentiment dashboards."
    )
    print(
        "6. Use interactive HTML charts in outputs/interactive/ for stakeholder exploration "
        "of category, price, and sentiment dynamics."
    )
    print("=" * 72 + "\n")


def main() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.05)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Google Play Store Analysis")
    print("=" * 72)

    apps = clean_apps(load_apps())
    reviews = clean_reviews(load_reviews())
    merged = merge_apps_reviews(apps, reviews)

    cat_summary = category_exploration(apps)
    stats = metrics_analysis(apps)
    sentiment_analysis(reviews, merged)

    cat_summary.to_csv(OUTPUT_DIR / "category_summary.csv")
    pd.DataFrame([stats]).to_csv(OUTPUT_DIR / "market_metrics.csv", index=False)

    print("\nGenerating static visualizations...")
    plot_category_distribution(apps)
    plot_rating_distribution(apps)
    plot_price_analysis(apps)
    plot_size_vs_rating(apps)
    plot_installs_vs_rating(apps)
    plot_sentiment_distribution(reviews)
    plot_sentiment_by_top_categories(merged)

    print("Generating interactive visualizations...")
    save_interactive_category_chart(apps)
    save_interactive_rating_installs(apps)
    save_interactive_price_by_category(apps)
    save_interactive_sentiment_sunburst(merged)
    save_interactive_size_rating(apps)

    print_recommendations(apps, cat_summary, stats)
    print(f"Static figures: {FIG_DIR.resolve()}")
    print(f"Interactive charts: {INTERACTIVE_DIR.resolve()}")


if __name__ == "__main__":
    main()
