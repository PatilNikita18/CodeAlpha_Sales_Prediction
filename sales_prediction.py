"""
TASK 4: Sales Prediction using Python
======================================
Predicts sales from advertising spend, target segment, and platform.
Pipeline: load -> clean -> feature engineering -> train regression models
-> evaluate -> analyze ad-spend impact -> business insights -> visualizations.

HOW TO USE
----------
1. Download the dataset from the task sheet link and save it as "data.csv"
   in the same folder as this script (or change DATA_PATH below).
2. Install requirements:
       pip install pandas numpy scikit-learn matplotlib seaborn
3. Run:
       python sales_prediction_task4.py

The script auto-detects common column name variants (e.g. "TV Ad Spend",
"tv_spend", "Advertising", "Segment", "Platform", "Sales", "Revenue") so it
should work even if your CSV's exact headers differ slightly. If it can't
find a column it needs, it will print a clear message telling you what to
rename in your CSV.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (9, 5)

DATA_PATH = "data.csv"   # <-- change if your file has a different name


# --------------------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns.")
    print("\nColumns found:", list(df.columns))
    print("\nFirst 5 rows:\n", df.head())
    return df


# --------------------------------------------------------------------------
# 2. AUTO-DETECT KEY COLUMNS (spend, segment, platform, target)
# --------------------------------------------------------------------------
def find_column(df, keywords):
    """Return the first column whose name contains any keyword (case-insensitive)."""
    for col in df.columns:
        for kw in keywords:
            if kw.lower() in col.lower():
                return col
    return None


def detect_schema(df):
    schema = {
        "spend_cols": [c for c in df.columns if any(
            k in c.lower() for k in ["spend", "budget", "ad", "advertis", "tv", "radio", "newspaper", "digital", "social"]
        ) and df[c].dtype != object],
        "segment_col": find_column(df, ["segment", "audience", "customer_type", "demographic"]),
        "platform_col": find_column(df, ["platform", "channel", "medium"]),
        "target_col": find_column(df, ["sales", "revenue", "units_sold", "target"]),
    }
    print("\nDetected schema:")
    for k, v in schema.items():
        print(f"  {k}: {v}")
    return schema


# --------------------------------------------------------------------------
# 3. CLEANING
# --------------------------------------------------------------------------
def clean_data(df, target_col):
    df = df.copy()

    # Drop obvious ID columns
    id_like = [c for c in df.columns if c.lower() in ("id", "index", "unnamed: 0")]
    df = df.drop(columns=id_like, errors="ignore")

    # Drop rows with missing target
    before = len(df)
    df = df.dropna(subset=[target_col])
    print(f"\nDropped {before - len(df)} rows with missing target.")

    # Fill missing numeric values with median, categorical with mode
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue
        if df[col].dtype in (np.float64, np.int64):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    # Drop exact duplicate rows
    dup = df.duplicated().sum()
    if dup:
        df = df.drop_duplicates()
        print(f"Dropped {dup} duplicate rows.")

    return df


# --------------------------------------------------------------------------
# 4. EDA (quick, saved as PNGs so you can drop them straight into a report)
# --------------------------------------------------------------------------
def run_eda(df, schema):
    target = schema["target_col"]

    # Correlation heatmap (numeric columns only)
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] > 1:
        plt.figure(figsize=(7, 6))
        sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("eda_correlation_heatmap.png", dpi=150)
        plt.close()

    # Spend vs Sales scatterplots
    for col in schema["spend_cols"]:
        plt.figure()
        sns.regplot(x=df[col], y=df[target], scatter_kws={"alpha": 0.5})
        plt.title(f"{col} vs {target}")
        plt.tight_layout()
        plt.savefig(f"eda_{col}_vs_{target}.png", dpi=150)
        plt.close()

    # Sales by segment / platform
    for cat_col in [schema["segment_col"], schema["platform_col"]]:
        if cat_col:
            plt.figure()
            sns.boxplot(x=df[cat_col], y=df[target])
            plt.title(f"{target} by {cat_col}")
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.savefig(f"eda_{target}_by_{cat_col}.png", dpi=150)
            plt.close()

    print("\nEDA plots saved as PNG files in the working directory.")


# --------------------------------------------------------------------------
# 5. FEATURE ENGINEERING + MODEL PIPELINE
# --------------------------------------------------------------------------
def build_pipeline(df, schema):
    target = schema["target_col"]
    numeric_features = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    categorical_features = [c for c in [schema["segment_col"], schema["platform_col"]] if c]

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])

    X = df[numeric_features + categorical_features]
    y = df[target]
    return X, y, preprocessor, numeric_features, categorical_features


# --------------------------------------------------------------------------
# 6. TRAIN + EVALUATE MULTIPLE MODELS
# --------------------------------------------------------------------------
def train_and_evaluate(X, y, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=42),
    }

    results = {}
    fitted_pipelines = {}

    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="r2")

        results[name] = {
            "R2": r2, "MAE": mae, "RMSE": rmse,
            "CV_R2_mean": cv_scores.mean(), "CV_R2_std": cv_scores.std(),
        }
        fitted_pipelines[name] = pipe

        print(f"\n--- {name} ---")
        print(f"  Test R^2:  {r2:.4f}")
        print(f"  Test MAE:  {mae:.4f}")
        print(f"  Test RMSE: {rmse:.4f}")
        print(f"  5-fold CV R^2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    best_name = max(results, key=lambda n: results[n]["R2"])
    print(f"\nBest model on held-out test set: {best_name}")
    return fitted_pipelines, results, best_name, X_test, y_test


# --------------------------------------------------------------------------
# 7. ANALYZE HOW AD SPEND IMPACTS SALES
# --------------------------------------------------------------------------
def analyze_impact(fitted_pipelines, numeric_features, categorical_features, best_name):
    pipe = fitted_pipelines[best_name]
    model = pipe.named_steps["model"]
    prep = pipe.named_steps["prep"]

    # Get feature names after one-hot encoding
    ohe = prep.named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(categorical_features)) if categorical_features else []
    all_feature_names = numeric_features + cat_names

    if hasattr(model, "coef_"):
        importance = pd.Series(model.coef_, index=all_feature_names)
        title = "Linear Regression Coefficients (impact on Sales)"
    elif hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=all_feature_names)
        title = "Random Forest Feature Importances"
    else:
        print("Model has no interpretable coefficients/importances.")
        return

    importance = importance.sort_values(key=abs, ascending=False)
    print(f"\n{title}:\n", importance)

    plt.figure(figsize=(8, 5))
    importance.plot(kind="barh")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("feature_impact.png", dpi=150)
    plt.close()
    print("Saved feature_impact.png")

    return importance


# --------------------------------------------------------------------------
# 8. BUSINESS INSIGHTS
# --------------------------------------------------------------------------
def print_business_insights(importance, spend_cols):
    print("\n" + "=" * 60)
    print("ACTIONABLE BUSINESS INSIGHTS")
    print("=" * 60)

    spend_impact = {c: importance.get(c, 0) for c in spend_cols if c in importance.index}
    if spend_impact:
        ranked = sorted(spend_impact.items(), key=lambda x: abs(x[1]), reverse=True)
        top_channel, top_val = ranked[0]
        print(f"- '{top_channel}' has the strongest measured effect on sales "
              f"(impact score: {top_val:.3f}). Prioritize budget here first.")
        if len(ranked) > 1:
            weak_channel, weak_val = ranked[-1]
            print(f"- '{weak_channel}' shows the weakest effect (impact score: {weak_val:.3f}); "
                  f"consider testing reduced spend or a different creative/targeting approach there.")

    cat_effects = [f for f in importance.index if f not in spend_cols]
    if cat_effects:
        top_cat = importance[cat_effects].abs().idxmax()
        print(f"- Among segment/platform categories, '{top_cat}' shows the largest deviation "
              f"in predicted sales — worth a dedicated campaign or messaging test.")

    print("- Recommendation: re-run this analysis periodically as new campaign data arrives, "
          "since channel effectiveness typically shifts over time (seasonality, ad fatigue, "
          "platform algorithm changes).")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    df = load_data(DATA_PATH)
    schema = detect_schema(df)

    if not schema["target_col"]:
        print("\n[!] Could not find a sales/target column automatically.")
        print("    Rename your target column to include 'sales' or 'revenue', then re-run.")
        return
    if not schema["spend_cols"]:
        print("\n[!] Could not find any advertising spend columns automatically.")
        print("    Rename spend columns to include a keyword like 'spend', 'ad', 'TV', 'radio', etc.")

    df = clean_data(df, schema["target_col"])
    run_eda(df, schema)

    X, y, preprocessor, numeric_features, categorical_features = build_pipeline(df, schema)
    fitted_pipelines, results, best_name, X_test, y_test = train_and_evaluate(X, y, preprocessor)

    importance = analyze_impact(fitted_pipelines, numeric_features, categorical_features, best_name)
    if importance is not None:
        print_business_insights(importance, schema["spend_cols"])

    print("\nDone. Check the working directory for saved PNG charts "
          "(EDA plots + feature_impact.png) to include in your submission.")


if __name__ == "__main__":
    main()