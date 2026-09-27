#!/usr/bin/env python3
"""
Ames Housing / Kaggle House Prices - Task 1
Data cleaning and data-quality audit.

Usage:
    python clean_data.py train.csv ames_cleaned.csv

The script:
1. Produces a complete missing-value report.
2. Applies domain-aware missing-value handling.
3. Retypes MSSubClass as categorical in the in-memory DataFrame.
4. Flags IQR outliers in GrLivArea, LotArea, and TotalBsmtSF.
5. Does not remove rows based on outliers.
6. Does not use SalePrice to make cleaning decisions.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


CATEGORICAL_NONE = [
    "Alley", "PoolQC", "MiscFeature", "Fence", "FireplaceQu",
    "MasVnrType", "GarageType", "GarageFinish", "GarageQual",
    "GarageCond", "BsmtQual", "BsmtCond", "BsmtExposure",
    "BsmtFinType1", "BsmtFinType2",
]

STRUCTURAL_ZERO = [
    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
    "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath",
    "GarageCars", "GarageArea",
]

OUTLIER_COLUMNS = ["GrLivArea", "LotArea", "TotalBsmtSF"]


def iqr_bounds(series: pd.Series):
    q1 = series.dropna().quantile(0.25)
    q3 = series.dropna().quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def clean_ames(input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    # Keep the target available, but never use it for cleaning decisions.
    if "SalePrice" not in df.columns:
        raise ValueError("Expected target column 'SalePrice' was not found.")

    # Complete missing-value audit before changing the data.
    missing_report = pd.DataFrame({
        "column": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "missing_count": df.isna().sum().values,
        "missing_pct": (df.isna().mean() * 100).round(3).values,
        "non_missing_count": df.notna().sum().values,
    })
    report_path = Path(output_path).with_name("MISSING_VALUE_REPORT.csv")
    missing_report.to_csv(report_path, index=False)

    clean = df.copy()

    # Numeric-looking but conceptually categorical code.
    clean["MSSubClass"] = clean["MSSubClass"].astype("category")

    # Missing means the feature is absent, not unknown.
    for col in CATEGORICAL_NONE:
        if col in clean.columns:
            clean[col] = clean[col].fillna("NoFeature")

    # Missing structural measurements/counts mean the structure is absent.
    for col in STRUCTURAL_ZERO:
        if col in clean.columns:
            clean[col] = clean[col].fillna(0)

    # GarageYrBlt: 0 encodes no garage; retain an explicit missingness flag.
    if "GarageYrBlt" in clean.columns:
        clean["GarageYrBlt_missing"] = clean["GarageYrBlt"].isna().astype("int8")
        clean["GarageYrBlt"] = clean["GarageYrBlt"].fillna(0)

    # LotFrontage: impute using the median within Neighborhood and retain
    # a missingness indicator. This avoids a single global mean/median.
    if "LotFrontage" in clean.columns:
        clean["LotFrontage_missing"] = clean["LotFrontage"].isna().astype("int8")
        neighborhood_medians = clean.groupby("Neighborhood")["LotFrontage"].transform("median")
        clean["LotFrontage"] = clean["LotFrontage"].fillna(neighborhood_medians)
        clean["LotFrontage"] = clean["LotFrontage"].fillna(clean["LotFrontage"].median())

    # Only one Electrical value is missing; use the observed mode.
    if "Electrical" in clean.columns:
        clean["Electrical"] = clean["Electrical"].fillna(
            clean["Electrical"].mode(dropna=True).iloc[0]
        )

    # Flag, but do not remove, IQR outliers. Thresholds are computed from
    # predictors only and never from SalePrice.
    for col in OUTLIER_COLUMNS:
        low, high = iqr_bounds(df[col])
        clean[f"{col}_outlier"] = (
            (clean[col] < low) | (clean[col] > high)
        ).astype("int8")

    # Quality checks.
    remaining_missing = int(clean.isna().sum().sum())
    if remaining_missing != 0:
        raise AssertionError(f"Cleaning left {remaining_missing} missing values.")

    clean.to_csv(output_path, index=False)

    print(f"Input shape:   {df.shape}")
    print(f"Output shape:  {clean.shape}")
    print(f"Missing values before: {int(df.isna().sum().sum())}")
    print(f"Missing values after:  {remaining_missing}")
    print(f"Exact duplicate rows:  {int(df.duplicated().sum())}")
    print(f"Saved cleaned data to: {output_path}")
    print(f"Saved missing report to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="Raw Ames/Kaggle train.csv")
    parser.add_argument(
        "output_csv",
        nargs="?",
        default="ames_cleaned.csv",
        help="Output cleaned CSV",
    )
    args = parser.parse_args()
    clean_ames(args.input_csv, args.output_csv)
