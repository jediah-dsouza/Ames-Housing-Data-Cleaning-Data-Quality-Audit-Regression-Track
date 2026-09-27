# DATA QUALITY REPORT — Ames Housing / House Prices

## 1. Scope and objective

This audit was performed on the uploaded Kaggle-style Ames Housing training file (`train.csv`), containing **1,460 rows and 81 columns**.

The objective is to create a documented, cleaned version for the regression track while avoiding premature modeling steps such as scaling or one-hot encoding.

- **Source:** Kaggle, *House Prices - Advanced Regression Techniques*
- **Target:** `SalePrice`
- **Predictors:** 79 explanatory variables, plus `Id` and `SalePrice`
- **Task:** data-quality audit and cleaning only
- **Cleaning decisions:** do **not** use `SalePrice` to impute, remove, or classify predictor records.
- **Rows dropped:** 0
- **Columns dropped:** 0
- **Final cleaned file:** `ames_cleaned.csv`
- **Reproducible script:** `clean_data.py`

> **Important leakage note:** This Task 1 exercise audits the full raw training file as requested. Once train/validation/test splitting begins, every learned cleaning statistic (for example a median or mode) must be fitted on the training portion only and then applied to validation/test data. `SalePrice` must never be used to choose predictor cleaning rules.

---

## 2. Why this task matters

Real-world datasets contain missing values, inconsistent types, duplicate-like records, and extreme observations. If these issues are ignored, later models can learn from artifacts rather than the intended signal.

The main lesson here is that **missing does not automatically mean unknown**. In this dataset, many blanks encode the absence of a physical feature: a blank `PoolQC` usually means there is no pool, while a blank `GarageType` indicates no garage. Replacing such values with a global mean or dropping the rows would destroy useful information.

---

## 3. Skills demonstrated

- Systematic missing-value diagnosis
- Full missing-value count and percentage reporting
- Exact duplicate detection
- Near-duplicate screening
- IQR-based numerical outlier detection
- Domain-aware distinction between feature absence and unknown values
- Numerical vs. categorical type correction
- Missingness indicators
- Reproducible Pandas/NumPy cleaning
- Awareness of preprocessing leakage

---

## 4. Missing-value audit

The raw file contains **7,829 missing cells across 19 columns**. The remaining 62 columns have no missing values.

### Interpretation of missingness

The pattern is strongly structured rather than plausibly MCAR (Missing Completely At Random). For example, `PoolQC` is missing for 1,453 of 1,460 observations (99.521%). This is consistent with a feature being absent for most houses, not with random measurement failure.

It is useful to distinguish this from **MNAR (Missing Not At Random)**. A missing value can carry information about the underlying feature or observation process, but this audit does not claim that every missingness mechanism is formally MNAR. The safest conclusion is that several missing fields are **structural/semantic absence indicators** and should be treated explicitly.

### Full missing-value report

| Column | Raw dtype | Missing count | Missing % | Non-missing |
|---|---|---:|---:|---:|
| Id | `int64` | 0 | 0.000% | 1460 |
| MSSubClass | `int64` | 0 | 0.000% | 1460 |
| MSZoning | `object` | 0 | 0.000% | 1460 |
| LotFrontage | `float64` | 259 | 17.740% | 1201 |
| LotArea | `int64` | 0 | 0.000% | 1460 |
| Street | `object` | 0 | 0.000% | 1460 |
| Alley | `object` | 1369 | 93.767% | 91 |
| LotShape | `object` | 0 | 0.000% | 1460 |
| LandContour | `object` | 0 | 0.000% | 1460 |
| Utilities | `object` | 0 | 0.000% | 1460 |
| LotConfig | `object` | 0 | 0.000% | 1460 |
| LandSlope | `object` | 0 | 0.000% | 1460 |
| Neighborhood | `object` | 0 | 0.000% | 1460 |
| Condition1 | `object` | 0 | 0.000% | 1460 |
| Condition2 | `object` | 0 | 0.000% | 1460 |
| BldgType | `object` | 0 | 0.000% | 1460 |
| HouseStyle | `object` | 0 | 0.000% | 1460 |
| OverallQual | `int64` | 0 | 0.000% | 1460 |
| OverallCond | `int64` | 0 | 0.000% | 1460 |
| YearBuilt | `int64` | 0 | 0.000% | 1460 |
| YearRemodAdd | `int64` | 0 | 0.000% | 1460 |
| RoofStyle | `object` | 0 | 0.000% | 1460 |
| RoofMatl | `object` | 0 | 0.000% | 1460 |
| Exterior1st | `object` | 0 | 0.000% | 1460 |
| Exterior2nd | `object` | 0 | 0.000% | 1460 |
| MasVnrType | `object` | 872 | 59.726% | 588 |
| MasVnrArea | `float64` | 8 | 0.548% | 1452 |
| ExterQual | `object` | 0 | 0.000% | 1460 |
| ExterCond | `object` | 0 | 0.000% | 1460 |
| Foundation | `object` | 0 | 0.000% | 1460 |
| BsmtQual | `object` | 37 | 2.534% | 1423 |
| BsmtCond | `object` | 37 | 2.534% | 1423 |
| BsmtExposure | `object` | 38 | 2.603% | 1422 |
| BsmtFinType1 | `object` | 37 | 2.534% | 1423 |
| BsmtFinSF1 | `int64` | 0 | 0.000% | 1460 |
| BsmtFinType2 | `object` | 38 | 2.603% | 1422 |
| BsmtFinSF2 | `int64` | 0 | 0.000% | 1460 |
| BsmtUnfSF | `int64` | 0 | 0.000% | 1460 |
| TotalBsmtSF | `int64` | 0 | 0.000% | 1460 |
| Heating | `object` | 0 | 0.000% | 1460 |
| HeatingQC | `object` | 0 | 0.000% | 1460 |
| CentralAir | `object` | 0 | 0.000% | 1460 |
| Electrical | `object` | 1 | 0.068% | 1459 |
| 1stFlrSF | `int64` | 0 | 0.000% | 1460 |
| 2ndFlrSF | `int64` | 0 | 0.000% | 1460 |
| LowQualFinSF | `int64` | 0 | 0.000% | 1460 |
| GrLivArea | `int64` | 0 | 0.000% | 1460 |
| BsmtFullBath | `int64` | 0 | 0.000% | 1460 |
| BsmtHalfBath | `int64` | 0 | 0.000% | 1460 |
| FullBath | `int64` | 0 | 0.000% | 1460 |
| HalfBath | `int64` | 0 | 0.000% | 1460 |
| BedroomAbvGr | `int64` | 0 | 0.000% | 1460 |
| KitchenAbvGr | `int64` | 0 | 0.000% | 1460 |
| KitchenQual | `object` | 0 | 0.000% | 1460 |
| TotRmsAbvGrd | `int64` | 0 | 0.000% | 1460 |
| Functional | `object` | 0 | 0.000% | 1460 |
| Fireplaces | `int64` | 0 | 0.000% | 1460 |
| FireplaceQu | `object` | 690 | 47.260% | 770 |
| GarageType | `object` | 81 | 5.548% | 1379 |
| GarageYrBlt | `float64` | 81 | 5.548% | 1379 |
| GarageFinish | `object` | 81 | 5.548% | 1379 |
| GarageCars | `int64` | 0 | 0.000% | 1460 |
| GarageArea | `int64` | 0 | 0.000% | 1460 |
| GarageQual | `object` | 81 | 5.548% | 1379 |
| GarageCond | `object` | 81 | 5.548% | 1379 |
| PavedDrive | `object` | 0 | 0.000% | 1460 |
| WoodDeckSF | `int64` | 0 | 0.000% | 1460 |
| OpenPorchSF | `int64` | 0 | 0.000% | 1460 |
| EnclosedPorch | `int64` | 0 | 0.000% | 1460 |
| 3SsnPorch | `int64` | 0 | 0.000% | 1460 |
| ScreenPorch | `int64` | 0 | 0.000% | 1460 |
| PoolArea | `int64` | 0 | 0.000% | 1460 |
| PoolQC | `object` | 1453 | 99.521% | 7 |
| Fence | `object` | 1179 | 80.753% | 281 |
| MiscFeature | `object` | 1406 | 96.301% | 54 |
| MiscVal | `int64` | 0 | 0.000% | 1460 |
| MoSold | `int64` | 0 | 0.000% | 1460 |
| YrSold | `int64` | 0 | 0.000% | 1460 |
| SaleType | `object` | 0 | 0.000% | 1460 |
| SaleCondition | `object` | 0 | 0.000% | 1460 |
| SalePrice | `int64` | 0 | 0.000% | 1460 |

---

## 5. Missing-value decisions

No rows or columns were dropped because of missingness.

### A. Structural categorical absence → `"None"`

The following fields describe optional house features. A missing value is interpreted as **feature absent**, not an unknown category:

- `Alley`
- `PoolQC`
- `MiscFeature`
- `Fence`
- `FireplaceQu`
- `MasVnrType`
- `GarageType`
- `GarageFinish`
- `GarageQual`
- `GarageCond`
- `BsmtQual`
- `BsmtCond`
- `BsmtExposure`
- `BsmtFinType1`
- `BsmtFinType2`

**Decision:** fill missing values with the explicit category `"NoFeature"`.

**Reason:** The dataset documentation uses missing values to represent the absence of the corresponding feature. The cleaned file uses the explicit category `NoFeature` rather than the literal string `None`; this avoids a common CSV-reader behavior in which `None` can be interpreted as a missing token. Keeping these as NaN would make a meaningful category look like a data defect; dropping rows would unnecessarily remove many valid houses.

### B. Structural numerical absence → `0`

The following numerical fields represent areas, counts, or capacities of optional structures:

- `MasVnrArea`
- `BsmtFinSF1`
- `BsmtFinSF2`
- `BsmtUnfSF`
- `TotalBsmtSF`
- `BsmtFullBath`
- `BsmtHalfBath`
- `GarageCars`
- `GarageArea`

**Decision:** fill missing values with `0`.

**Reason:** If the basement, masonry veneer, or garage is absent, its area/count/capacity is naturally zero. A mean/median would invent a feature that the house does not have.

### C. `GarageYrBlt` → `0` plus missingness indicator

**Raw missing:** 81 (5.548%).

**Decision:** create `GarageYrBlt_missing` (`1` if originally missing, otherwise `0`) and fill missing `GarageYrBlt` with `0`.

**Reason:** A missing garage year is normally caused by the absence of a garage. The explicit indicator preserves the original missingness information, while `0` gives the cleaned table a consistent numeric value. Downstream modeling should treat `0` as an absence code, not as an actual construction year.

### D. `LotFrontage` → neighborhood median plus missingness indicator

**Raw missing:** 259 (17.740%).

**Decision:** create `LotFrontage_missing`, then impute `LotFrontage` with the median `LotFrontage` within the same `Neighborhood`. If a neighborhood had no observed frontage values, use the overall median as a fallback.

**Reason:** Frontage is strongly tied to neighborhood layout, so a neighborhood-conditioned median is more defensible than one global mean. The missingness indicator preserves information about which observations required imputation.

### E. `Electrical` → mode

**Raw missing:** 1 (0.068%).

**Decision:** fill with the observed mode.

**Reason:** There is only one missing value, and `Electrical` is categorical. A mode is appropriate for this very small amount of missingness; a numerical statistic would be inappropriate.

### Result

After these decisions, the cleaned dataset contains **0 missing cells**. The cleaning process does not drop observations or predictors.

---

## 6. Duplicate and near-duplicate audit

### Exact duplicates

- Duplicate rows across all 81 original columns: **0**
- Duplicate rows after excluding `Id`: **0**
- Duplicate predictor records after excluding both `Id` and `SalePrice`: **0**

Therefore, there is no evidence of exact duplicate records requiring removal.

### Near-duplicate screening

A conservative screening rule was used:

> Exclude `Id` and `SalePrice`, then flag pairs matching exactly on at least **77 of the 79 predictors** (97.47% or more).

Four candidate pairs were found:

| Id A | Id B | Matching predictors | Differing fields | Decision |
|---:|---:|---:|---|---|
| 1442 | 691 | 78/79 (98.73%) | BsmtExposure | Retain; differences are plausible feature-level differences, not duplicate records. |
| 1089 | 194 | 78/79 (98.73%) | MoSold | Retain; differences are plausible feature-level differences, not duplicate records. |
| 1369 | 594 | 77/79 (97.47%) | YearRemodAdd, MoSold | Retain; differences are plausible feature-level differences, not duplicate records. |
| 146 | 1089 | 77/79 (97.47%) | OverallQual, YearRemodAdd | Retain; differences are plausible feature-level differences, not duplicate records. |

These pairs were **retained**. They are not exact duplicates, and the differing fields are legitimate house attributes such as basement exposure, month sold, remodeling year, or overall quality. Similar houses are expected in a housing dataset, so similarity alone is not evidence of accidental duplication.

---

## 7. Numerical outlier audit

The IQR rule was applied independently to three meaningful numerical predictors:

> Lower bound = Q1 − 1.5 × IQR  
> Upper bound = Q3 + 1.5 × IQR

| Column | Q1 | Q3 | IQR | Lower bound | Upper bound | Flagged | Maximum |
|---|---:|---:|---:|---:|---:|---:|---:|
| `GrLivArea` | 1129.50 | 1776.75 | 647.25 | 158.62 | 2747.62 | 31 | 5642 |
| `LotArea` | 7553.50 | 11601.50 | 4048.00 | 1481.50 | 17673.50 | 69 | 215245 |
| `TotalBsmtSF` | 795.75 | 1298.25 | 502.50 | 42.00 | 2052.00 | 61 | 6110 |

### 7.1 `GrLivArea`

**31 observations** are above the IQR upper bound of 2,747.63 sq ft.

Large living areas are plausible in real houses, so the existence of an IQR outlier does **not** prove a data-entry error. The most extreme observation (`GrLivArea = 5,642`) is unusual enough to warrant manual/source verification, but this Task 1 cleaning does not have sufficient target-independent evidence to delete it.

**Decision:** retain all observations and create `GrLivArea_outlier`.

**Reason:** an outlier detector identifies unusual observations; it does not establish that the values are wrong.

### 7.2 `LotArea`

**69 observations** are above the IQR upper bound of 17,673.50 sq ft, with a maximum of 215,245 sq ft.

Large lots are plausible in residential data and can represent genuine properties rather than measurement errors. Some extreme lots are therefore expected to be legitimate observations.

**Decision:** retain all observations and create `LotArea_outlier`.

**Reason:** no target-independent evidence in the raw table proves that these values are erroneous.

### 7.3 `TotalBsmtSF`

**61 observations** are above the IQR upper bound of 2,052.00 sq ft, with a maximum of 6,110 sq ft.

Large basements can be genuine, particularly for large houses. The extreme value is unusual but internally compatible with the existence of a very large property; without an external source or construction record, deletion would be speculative.

**Decision:** retain all observations and create `TotalBsmtSF_outlier`.

**Reason:** flagging preserves information while making the unusual observations available for later sensitivity analysis.

### Overall outlier policy

**No outlier rows were deleted.** The cleaned dataset contains explicit binary outlier flags instead.

This is preferable at the data-quality stage because an IQR rule answers “is this observation statistically unusual?” rather than “is this observation factually wrong?”

---

## 8. Data-type corrections

### `MSSubClass`

`MSSubClass` is stored as an integer in the raw CSV, but its values are **codes for building class**, not a continuous measurement.

**Decision:** cast `MSSubClass` to a Pandas `category` in `clean_data.py`.

**Reason:** Treating class codes such as 20, 30, 60, etc. as continuous numbers would imply meaningful numeric distances between categories.

The cleaned CSV itself is still a CSV, so a CSV reader may infer `MSSubClass` as numeric again. The reproducible script explicitly restores the categorical dtype after loading.

### Other numeric columns

Ordinal ratings such as `OverallQual` and `OverallCond` remain numeric because their ordered values carry meaningful ranking information. Calendar fields such as `YearBuilt` and `YearRemodAdd` remain numeric because they represent years.

### `Id`

`Id` is retained as an identifier for traceability and duplicate investigation. It is **not treated as a substantive house feature** in the cleaning logic.

---

## 9. Target-variable protection

`SalePrice` exists in the raw file and remains unchanged.

No missing-value rule, outlier rule, type correction, or duplicate decision uses `SalePrice`.

This is deliberate: using the target to decide which predictor values are “bad” would introduce target information into preprocessing and can create leakage.

---

## 10. Leakage risk for later modeling

This Task 1 exercise uses the full uploaded training file because the assignment explicitly says train/test splitting begins in Week 2.

For actual model development, the following rule must be followed:

1. Split the data first.
2. Fit imputation statistics (median/mode/group statistics) on the training split only.
3. Apply those fitted statistics to validation/test data.
4. Fit encoders/scalers on training data only.
5. Never use `SalePrice` to determine predictor preprocessing.

The current report therefore documents the cleaning logic without pretending that the full-dataset medians are suitable for a future leakage-safe modeling pipeline.

---

## 11. Final cleaned dataset

The cleaned file contains:

- **1,460 rows**
- **86 columns**
- Original 81 columns preserved
- 2 missingness indicators:
  - `LotFrontage_missing`
  - `GarageYrBlt_missing`
- 3 IQR outlier flags:
  - `GrLivArea_outlier`
  - `LotArea_outlier`
  - `TotalBsmtSF_outlier`
- **0 missing cells**
- **0 rows dropped**
- **0 columns dropped**
- `SalePrice` retained unchanged
- No scaling
- No one-hot encoding

The additional flags are intentionally simple audit features; they do not replace the original measurements.

---

## 12. Files produced

- `ames_cleaned.csv` — cleaned dataset
- `clean_data.py` — reproducible cleaning script
- `MISSING_VALUE_REPORT.csv` — machine-readable full missing-value audit
- `DATA_QUALITY_REPORT.md` — this documented audit

## 13. Reproducibility

Run:

```bash
python clean_data.py train.csv ames_cleaned.csv
```

The script also writes `MISSING_VALUE_REPORT.csv` beside the requested output CSV.

## 14. Assignment checklist

| Requirement | Status |
|---|---|
| Full missing-value count and percentage per column | Complete |
| At least 5 explicit missing-value decisions | Complete; 19 affected columns documented |
| No blanket mean imputation | Complete |
| Duplicate-row check | Complete |
| Near-duplicate screening | Complete |
| At least 3 numerical outlier columns | Complete; 3 IQR-based flags |
| Outlier error-vs-extreme reasoning | Complete |
| Correct categorical/numerical typing | Complete; `MSSubClass` treated as categorical |
| `SalePrice` protected from cleaning decisions | Complete |
| No scaling/encoding | Complete |
| Leakage risk documented | Complete |
| Reproducible Python script | Complete |
