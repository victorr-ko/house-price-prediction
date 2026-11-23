# House Price Prediction (Ames, Iowa)

Predicting residential sale prices from 79 property features.

> **Capstone project** for the **Diploma in Data Science** at [AltSchool Africa](https://altschoolafrica.com/).

---

## Results at a glance

The final model, a tuned **Gradient Boosting Regressor**, was evaluated on a held-out test set of 292 houses:

| Metric | Test score |
|---|---|
| **R²** | **0.9033** (explains about 90% of the variation in sale prices) |
| **RMSE** | $27,231 |
| **MAE** | $16,702 (about 10% of the median sale price of $163,000) |

**Most influential features:** `OverallQual` (overall material and finish) and `GrLivArea` (above-ground living area) together account for about 65% of the model's feature importance.

---

## Problem statement

This project builds a machine learning model that predicts the sale price of residential homes in Ames, Iowa from 79 property features, and identifies which features drive price the most. Performance is evaluated on unseen houses using RMSE, MAE and R².

## Dataset

- **Source:** [Kaggle: House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) (the Ames Housing dataset compiled by Dean De Cock)
- **Size:** 1,460 houses, 81 columns (79 features, plus `Id` and the target `SalePrice`)
- **Target:** `SalePrice`, ranging from $34,900 to $755,000 (right-skewed, median $163,000)
- **Feature dictionary:** [`data_description.txt`](data_description.txt)

## Project workflow

1. **Data loading and inspection:** checked data types, missing values (19 columns, for example `PoolQC`, `MiscFeature`, `Alley`) and duplicates (none).
2. **Exploratory data analysis:** distributions of all numeric features and of `SalePrice`; relationships between price and `GrLivArea`, `OverallQual`, `YearBuilt`, `MSSubClass` and `MSZoning`.
3. **Outlier analysis (IQR method):** 61 outliers in `SalePrice` and 31 in `GrLivArea`, all on the high side. Almost all were genuine high-quality homes, so they were kept.
4. **Feature engineering:**
   - `HouseAge`: year sold minus year built
   - `Remodeled`: 1 if the house was remodelled, 0 otherwise
   - `SeasonSold`: season of sale, derived from the month
   - `HasPool`: 1 if the house has a pool
   - `MSSubClass` converted from integer codes to a categorical column
5. **Train/test split (80/20)** done *before* any preprocessing, to prevent data leakage.
6. **Preprocessing pipeline** (`ColumnTransformer` inside a `Pipeline`):
   - Numeric columns: median imputation, then standard scaling
   - Categorical columns: missing values filled with `'None'` (a missing value usually means the feature is absent), then one-hot encoding with `handle_unknown='ignore'`
7. **Model comparison:** Linear Regression, Decision Tree, Random Forest and Gradient Boosting.
8. **Hyperparameter tuning:** `GridSearchCV` over 32 combinations with 5-fold cross-validation (160 fits), with the full pipeline inside the search so preprocessing is refitted on each fold.
9. **Evaluation and interpretation:** test-set metrics, feature importances and partial dependence plots.

## Model comparison

| Model | Test RMSE | Test MAE | Test R² |
|---|---|---|---|
| **Gradient Boosting (tuned)** | **27,231** | **16,702** | **0.9033** |
| Gradient Boosting (default) | 27,267 | 17,087 | 0.9031 |
| Random Forest | 28,897 | 17,643 | 0.8911 |
| Decision Tree | 43,839 | 27,141 | 0.7494 |
| Linear Regression | 70,142 | 21,567 | 0.3586 |

**Observations**

- The **Decision Tree** overfit badly: perfect training score (R² = 1.0) but a much weaker test score.
- **Linear Regression's** RMSE was driven by one extreme miss (one house was predicted at about -$927,000), which shows its sensitivity to rare combinations of one-hot features.
- **Tuning** improved Gradient Boosting only slightly. The cross-validated RMSE (about $27,857) is close to the test RMSE, which suggests the result is not a lucky split.

## Key insights

- **Overall quality matters most.** In the partial dependence plot, predicted price rises from about $162,000 at quality 5 to about $233,000 at quality 9.
- **Living area comes second.** Predicted price rises by about $56,000 between roughly 900 and 2,450 sq ft.
- **Basement size, garage capacity and year built** have positive but much smaller effects once quality and size are accounted for.
- The engineered features (`HouseAge`, `Remodeled`, `SeasonSold`, `HasPool`) did not appear in the top 10 most important features.

## Limitations and next steps

- Some overfitting remains (train R² 0.98 compared with test R² 0.90).
- The best hyperparameters sit at the edge of the search grid, so a wider search could help.
- Possible improvements: log-transforming the right-skewed `SalePrice`, trying XGBoost or LightGBM, and adding features such as total square footage.

---

## Tools and libraries

Python 3.11, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Jupyter