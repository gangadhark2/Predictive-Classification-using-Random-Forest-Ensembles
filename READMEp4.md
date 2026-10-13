# Predictive Classification using Random Forest Ensembles

Minor project applying a **Random Forest** classifier to the **Forest Cover Type** dataset (Roosevelt National Forest, Colorado), with a full workflow: EDA, preprocessing, model development and evaluation.

## Problem statement

Predict whether a 30 m x 30 m forest cell is **Cottonwood/Willow** (`Cover_Type` 4) or any other cover type, using only cartographic variables (elevation, slope, distances to water/roads/fire points, hillshade, wilderness area and soil type).

Cottonwood/Willow is a very small share of the data (well under 1%), so this is a **highly imbalanced** binary problem, similar to fraud detection. Accuracy alone is misleading, so the project is evaluated with precision, recall, F1 and ROC-AUC.

## Dataset

- **Name:** Forest Cover Type (UCI / Kaggle "Forest Cover Type Prediction")
- **Size:** 581,012 rows, 54 features, 7 cover types
- **Features:** 10 continuous cartographic variables, 4 wilderness-area binary columns, 40 soil-type binary columns
- **Loading:** the first notebook downloads it with `sklearn.datasets.fetch_covtype` and saves it as `forest_cover.csv`. You can also place the Kaggle CSV in the folder under that name (the column names must match).

## Project files

| File | Purpose |
|---|---|
| `Forest_Cover_EDA.ipynb` | Phase I: exploratory data analysis |
| `Forest_Cover_Preprocessing.ipynb` | Phase II: cleaning, encoding, train/test split, scaling decision |
| `Forest_Cover_Model_Development.ipynb` | Phase III: hyperparameter justification, final model, all five required metrics |
| `forest_cover_rf_project.py` | Optional all-in-one script that runs the whole pipeline and writes the Word report automatically |
| `README.md` | This file |

## How to run

1. Open **Google Colab** (or local Jupyter) and upload the notebooks.
2. Run the notebooks **in this order**, all in the same working folder:
   1. `Forest_Cover_EDA.ipynb`
   2. `Forest_Cover_Preprocessing.ipynb`
   3. `Forest_Cover_Model_Development.ipynb`
3. In each notebook choose **Runtime > Run all** (Colab) or **Run All Cells** (Jupyter).

Requirements: Python 3.9+, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`. The all-in-one script also needs `python-docx`.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn python-docx
```

An internet connection is needed the first time to download the dataset. Full model tuning takes roughly 10 minutes on a standard Colab machine.

### Alternative: one command

```bash
python forest_cover_rf_project.py
```

This produces all figures and `Forest_Cover_RF_Technical_Report.docx` in an `outputs/` folder.

## Workflow summary

**Phase I: EDA**
- Structure, missing values, duplicates, summary statistics
- Class imbalance of the target
- Feature distributions, skewness and outliers (IQR rule)
- Correlation heatmap
- Elevation, wilderness area and soil type against the target

**Phase II: Preprocessing**

| Step | Decision | Reason |
|---|---|---|
| Missing values | Median imputation as a safeguard | The dataset normally has none |
| Outliers / duplicates | Kept | Valid terrain extremes and repeated measurements |
| Encoding | One-hot for wilderness area; soil columns already one-hot | Nominal data, so no false ordering |
| Leakage control | `Cover_Type` removed from the features | The target is derived from it |
| Split | Stratified 80:20, `random_state=42` | The rare class must appear in both sets |
| Scaling | Skipped, verified by comparing F1 on raw vs scaled features | Random Forest is threshold-based, not distance-based |

**Phase III: Model development**
- `n_estimators` and `max_depth` chosen by 3-fold stratified cross-validation of minority-class F1 on the training data only, using a parsimony rule (smallest value within 0.01 of the best)
- `class_weight='balanced_subsample'` to handle the imbalance
- Final model trained on the full training set and evaluated once on the held-out test set
- Comparison with `HistGradientBoostingClassifier`

## Evaluation metrics (all required by the brief)

| Metric | Deliverable |
|---|---|
| Confusion matrix | Heatmap of TP, TN, FP, FN (counts and row-normalised) |
| Precision and recall | Interpretation, precision-recall curve and threshold sweep |
| F1-score | Value plus full classification report |
| ROC-AUC | ROC curve plot and AUC score |
| Feature importance | Bar chart of the top 15 features |

## Outputs generated when you run the notebooks

```
forest_cover.csv            dataset
X_train.csv  X_test.csv     preprocessed features
y_train.csv  y_test.csv     target labels
final_results.csv           all final metrics and chosen hyperparameters
eda_figures/                EDA plots (300 dpi)
prep_figures/               preprocessing plots (300 dpi)
model_figures/              tuning and evaluation plots (300 dpi)
```

## Results

Results are produced when you run the notebooks and are saved in `final_results.csv`. Add your final numbers here after running:

| Metric | Value |
|---|---|
| n_estimators | |
| max_depth | |
| Precision | |
| Recall | |
| F1-score | |
| ROC-AUC | |

## Limitations

- Neighbouring cells are spatially similar, so a random split can overestimate performance on a completely new region. Spatially blocked validation would be a stricter test.
- Impurity-based feature importance favours continuous variables and splits credit between correlated features.
- The decision threshold defaults to 0.5. It should be tuned on a validation set according to the relative cost of false negatives and false positives.

## Possible improvements

- Wider hyperparameter search (`RandomizedSearchCV` over `min_samples_leaf`, `max_features`)
- Gradient boosting (XGBoost / LightGBM)
- Resampling such as SMOTE or under-sampling
- Permutation importance or SHAP for more reliable interpretation

## Tools

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, python-docx.
