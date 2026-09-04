# GROUP 6 — Pima Indians Diabetes Complete ML Solution
# Run this script from the folder containing pima_Missing_values.csv.

# If necessary, install packages in your environment:
# %pip install pandas numpy matplotlib seaborn scikit-learn joblib

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("Libraries imported successfully.")


# Expected local file. Change this path if your downloaded filename is different.
DATA_PATH = "pima_Missing_values.csv"

if not os.path.exists(DATA_PATH):
    # Also check common alternative names.
    candidates = [
        "pima-indians-diabetes.csv",
        "diabetes.csv",
        "pima.csv",
        "Pima_Missing_values.csv"
    ]
    found = next((f for f in candidates if os.path.exists(f)), None)
    if found:
        DATA_PATH = found

print("Dataset path:", DATA_PATH)
print("Exists:", os.path.exists(DATA_PATH))

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "Download the assigned Kaggle CSV and place it beside this notebook, "
        "then set DATA_PATH to the exact filename."
    )


raw_df = pd.read_csv(DATA_PATH)

# Preserve an in-memory copy of the raw dataset.
raw_df_original = raw_df.copy(deep=True)

print("Raw shape:", raw_df.shape)
display(raw_df.head())


EXPECTED_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

# If exactly 9 columns are present and names do not match the expected semantic names,
# assign the standard Pima column names.
if raw_df.shape[1] == 9:
    normalized = [str(c).strip().lower().replace("_", "").replace(" ", "") for c in raw_df.columns]
    expected_norm = [c.lower().replace("_", "").replace(" ", "") for c in EXPECTED_COLUMNS]

    if normalized != expected_norm:
        raw_df.columns = EXPECTED_COLUMNS

# Normalize common alternative spellings.
rename_map = {}
for c in raw_df.columns:
    key = str(c).strip().lower().replace("_", "").replace(" ", "")
    aliases = {
        "preg": "Pregnancies",
        "pregnant": "Pregnancies",
        "pregnancies": "Pregnancies",
        "plas": "Glucose",
        "gluc": "Glucose",
        "glucose": "Glucose",
        "pres": "BloodPressure",
        "bp": "BloodPressure",
        "bloodpressure": "BloodPressure",
        "skin": "SkinThickness",
        "triceps": "SkinThickness",
        "skinthickness": "SkinThickness",
        "insu": "Insulin",
        "insulin": "Insulin",
        "mass": "BMI",
        "bmi": "BMI",
        "pedi": "DiabetesPedigreeFunction",
        "pedigree": "DiabetesPedigreeFunction",
        "diabetespedigreefunction": "DiabetesPedigreeFunction",
        "age": "Age",
        "class": "Outcome",
        "outcome": "Outcome",
        "diabetes": "Outcome"
    }
    if key in aliases:
        rename_map[c] = aliases[key]

raw_df = raw_df.rename(columns=rename_map)

missing_expected = [c for c in EXPECTED_COLUMNS if c not in raw_df.columns]
if missing_expected:
    raise ValueError(f"Missing expected columns: {missing_expected}. Found: {list(raw_df.columns)}")

raw_df = raw_df[EXPECTED_COLUMNS].copy()

# Force numeric values where possible.
for col in EXPECTED_COLUMNS:
    raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

print(raw_df.dtypes)
display(raw_df.head())


print("Dimensions:", raw_df.shape)
print("\nColumns:")
print(raw_df.columns.tolist())

print("\nData types:")
print(raw_df.dtypes)

print("\nDescriptive statistics:")
display(raw_df.describe().T)

print("\nExplicit missing values:")
missing_count = raw_df.isna().sum()
missing_pct = raw_df.isna().mean().mul(100).round(2)
missing_table = pd.DataFrame({
    "Missing_Count": missing_count,
    "Missing_Percentage": missing_pct
})
display(missing_table)


zero_sensitive_cols = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
]

zero_table = pd.DataFrame({
    "Zero_Count": [int((raw_df[c] == 0).sum()) for c in EXPECTED_COLUMNS],
    "Zero_Percentage": [round((raw_df[c] == 0).mean() * 100, 2) for c in EXPECTED_COLUMNS]
}, index=EXPECTED_COLUMNS)

display(zero_table)

# Create a cleaned working copy. Never modify the preserved raw copy.
df = raw_df.copy(deep=True)

for col in zero_sensitive_cols:
    df[col] = df[col].replace(0, np.nan)

print("Missing values after converting physiologically impossible zeros to NaN:")
display(pd.DataFrame({
    "Missing_Count": df.isna().sum(),
    "Missing_Percentage": df.isna().mean().mul(100).round(2)
}))


duplicate_count = int(df.duplicated().sum())
print("Duplicate rows:", duplicate_count)

# Display duplicate rows if any.
if duplicate_count:
    display(df[df.duplicated(keep=False)].sort_values(EXPECTED_COLUMNS))

# IQR-based outlier screening.
feature_cols = EXPECTED_COLUMNS[:-1]
outlier_summary = []

for col in feature_cols:
    s = df[col].dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((s < lower) | (s > upper)).sum())
    outlier_summary.append([col, q1, q3, iqr, lower, upper, count])

outlier_table = pd.DataFrame(
    outlier_summary,
    columns=["Feature", "Q1", "Q3", "IQR", "Lower_Bound", "Upper_Bound", "Outlier_Count"]
)
display(outlier_table)

plt.figure(figsize=(12, 7))
sns.boxplot(data=df[feature_cols])
plt.xticks(rotation=45)
plt.title("Outlier Screening with Boxplots")
plt.tight_layout()
plt.show()


class_counts = df["Outcome"].value_counts().sort_index()
class_pct = df["Outcome"].value_counts(normalize=True).sort_index().mul(100).round(2)

class_table = pd.DataFrame({
    "Count": class_counts,
    "Percentage": class_pct
})
display(class_table)

plt.figure(figsize=(6, 4))
sns.countplot(x="Outcome", data=df)
plt.title("Target Class Distribution")
plt.xlabel("Outcome (0 = No diabetes, 1 = Diabetes)")
plt.ylabel("Number of patients")
plt.show()


# Histograms
df[feature_cols].hist(figsize=(14, 10), bins=20)
plt.suptitle("Feature Distributions", y=1.02)
plt.tight_layout()
plt.show()

# Correlation heatmap after the zero-to-NaN conversion.
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# Outcome relationship for major predictors.
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
sns.boxplot(data=df, x="Outcome", y="Glucose", ax=axes[0, 0])
sns.boxplot(data=df, x="Outcome", y="BMI", ax=axes[0, 1])
sns.boxplot(data=df, x="Outcome", y="Age", ax=axes[1, 0])
sns.boxplot(data=df, x="Outcome", y="DiabetesPedigreeFunction", ax=axes[1, 1])
plt.tight_layout()
plt.show()


X = df[feature_cols].copy()
y = df["Outcome"].astype(int).copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("\nTraining class distribution:")
print(y_train.value_counts(normalize=True).sort_index().round(3))
print("\nTest class distribution:")
print(y_test.value_counts(normalize=True).sort_index().round(3))


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

def make_pipeline(model, imputer="median", scale=True, select_features=False):
    if imputer == "median":
        imputer_step = SimpleImputer(strategy="median")
    elif imputer == "mean":
        imputer_step = SimpleImputer(strategy="mean")
    elif imputer == "knn":
        imputer_step = KNNImputer(n_neighbors=5)
    else:
        raise ValueError("Unknown imputer")

    steps = [("imputer", imputer_step)]
    if scale:
        steps.append(("scaler", StandardScaler()))
    if select_features:
        steps.append(("selector", SelectKBest(score_func=mutual_info_classif, k=6)))
    steps.append(("model", model))
    return Pipeline(steps)

models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
        True
    ),
    "KNN": (
        KNeighborsClassifier(n_neighbors=15, weights="distance"),
        True
    ),
    "Random Forest": (
        RandomForestClassifier(
            n_estimators=400, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        False
    ),
    "SVM": (
        SVC(
            kernel="rbf", probability=True,
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        True
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        False
    )
}

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}


baseline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE))
])

baseline_scores = cross_validate(
    baseline, X_train, y_train,
    cv=cv, scoring=scoring, n_jobs=-1
)

baseline_result = {
    metric: baseline_scores[f"test_{metric}"].mean()
    for metric in scoring
}
print("Baseline CV performance:")
display(pd.DataFrame([baseline_result], index=["Dummy Baseline"]).round(4))


comparison_rows = []

for name, (model, scale) in models.items():
    pipe = make_pipeline(model, imputer="median", scale=scale)
    scores = cross_validate(
        pipe, X_train, y_train,
        cv=cv, scoring=scoring, n_jobs=-1
    )
    comparison_rows.append({
        "Model": name,
        **{metric: scores[f"test_{metric}"].mean() for metric in scoring},
        **{f"{metric}_std": scores[f"test_{metric}"].std() for metric in scoring}
    })

comparison_df = pd.DataFrame(comparison_rows).sort_values(
    "roc_auc", ascending=False
).reset_index(drop=True)

display(comparison_df.round(4))


strategy_rows = []

strategy_models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
        True
    ),
    "Random Forest": (
        RandomForestClassifier(
            n_estimators=400, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        False
    )
}

for strategy in ["median", "knn"]:
    for name, (model, scale) in strategy_models.items():
        pipe = make_pipeline(model, imputer=strategy, scale=scale)
        scores = cross_validate(
            pipe, X_train, y_train,
            cv=cv, scoring=scoring, n_jobs=-1
        )
        strategy_rows.append({
            "Imputation": strategy,
            "Model": name,
            **{metric: scores[f"test_{metric}"].mean() for metric in scoring}
        })

strategy_df = pd.DataFrame(strategy_rows).sort_values(
    ["roc_auc", "f1"], ascending=False
).reset_index(drop=True)

display(strategy_df.round(4))


param_grids = {
    "Logistic Regression": (
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
        True,
        {
            "model__C": [0.01, 0.1, 1, 10, 100],
            "model__solver": ["liblinear", "lbfgs"]
        }
    ),
    "Random Forest": (
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        False,
        {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 4, 8, 12],
            "model__min_samples_leaf": [1, 2, 5],
            "model__max_features": ["sqrt", "log2"]
        }
    ),
    "SVM": (
        SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        True,
        {
            "model__C": [0.1, 1, 10],
            "model__gamma": ["scale", 0.01, 0.1],
            "model__kernel": ["rbf"]
        }
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        False,
        {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [1, 2, 3],
            "model__subsample": [0.8, 1.0]
        }
    )
}

tuned = {}
tuning_rows = []

for name, (model, scale, grid) in param_grids.items():
    pipe = make_pipeline(model, imputer="median", scale=scale)
    search = GridSearchCV(
        pipe, grid, cv=cv, scoring="roc_auc",
        n_jobs=-1, refit=True
    )
    search.fit(X_train, y_train)
    tuned[name] = search

    tuning_rows.append({
        "Model": name,
        "Best_CV_ROC_AUC": search.best_score_,
        "Best_Params": search.best_params_
    })

tuning_df = pd.DataFrame(tuning_rows).sort_values(
    "Best_CV_ROC_AUC", ascending=False
).reset_index(drop=True)

display(tuning_df.round(4))


best_name = tuning_df.iloc[0]["Model"]
best_search = tuned[best_name]
final_model = best_search.best_estimator_

print("Selected final model:", best_name)
print("Best CV ROC-AUC:", round(best_search.best_score_, 4))
print("Best parameters:", best_search.best_params_)


final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)
y_prob = final_model.predict_proba(X_test)[:, 1]

final_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1-score": f1_score(y_test, y_pred, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, y_prob)
}

final_metrics_df = pd.DataFrame(final_metrics, index=["Test Score"]).T
display(final_metrics_df.round(4))

print("\nClassification report:")
print(classification_report(
    y_test, y_pred,
    target_names=["No diabetes", "Diabetes"],
    zero_division=0
))


cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No diabetes", "Diabetes"]
)
disp.plot()
plt.title(f"Confusion Matrix — {best_name}")
plt.show()

RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title(f"ROC Curve — {best_name}")
plt.show()

print("Confusion matrix:")
print(cm)


error_analysis = X_test.copy()
error_analysis["Actual"] = y_test.values
error_analysis["Predicted"] = y_pred
error_analysis["Probability_Diabetes"] = y_prob

error_analysis["Error_Type"] = np.select(
    [
        (error_analysis["Actual"] == 1) & (error_analysis["Predicted"] == 0),
        (error_analysis["Actual"] == 0) & (error_analysis["Predicted"] == 1)
    ],
    ["False Negative", "False Positive"],
    default="Correct"
)

print("Error counts:")
display(error_analysis["Error_Type"].value_counts())

print("\nFalse negatives:")
display(error_analysis[error_analysis["Error_Type"] == "False Negative"])

print("\nFalse positives:")
display(error_analysis[error_analysis["Error_Type"] == "False Positive"])

print("\nMean feature values by error type:")
display(error_analysis.groupby("Error_Type")[feature_cols].mean().round(2))


model_step = final_model.named_steps["model"]

if hasattr(model_step, "coef_"):
    coefficients = pd.Series(
        model_step.coef_[0],
        index=feature_cols
    ).sort_values(key=np.abs, ascending=False)

    print("Logistic-regression coefficients:")
    display(coefficients.to_frame("Coefficient"))

    coefficients.sort_values().plot(kind="barh", figsize=(8, 5))
    plt.title("Feature Coefficients")
    plt.tight_layout()
    plt.show()

elif hasattr(model_step, "feature_importances_"):
    importances = pd.Series(
        model_step.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    print("Tree-based feature importance:")
    display(importances.to_frame("Importance"))

    importances.sort_values().plot(kind="barh", figsize=(8, 5))
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

else:
    from sklearn.inspection import permutation_importance

    perm = permutation_importance(
        final_model, X_test, y_test,
        scoring="roc_auc", n_repeats=20,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    importances = pd.Series(
        perm.importances_mean,
        index=feature_cols
    ).sort_values(ascending=False)

    display(importances.to_frame("Permutation_Importance"))


threshold_rows = []

for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    pred_t = (y_prob >= threshold).astype(int)
    threshold_rows.append({
        "Threshold": threshold,
        "Precision": precision_score(y_test, pred_t, zero_division=0),
        "Recall": recall_score(y_test, pred_t, zero_division=0),
        "F1": f1_score(y_test, pred_t, zero_division=0),
        "Accuracy": accuracy_score(y_test, pred_t)
    })

threshold_df = pd.DataFrame(threshold_rows)
display(threshold_df.round(4))


MODEL_PATH = "pima_diabetes_final_model.joblib"
joblib.dump(final_model, MODEL_PATH)

print("Saved:", MODEL_PATH)
print("File exists:", os.path.exists(MODEL_PATH))


# Reload to demonstrate deployment.
loaded_model = joblib.load(MODEL_PATH)

def predict_diabetes(
    pregnancies, glucose, blood_pressure,
    skin_thickness, insulin, bmi,
    diabetes_pedigree_function, age
):
    patient = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree_function,
        "Age": age
    }])

    # Treat physiologically impossible zeros as missing, just as during training.
    for col in zero_sensitive_cols:
        patient[col] = patient[col].replace(0, np.nan)

    probability = float(loaded_model.predict_proba(patient)[:, 1][0])
    prediction = int(probability >= 0.50)

    return {
        "Prediction": prediction,
        "Class": "Diabetes" if prediction == 1 else "No diabetes",
        "Probability_of_diabetes": round(probability, 4)
    }

# Example:
example_patient = predict_diabetes(
    pregnancies=6,
    glucose=148,
    blood_pressure=72,
    skin_thickness=35,
    insulin=155,
    bmi=33.6,
    diabetes_pedigree_function=0.627,
    age=50
)

example_patient


