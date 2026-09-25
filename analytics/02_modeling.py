import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score
)
from imblearn.over_sampling import SMOTE

# -----------------------------------------------------------------------------
# 1. LOAD FROM COMMITTED FALLBACK & STRATIFIED SPLIT
# -----------------------------------------------------------------------------
df = pd.read_csv("titanic.csv")

# Feature definition: keep core informative attributes; exclude targets and leakage flags
feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
X = df[feature_cols].copy()
y = df["survived"].copy()

# Stratification prevents distributional divergence in target balance (~61.6% dead / 38.4% survived)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Dataset Split Completed. Train: {X_train.shape[0]} rows, Test: {X_test.shape[0]} rows")

# -----------------------------------------------------------------------------
# 2. COLUMN TRANSFORMER (PREPROCESSING FIT ONLY ON TRAIN)
# -----------------------------------------------------------------------------
numeric_features = ["pclass", "age", "sibsp", "parch", "fare"]
categorical_features = ["sex", "embarked"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# -----------------------------------------------------------------------------
# 3. CLASSIFIER BENCHMARK (LR, DECISION TREE, RANDOM FOREST)
# -----------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=3, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
}

results = []
trained_pipelines = {}

for name, clf in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)
    trained_pipelines[name] = pipe

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1 Score": round(f1, 4),
        "ROC-AUC": round(auc, 4),
        "Confusion Matrix": cm.tolist()
    })

df_clf_results = pd.DataFrame(results)
print("\n=== CLASSIFICATION BENCHMARK ===")
print(df_clf_results.to_string(index=False))

# Render Decision Tree with feature and class names
dt_pipeline = trained_pipelines["Decision Tree"]
fitted_tree = dt_pipeline.named_steps["classifier"]
ohe_feature_names = dt_pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(categorical_features).tolist()
all_feature_names = numeric_features + ohe_feature_names

plt.figure(figsize=(16, 9))
plot_tree(
    fitted_tree,
    feature_names=all_feature_names,
    class_names=["Died", "Survived"],
    filled=True,
    rounded=True,
    fontsize=9
)
plt.title("Decision Tree Visualization (Max Depth = 3)")
plt.savefig("plots/04_decision_tree.png")
plt.close()

# -----------------------------------------------------------------------------
# 4. IMBALANCE STRATEGY COMPARISON (ON RANDOM FOREST)
# -----------------------------------------------------------------------------
# A: Baseline (already executed)
rf_baseline_f1 = df_clf_results.loc[df_clf_results["Model"] == "Random Forest", "F1 Score"].values[0]

# B: Balanced Class Weight
pipe_balanced = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42))
])
pipe_balanced.fit(X_train, y_train)
y_pred_bal = pipe_balanced.predict(X_test)
rf_bal_f1 = f1_score(y_test, y_pred_bal)

# C: SMOTE (Fit on transformed training features only to prevent data leakage)
X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_trans, y_train)

rf_smote = RandomForestClassifier(n_estimators=100, random_state=42)
rf_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = rf_smote.predict(X_test_trans)
rf_smote_f1 = f1_score(y_test, y_pred_smote)

print("\n=== CLASS IMBALANCE BENCHMARK (RANDOM FOREST) ===")
print(f"1. Baseline RF F1:           {rf_baseline_f1:.4f}")
print(f"2. Balanced Weights RF F1:   {rf_bal_f1:.4f}")
print(f"3. SMOTE Resampled RF F1:    {rf_smote_f1:.4f}")

# -----------------------------------------------------------------------------
# 5. HYPERPARAMETER TUNING & OUT-OF-BAG (OOB) SCORE
# -----------------------------------------------------------------------------
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [4, 6, 8],
    "classifier__max_features": ["sqrt", "log2"]
}

rf_base_estimator = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(oob_score=True, random_state=42))
])

grid_search = GridSearchCV(
    estimator=rf_base_estimator,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

best_pipeline = grid_search.best_estimator_
best_oob = best_pipeline.named_steps["classifier"].oob_score_
print("\n=== HYPERPARAMETER TUNING RESULTS ===")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Out-of-Bag (OOB) Score: {best_oob:.4f}")

# -----------------------------------------------------------------------------
# 6. REGRESSION SIDE-TASK: FARE PREDICTION & HETEROSCEDASTICITY CHECK
# -----------------------------------------------------------------------------
reg_features = ["pclass", "sex", "age", "sibsp", "parch", "embarked"]
X_reg = df[reg_features].copy()
y_reg = df["fare"].copy()

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.20, random_state=42
)

reg_num_cols = ["pclass", "age", "sibsp", "parch"]
reg_cat_cols = ["sex", "embarked"]

reg_preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline(steps=[("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), reg_num_cols),
    ("cat", Pipeline(steps=[("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore"))]), reg_cat_cols)
])

reg_pipeline = Pipeline(steps=[
    ("preprocessor", reg_preprocessor),
    ("regressor", LinearRegression())
])
reg_pipeline.fit(X_train_r, y_train_r)

y_pred_r = reg_pipeline.predict(X_test_r)
residuals = y_test_r - y_pred_r

mae = mean_absolute_error(y_test_r, y_pred_r)
rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
r2 = r2_score(y_test_r, y_pred_r)
n = len(y_test_r)
p = X_train_r.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("\n=== REGRESSION METRICS (PREDICTING FARE) ===")
print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f} | Adjusted R2: {adj_r2:.4f}")

# Residual Plot
plt.figure(figsize=(8, 5))
plt.scatter(y_pred_r, residuals, alpha=0.5, color="purple")
plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
plt.xlabel("Predicted Fare ($)")
plt.ylabel("Residuals ($)")
plt.title("Residuals vs. Fitted Values (Fare Regression)")
plt.tight_layout()
plt.savefig("plots/05_fare_regression_residuals.png")
plt.close()

# -----------------------------------------------------------------------------
# 7. ARTIFACT EXPORT & ROUND-TRIP PREDICTION TEST
# -----------------------------------------------------------------------------
model_filename = "best_pipeline.joblib"
joblib.dump(best_pipeline, model_filename)
print(f"\nSaved complete pipeline to: {model_filename}")

# Round-trip verification on raw, unpreprocessed observation
reloaded_pipeline = joblib.load(model_filename)
raw_test_record = pd.DataFrame([{
    "pclass": 3,
    "sex": "female",
    "age": 22.0,
    "sibsp": 1,
    "parch": 0,
    "fare": 7.25,
    "embarked": "S"
}])

prediction = reloaded_pipeline.predict(raw_test_record)
prob = reloaded_pipeline.predict_proba(raw_test_record)[0, 1]
print(f"Verification Test -> Class: {prediction[0]}, Survival Probability: {prob:.4f}")