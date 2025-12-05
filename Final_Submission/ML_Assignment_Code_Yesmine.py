# ---------------------------------------------------------------------------------#
# I. Preparation Script
# ---------------------------------------------------------------------------------#

import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset from local file
data = pd.read_csv("dataset.csv")

# Random split into learning and test sets
learning_df, test_df = train_test_split(
    data,
    test_size=0.2,       # 80% learning, 20% test
    random_state=0       # fixed seed for reproducibility
)

# Save the two sets as Python objects in the working directory
learning_df.to_pickle("learning.pkl")
test_df.to_pickle("test.pkl")

# Small check
print("Learning set shape:", learning_df.shape)
print("Test set shape:", test_df.shape)

# ---------------------------------------------------------------------------------#
# II. Model Building Script
# ---------------------------------------------------------------------------------#

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

import joblib

# 1. Load learning set and separate X and y

learning_df = pd.read_pickle("learning.pkl")

y = learning_df["target"]
# idx is NOT an explanatory variable
X = learning_df.drop(columns=["target", "idx"])

# 2. Preprocessing pipelines

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# 3. Baseline Decision Tree

baseline_pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", DecisionTreeRegressor(random_state=0))
])

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=0
)

baseline_pipe.fit(X_train, y_train)
y_valid_pred = baseline_pipe.predict(X_valid)
baseline_rmse = root_mean_squared_error(y_valid, y_valid_pred)
print("Baseline Decision Tree RMSE (simple split):", baseline_rmse)

joblib.dump(baseline_pipe, "baseline_model.joblib")
print("Baseline model saved to baseline_model.joblib")

# 4. Model selection with GridSearchCV + KFold

cv = KFold(n_splits=5, shuffle=True, random_state=0)

# 4.a Tuned Decision Tree
pipe_dt = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", DecisionTreeRegressor(random_state=0))
])

param_grid_dt = {
    "model__max_depth": [3, 5, 8, 12, 14],
    "model__min_samples_split": [2, 10, 50, 100, 200]
}

dt_search = GridSearchCV(
    estimator=pipe_dt,
    param_grid=param_grid_dt,
    cv=cv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

dt_search.fit(X, y)

dt_rmse_cv = -dt_search.best_score_
dt_std_cv = dt_search.cv_results_["std_test_score"][dt_search.best_index_]

print("Best Decision Tree parameters:", dt_search.best_params_)
print("Best Decision Tree CV RMSE (mean over folds):", dt_rmse_cv)
print("Decision Tree CV RMSE std over folds:", dt_std_cv)

# 4.b Tuned Random Forest
pipe_rf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(random_state=0, n_jobs=-1))
])

param_grid_rf = {
    "model__n_estimators": [200, 300, 400],
    "model__max_depth": [2, None],
    "model__min_samples_split": [2, 5, 10]
}

rf_search = GridSearchCV(
    estimator=pipe_rf,
    param_grid=param_grid_rf,
    cv=cv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

rf_search.fit(X, y)

rf_rmse_cv = -rf_search.best_score_
rf_std_cv = rf_search.cv_results_["std_test_score"][rf_search.best_index_]

print("Best Random Forest parameters:", rf_search.best_params_)
print("Best Random Forest CV RMSE (mean over folds):", rf_rmse_cv)
print("Random Forest CV RMSE std over folds:", rf_std_cv)

# 5. Compare models and choose the final one (prefer the simpler model if performances are similar)

tolerance = 0.001
noise_level = max(dt_std_cv, rf_std_cv)

if (rf_rmse_cv + max(tolerance, noise_level)) < dt_rmse_cv:
    final_model = rf_search.best_estimator_
    final_model_name = "Random Forest"
    final_rmse_cv = rf_rmse_cv
else:
    final_model = dt_search.best_estimator_
    final_model_name = "Decision Tree"
    final_rmse_cv = dt_rmse_cv

print("Selected model:", final_model_name)
print("Selected model CV RMSE (mean over folds):", final_rmse_cv)

# 6. Save the final selected model

joblib.dump(final_model, "final_model.joblib")
print("Final model saved to final_model.joblib")

# ---------------------------------------------------------------------------------#
# III. Result Analysis Script
# ---------------------------------------------------------------------------------#

import pandas as pd
import numpy as np

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load test set and final model

test_df = pd.read_pickle("test.pkl")
model = joblib.load("final_model.joblib")

# Separate features and target for the test set
y_test = test_df["target"]
# idx is not an explanatory variable
X_test = test_df.drop(columns=["target", "idx"])

# 2. Predict on the test set and compute metrics

y_pred = model.predict(X_test)

rmse_test = root_mean_squared_error(y_test, y_pred)
mae_test = mean_absolute_error(y_test, y_pred)
r2_test = r2_score(y_test, y_pred)

print("Test RMSE:", rmse_test)
print("Test MAE :", mae_test)
print("Test R²  :", r2_test)

# 3. Diagnostic plots (regression setting)

sns.set_theme()

# 3.1 True vs predicted (with an optimal line y = x)
plt.figure()
sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
plt.xlabel("True target")
plt.ylabel("Predicted target")
plt.title("True vs Predicted on Test Set")
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
plt.tight_layout()
plt.savefig("plot_true_vs_pred.png")
plt.show()

# 3.2 Error distribution
errors = y_pred - y_test
plt.figure()
sns.histplot(errors, kde=True)
plt.xlabel("Prediction error (y_pred - y_true)")
plt.title("Error distribution on Test Set")
plt.tight_layout()
plt.savefig("plot_error_distribution.png")
plt.show()

# 3.3 Distribution of true vs predicted values
dist_df = pd.DataFrame({
    "value": pd.concat([y_test, pd.Series(y_pred, index=y_test.index)]),
    "type": ["true"] * len(y_test) + ["predicted"] * len(y_test)
})

plt.figure()
sns.kdeplot(data=dist_df, x="value", hue="type", common_norm=False)
plt.xlabel("Target value")
plt.title("Distribution of true vs predicted target (Test Set)")
plt.tight_layout()
plt.savefig("plot_distribution_true_pred.png")
plt.show()

# 3.4 Error quantiles (95% interval)
q_low, q_high = np.quantile(errors, [0.025, 0.975])
print("95% empirical error interval: [", q_low, ",", q_high, "]")

# 3.5 Median absolute error
median_ae = np.median(np.abs(errors))
print("Median absolute error:", median_ae)

# 3.6 Maximum absolute error
abs_errors = np.abs(errors)
max_abs_error = abs_errors.max()
idx_max_error = abs_errors.idxmax()

print("Maximum absolute error:", max_abs_error)
print("Index of worst prediction in test set:", idx_max_error)

# 3.7 Worst predicted observation
print("Worst predicted observation (features + true target):")
print(test_df.loc[idx_max_error])

# 4. Permutation-based feature importance (on test set)

# Baseline RMSE on the test set
baseline_rmse = rmse_test

feature_importance = []

rng = np.random.RandomState(0)  # for reproducibility

for col in X_test.columns:
    X_perm = X_test.copy()
    # permute one column
    X_perm[col] = rng.permutation(X_perm[col].values)
    y_perm_pred = model.predict(X_perm)
    rmse_perm = root_mean_squared_error(y_test, y_perm_pred)
    delta_rmse = rmse_perm - baseline_rmse
    feature_importance.append((col, delta_rmse))

fi_df = pd.DataFrame(feature_importance, columns=["feature", "delta_rmse"])
fi_df = fi_df.sort_values("delta_rmse", ascending=False)

print("\nPermutation-based feature importance (top 10 by ΔRMSE):")
print(fi_df.head(10))

# Bar plot of permutation importance
plt.figure(figsize=(8, 5))
sns.barplot(data=fi_df, x="delta_rmse", y="feature")
plt.xlabel("Increase in RMSE after permutation (ΔRMSE)")
plt.ylabel("Feature")
plt.title("Permutation feature importance on Test Set")
plt.tight_layout()
plt.savefig("plot_permutation.png")
plt.show()

# 5. Predictions for evaluation.csv (prediction.csv)

eval_df = pd.read_csv("evaluation.csv")
X_eval = eval_df.drop(columns=["idx"])
y_eval_pred = model.predict(X_eval)

predictions = pd.DataFrame({
    "idx": eval_df["idx"],
    "target": y_eval_pred
})

predictions.to_csv("predictions.csv", index=False)
print("\npredictions.csv written with shape:", predictions.shape)