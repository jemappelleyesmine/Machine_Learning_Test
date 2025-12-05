# Model building

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