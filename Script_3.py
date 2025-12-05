# Result Analysis

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
