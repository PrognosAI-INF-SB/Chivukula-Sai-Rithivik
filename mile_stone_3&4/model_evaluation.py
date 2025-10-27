# milestone3_evaluation_full_fd004.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from keras.losses import MeanSquaredError
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === Configuration ===
fd_number = 4  # Change to 1,2,3,4 as needed
window_size = 30
sequences_path = fr"C:\Users\chsai\OneDrive\Desktop\infosys internship\dataset\CMaps\mile stone1 results\FD00{fd_number}_sequences.npz"
model_path = fr"C:\Users\chsai\OneDrive\Desktop\infosys internship\dataset\CMaps\mile stone2 results\fd00{fd_number}_model.h5"
results_dir = r"C:\Users\chsai\OneDrive\Desktop\infosys internship\dataset\CMaps\milestone3 results"

# RUL thresholds
WARNING_THRESHOLD = 50
CRITICAL_THRESHOLD = 20

# === Load sequences and model ===
print(f"Loading sequences and trained model for FD00{fd_number}...")
data = np.load(sequences_path)
X_test_all = data["X_test"]
y_test_all = data["y_test"]

model = load_model(model_path, custom_objects={'mse': MeanSquaredError()})
print("Model loaded successfully.")

# === Predict RUL for all sequences ===
print("Predicting RUL for all sequences...")
y_pred_all = model.predict(X_test_all).flatten()

# === Compute metrics ===
rmse = np.sqrt(mean_squared_error(y_test_all, y_pred_all))
mae = mean_absolute_error(y_test_all, y_pred_all)
r2 = r2_score(y_test_all, y_pred_all)

metrics_df = pd.DataFrame({
    "FD_Set": [f"FD00{fd_number}"],
    "RMSE": [rmse],
    "MAE": [mae],
    "R2_Score": [r2]
})
metrics_df.to_csv(os.path.join(results_dir, f"metrics_fd{fd_number}.csv"), index=False)
print(f"Metrics saved to {results_dir}")

print(f"\nEvaluation Report — FD00{fd_number}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R² Score : {r2:.4f}")

# === Assign RUL labels ===
def label_rul(rul):
    if rul <= CRITICAL_THRESHOLD:
        return "CRITICAL"
    elif rul <= WARNING_THRESHOLD:
        return "WARNING"
    else:
        return "NORMAL"

labels_all = [label_rul(r) for r in y_pred_all]

labeled_df = pd.DataFrame({
    "Sample_Index": np.arange(len(y_test_all)),
    "Actual_RUL": y_test_all,
    "Predicted_RUL": y_pred_all,
    "Condition_Label": labels_all
})
labeled_df.to_csv(os.path.join(results_dir, f"rul_labels_fd{fd_number}.csv"), index=False)
print(f"Labeled predictions saved to: {results_dir}")

# === Plots ===
# 1. Predicted vs Actual
plt.figure(figsize=(8,6))
plt.scatter(y_test_all, y_pred_all, alpha=0.5, color="dodgerblue")
plt.plot([y_test_all.min(), y_test_all.max()], [y_test_all.min(), y_test_all.max()], 'r--', linewidth=2)
plt.xlabel("Actual RUL")
plt.ylabel("Predicted RUL")
plt.title(f"Predicted vs Actual RUL (FD00{fd_number})")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, f"pred_vs_actual_fd{fd_number}.png"))
plt.close()

# 2. Residual Distribution
residuals = y_test_all - y_pred_all
plt.figure(figsize=(8,5))
sns.histplot(residuals, bins=40, kde=True, color="mediumvioletred")
plt.xlabel("Residual (y_test - y_pred)")
plt.ylabel("Frequency")
plt.title(f"Residual Distribution (FD00{fd_number})")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, f"residual_dist_fd{fd_number}.png"))
plt.close()

# 3. Residual Trend (first 600 points)
plt.figure(figsize=(10,5))
plt.plot(residuals[:600], color="orange", linewidth=1.2)
plt.xlabel("Sample Index")
plt.ylabel("Residual")
plt.title(f"Residual Trend (First 600 Samples) — FD00{fd_number}")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, f"residual_trend_fd{fd_number}.png"))
plt.close()

# === Bias Analysis ===
mean_res = np.mean(residuals)
std_res = np.std(residuals)
bias_type = "Underestimation" if mean_res > 0 else "Overestimation"
bias_strength = "Low" if abs(mean_res) < std_res * 0.5 else "High"

bias_df = pd.DataFrame({
    "Mean_Residual": [mean_res],
    "Std_Residual": [std_res],
    "Bias_Type": [bias_type],
    "Bias_Strength": [bias_strength]
})
bias_df.to_csv(os.path.join(results_dir, f"bias_analysis_fd{fd_number}.csv"), index=False)
print(f"Bias analysis saved to: {results_dir}")

print(f"\nEvaluation completed successfully for FD00{fd_number}")
