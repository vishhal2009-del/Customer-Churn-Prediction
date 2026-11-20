
# ============================================================
# CUSTOMER CHURN PREDICTION
# STEP 6: MODEL PERFORMANCE EVALUATION
# ============================================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)

# ============================================================
# 1. LOAD TESTING DATA
# ============================================================

data = np.load("dataset/training_data.npz")

X_test = data["X_test"]
y_test = data["y_test"].astype(int)

print("Testing data loaded successfully!")
print("Testing samples:", X_test.shape[0])

# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

model = tf.keras.models.load_model(
    "model/churn_model.keras"
)

print("\nTrained model loaded successfully!")

# ============================================================
# 3. MAKE PREDICTIONS
# ============================================================

y_probability = model.predict(
    X_test,
    verbose=0
).ravel()

# Convert probabilities into binary predictions
y_pred = (y_probability >= 0.5).astype(int)

# ============================================================
# 4. CALCULATE PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

auc_score = roc_auc_score(
    y_test,
    y_probability
)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"ROC-AUC: {auc_score:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Churn", "Churn"],
        zero_division=0
    )
)

# ============================================================
# 5. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

print("\n========== CONFUSION MATRIX ==========")

print(cm)

print("\nTrue Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

# ============================================================
# 6. CREATE MODEL DIRECTORY
# ============================================================

os.makedirs("model", exist_ok=True)

# ============================================================
# 7. PLOT CONFUSION MATRIX
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)

disp.plot(cmap="Blues", values_format="d")

plt.title("Customer Churn Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "model/confusion_matrix.png",
    dpi=300
)

plt.show()

# ============================================================
# 8. ROC-AUC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(8, 5))

plt.plot(
    fpr,
    tpr,
    label=f"ROC Curve (AUC = {auc_score:.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC-AUC Curve")

plt.legend()
plt.grid(True)

plt.savefig(
    "model/roc_curve.png",
    dpi=300
)

plt.show()

# ============================================================
# 9. SAVE EVALUATION RESULTS
# ============================================================

with open("model/evaluation_results.txt", "w") as file:

    file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    file.write(f"ROC-AUC: {auc_score:.4f}\n")

    file.write("\nClassification Report:\n")

    file.write(
        classification_report(
            y_test,
            y_pred,
            target_names=["No Churn", "Churn"],
            zero_division=0
        )
    )

print("\n========== STEP 6 COMPLETED ==========")

print("Confusion matrix saved!")
print("ROC curve saved!")
print("Evaluation results saved!")
