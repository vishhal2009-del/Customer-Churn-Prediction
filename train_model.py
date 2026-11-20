
# ============================================================
# CUSTOMER CHURN PREDICTION USING DEEP NEURAL NETWORK
# STEP 5: BUILD AND TRAIN DEEP NEURAL NETWORK
# ============================================================

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

# Updated TensorFlow Keras imports
from tensorflow import keras

from tensorflow.keras import Sequential

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

# ============================================================
# 1. LOAD PREPARED TRAINING DATA
# ============================================================

data = np.load("dataset/training_data.npz")

X_train = data["X_train"]
X_test = data["X_test"]

y_train = data["y_train"]
y_test = data["y_test"]

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# ============================================================
# 2. CREATE MODEL DIRECTORY
# ============================================================

os.makedirs("model", exist_ok=True)

# ============================================================
# 3. BUILD DEEP NEURAL NETWORK
# ============================================================

model = keras.Sequential([

    keras.layers.Input(shape=(X_train.shape[1],)),

    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),

    Dense(16, activation="relu"),

    Dense(1, activation="sigmoid")

])

# ============================================================
# 4. COMPILE MODEL
# ============================================================

model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc")
    ]

)

# ============================================================
# 5. DISPLAY MODEL ARCHITECTURE
# ============================================================

print("\n========== MODEL SUMMARY ==========")

model.summary()

# ============================================================
# 6. DEFINE CALLBACKS
# ============================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=0.00001
)

# ============================================================
# 7. TRAIN NEURAL NETWORK
# ============================================================

print("\n========== TRAINING STARTED ==========")

history = model.fit(

    X_train,
    y_train,

    epochs=50,
    batch_size=32,

    validation_split=0.2,

    callbacks=[
        early_stop,
        reduce_lr
    ],

    verbose=1

)

# ============================================================
# 8. EVALUATE MODEL
# ============================================================

print("\n========== MODEL EVALUATION ==========")

results = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

for name, value in zip(model.metrics_names, results):
    print(f"{name}: {value:.4f}")

# ============================================================
# 9. SAVE TRAINED MODEL
# ============================================================

model.save("model/churn_model.keras")

print("\nModel saved successfully!")

# ============================================================
# 10. PLOT ACCURACY
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Neural Network Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend()
plt.grid(True)

plt.savefig("model/accuracy.png")

plt.show()

# ============================================================
# 11. PLOT LOSS
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Neural Network Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.grid(True)

plt.savefig("model/loss.png")

plt.show()

# ============================================================
# PROJECT COMPLETED
# ============================================================

print("\n========== STEP 5 COMPLETED ==========")

print("Deep Neural Network training completed!")
print("Trained model: model/churn_model.keras")
print("Accuracy graph: model/accuracy.png")
print("Loss graph: model/loss.png")
