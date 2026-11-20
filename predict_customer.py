
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# --------------------------------------------------
# 1. LOAD TRAINED MODEL AND PREPROCESSING FILES
# --------------------------------------------------

model = tf.keras.models.load_model("model/churn_model.keras")

scaler = joblib.load("dataset/scaler.pkl")
feature_columns = joblib.load("dataset/feature_columns.pkl")

# Load original dataset to preserve training categories
original_data = pd.read_csv("dataset/Telco-Customer-Churn.csv")

original_data["TotalCharges"] = pd.to_numeric(
    original_data["TotalCharges"], errors="coerce"
).fillna(0)

original_data = original_data.drop(
    columns=["customerID", "Churn"]
)

# --------------------------------------------------
# 2. GET CUSTOMER DETAILS
# --------------------------------------------------

print("\n===================================")
print(" CUSTOMER CHURN PREDICTION SYSTEM")
print("===================================")

def get_choice(question, options):
    print(f"\n{question}")
    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")

    while True:
        try:
            choice = int(input("Enter option number: "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
        except ValueError:
            pass

        print("Invalid choice. Please try again.")


def get_number(question, minimum=None, maximum=None):
    while True:
        try:
            value = float(input(f"\n{question}: "))

            if minimum is not None and value < minimum:
                print(f"Enter a value of at least {minimum}.")
                continue

            if maximum is not None and value > maximum:
                print(f"Enter a value no greater than {maximum}.")
                continue

            return value

        except ValueError:
            print("Please enter a valid number.")


customer = {}

customer["gender"] = get_choice(
    "Gender", ["Female", "Male"]
)

customer["SeniorCitizen"] = int(get_choice(
    "Senior Citizen?", ["0 - No", "1 - Yes"]
)[0])

customer["Partner"] = get_choice(
    "Has Partner?", ["Yes", "No"]
)

customer["Dependents"] = get_choice(
    "Has Dependents?", ["Yes", "No"]
)

customer["tenure"] = get_number(
    "Number of months with the company", 0, 72
)

customer["PhoneService"] = get_choice(
    "Phone Service", ["Yes", "No"]
)

customer["MultipleLines"] = get_choice(
    "Multiple Lines",
    ["No phone service", "No", "Yes"]
)

customer["InternetService"] = get_choice(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)

internet_options = [
    "No internet service", "No", "Yes"
]

customer["OnlineSecurity"] = get_choice(
    "Online Security", internet_options
)

customer["OnlineBackup"] = get_choice(
    "Online Backup", internet_options
)

customer["DeviceProtection"] = get_choice(
    "Device Protection", internet_options
)

customer["TechSupport"] = get_choice(
    "Tech Support", internet_options
)

customer["StreamingTV"] = get_choice(
    "Streaming TV", internet_options
)

customer["StreamingMovies"] = get_choice(
    "Streaming Movies", internet_options
)

customer["Contract"] = get_choice(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

customer["PaperlessBilling"] = get_choice(
    "Paperless Billing", ["Yes", "No"]
)

customer["PaymentMethod"] = get_choice(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

customer["MonthlyCharges"] = get_number(
    "Monthly Charges", 0
)

customer["TotalCharges"] = get_number(
    "Total Charges", 0
)

# --------------------------------------------------
# 3. PREPROCESS CUSTOMER DATA
# --------------------------------------------------

customer_df = pd.DataFrame([customer])

# Combine with original data so dummy columns
# use the same categories as during training
combined = pd.concat(
    [original_data, customer_df],
    ignore_index=True
)

combined_encoded = pd.get_dummies(
    combined,
    drop_first=True,
    dtype=int
)

# Select only the new customer's row
X_customer = combined_encoded.iloc[[-1]]

# Match the exact training column order
X_customer = X_customer.reindex(
    columns=feature_columns,
    fill_value=0
)

# Scale numerical columns
numerical_columns = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

X_customer[numerical_columns] = scaler.transform(
    X_customer[numerical_columns]
)

X_customer = X_customer.astype("float32")

# --------------------------------------------------
# 4. PREDICT CHURN
# --------------------------------------------------

prediction = model.predict(
    X_customer, verbose=0
)

churn_probability = float(prediction[0][0])

# --------------------------------------------------
# 5. DISPLAY RESULT
# --------------------------------------------------

print("\n===================================")
print("         PREDICTION RESULT")
print("===================================")

print(
    f"Churn Probability: {churn_probability * 100:.2f}%"
)

if churn_probability >= 0.5:
    print("\nPrediction: CUSTOMER LIKELY TO CHURN")
else:
    print("\nPrediction: CUSTOMER LIKELY TO STAY")

print("===================================")
