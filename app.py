
import os
import urllib.request
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

st.title("Customer Churn Prediction System")
st.divider()


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATHS = [
    os.path.join(BASE_DIR, "model", "churn_model.keras"),
    os.path.join(BASE_DIR, "models", "churn_model.keras")
]

SCALER_PATH = os.path.join(BASE_DIR, "dataset", "scaler.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "dataset", "feature_columns.pkl")

DATASET_PATHS = [
    os.path.join(BASE_DIR, "dataset", "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
    os.path.join(BASE_DIR, "dataset", "Telco-Customer-Churn.csv")
]

NUMERIC_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

REQUIRED_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    for path in DATASET_PATHS:

        if os.path.exists(path):

            df = pd.read_csv(path)

            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"],
                errors="coerce"
            )

            df = df.dropna(subset=["TotalCharges"])

            return df

    return None


# ============================================================
# LOAD MODEL AND PREPROCESSING FILES
# ============================================================

@st.cache_resource
def load_model_and_preprocessing():

    model_path = next(
        (path for path in MODEL_PATHS if os.path.exists(path)),
        None
    )

    if model_path is None:
        raise FileNotFoundError(
            "Model file not found. Check model/churn_model.keras "
            "or models/churn_model.keras."
        )

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("dataset/scaler.pkl was not found.")

    if not os.path.exists(FEATURE_PATH):
        raise FileNotFoundError(
            "dataset/feature_columns.pkl was not found."
        )

    model = tf.keras.models.load_model(model_path)

    scaler = joblib.load(SCALER_PATH)

    feature_columns = joblib.load(FEATURE_PATH)

    return model, scaler, list(feature_columns)


# ============================================================
# USD TO INR EXCHANGE RATE
# ============================================================

FALLBACK_RATE = 92.68


@st.cache_data(ttl=3600)
def get_exchange_rate():

    try:

        url = "https://open.er-api.com/v6/latest/USD"

        request = urllib.request.Request(
            url,
            headers={"User-Agent": "CustomerChurnPrediction/1.0"}
        )

        with urllib.request.urlopen(request, timeout=10) as response:

            data = json.loads(response.read().decode("utf-8"))

        rate = float(data["rates"]["INR"])

        if rate <= 0:
            raise ValueError("Invalid exchange rate.")

        return rate, True

    except Exception:

        return FALLBACK_RATE, False


USD_INR_RATE, LIVE_RATE_AVAILABLE = get_exchange_rate()


# ============================================================
# CURRENCY CONVERSION
# ============================================================

def inr_to_usd(amount):

    return float(amount) / USD_INR_RATE


def usd_to_inr(amount):

    return float(amount) * USD_INR_RATE


# ============================================================
# PREPROCESSING FUNCTION
# ============================================================

def prepare_model_input(
    customer_data,
    original_data,
    scaler,
    feature_columns
):

    base_features = original_data.drop(
        columns=["customerID", "Churn"],
        errors="ignore"
    ).copy()

    base_features["TotalCharges"] = pd.to_numeric(
        base_features["TotalCharges"],
        errors="coerce"
    )

    base_features = base_features.dropna(
        subset=["TotalCharges"]
    )

    combined_data = pd.concat(
        [base_features, customer_data],
        ignore_index=True
    )

    encoded_data = pd.get_dummies(
        combined_data,
        drop_first=True
    )

    customer_encoded = encoded_data.iloc[
        len(base_features):
    ].copy()

    customer_encoded = customer_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    customer_encoded = customer_encoded.astype(float)

    available_numeric = [
        col for col in NUMERIC_COLUMNS
        if col in customer_encoded.columns
    ]

    if len(available_numeric) != len(NUMERIC_COLUMNS):
        raise ValueError(
            "The saved feature columns do not contain all "
            "required numerical columns."
        )

    customer_encoded.loc[:, available_numeric] = (
        scaler.transform(
            customer_encoded[available_numeric]
        )
    )

    return customer_encoded


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_churn(model, input_data):

    predictions = model.predict(
        input_data,
        verbose=0
    )

    probabilities = np.asarray(predictions).reshape(-1)

    return probabilities


# ============================================================
# LOAD PROJECT FILES
# ============================================================

try:

    model, scaler, feature_columns = (
        load_model_and_preprocessing()
    )

    original_data = load_dataset()

    if original_data is None:
        st.error(
            "Dataset not found. Place your Telco Customer "
            "Churn CSV file inside the dataset folder."
        )
        st.stop()

except Exception as error:

    st.error(f"Unable to load project files: {error}")
    st.stop()


# ============================================================
# INDIVIDUAL CUSTOMER PREDICTION
# ============================================================

st.header("Individual Customer Prediction")

with st.form("individual_prediction_form"):

    st.subheader("Personal Information")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            index=None,
            placeholder="Select an option"
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            ["Yes", "No"],
            index=None,
            placeholder="Select an option"
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"],
            index=None,
            placeholder="Select an option"
        )

    with col2:

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"],
            index=None,
            placeholder="Select an option"
        )

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=100,
            value=None,
            step=1,
            placeholder="Enter tenure"
        )

    st.divider()

    st.subheader("Services Information")

    col1, col2 = st.columns(2)

    with col1:

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"],
            index=None,
            placeholder="Select an option"
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"],
            index=None,
            placeholder="Select an option"
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
            index=None,
            placeholder="Select an option"
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"],
            index=None,
            placeholder="Select an option"
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"],
            index=None,
            placeholder="Select an option"
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"],
            index=None,
            placeholder="Select an option"
        )

    with col2:

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"],
            index=None,
            placeholder="Select an option"
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"],
            index=None,
            placeholder="Select an option"
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"],
            index=None,
            placeholder="Select an option"
        )

    st.divider()

    st.subheader("Billing Information")

    if LIVE_RATE_AVAILABLE:

        st.caption(
            f"Exchange Rate: 1 USD = ₹{USD_INR_RATE:,.2f} INR"
        )

    else:

        st.caption(
            f"Exchange Rate: 1 USD = ₹{USD_INR_RATE:,.2f} INR "
            "(fallback rate; live rate unavailable)"
        )

    col1, col2 = st.columns(2)

    with col1:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ],
            index=None,
            placeholder="Select an option"
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"],
            index=None,
            placeholder="Select an option"
        )

    with col2:

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ],
            index=None,
            placeholder="Select an option"
        )

        monthly_charges_inr = st.number_input(
            "Monthly Charges (₹)",
            min_value=0.0,
            value=None,
            step=10.0,
            placeholder="Enter monthly charges in INR"
        )

    submitted = st.form_submit_button(
        "Predict Churn",
        use_container_width=True
    )


# ============================================================
# INDIVIDUAL PREDICTION PROCESSING
# ============================================================

if submitted:

    required_inputs = {
        "Gender": gender,
        "Senior Citizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure": tenure,
        "Phone Service": phone_service,
        "Multiple Lines": multiple_lines,
        "Internet Service": internet_service,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Contract": contract,
        "Paperless Billing": paperless_billing,
        "Payment Method": payment_method,
        "Monthly Charges": monthly_charges_inr
    }

    missing_fields = [
        field for field, value in required_inputs.items()
        if value is None
    ]

    if missing_fields:

        st.error(
            "Please complete all required fields: "
            + ", ".join(missing_fields)
        )

    elif monthly_charges_inr <= 0:

        st.error("Monthly charges must be greater than zero.")

    elif tenure <= 0:

        st.error("Tenure must be greater than zero months.")

    else:

        try:

            total_charges_inr = (
                monthly_charges_inr * tenure
            )

            monthly_charges_usd = inr_to_usd(
                monthly_charges_inr
            )

            total_charges_usd = inr_to_usd(
                total_charges_inr
            )

            customer = pd.DataFrame([{

                "gender": gender,

                "SeniorCitizen": (
                    1 if senior_citizen == "Yes" else 0
                ),

                "Partner": partner,
                "Dependents": dependents,

                "tenure": int(tenure),

                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,

                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,

                "MonthlyCharges": monthly_charges_usd,
                "TotalCharges": total_charges_usd

            }])

            processed_customer = prepare_model_input(
                customer,
                original_data,
                scaler,
                feature_columns
            )

            probability = predict_churn(
                model,
                processed_customer
            )[0]

            churn_percentage = probability * 100

            st.divider()

            st.header("Prediction Results")

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                st.metric(
                    "Churn Probability",
                    f"{churn_percentage:.2f}%"
                )

            with result_col2:

                st.metric(
                    "Prediction",
                    "Likely to Churn"
                    if probability >= 0.5
                    else "Likely to Stay"
                )

            probability_chart = pd.DataFrame({
                "Status": ["Churn Probability", "Stay Probability"],
                "Probability": [
                    probability * 100,
                    (1 - probability) * 100
                ]
            })

            fig = px.bar(
                probability_chart,
                x="Status",
                y="Probability",
                title="Customer Prediction Probability",
                range_y=[0, 100],
                text_auto=".2f"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.subheader("Customer Retention Recommendations")

            if probability >= 0.5:

                st.warning(
                    "This customer has been classified as likely to churn."
                )

                st.write(
                    "- Offer suitable retention benefits."
                )

                st.write(
                    "- Review the customer's current service plan."
                )

                st.write(
                    "- Provide timely customer support."
                )

                st.write(
                    "- Consider a suitable contract renewal offer."
                )

            else:

                st.success(
                    "This customer has been classified as likely to stay."
                )

                st.write(
                    "- Continue providing reliable service."
                )

                st.write(
                    "- Maintain customer satisfaction."
                )

                st.write(
                    "- Offer relevant upgrades when appropriate."
                )

            st.subheader("Customer Billing Summary")

            bill_col1, bill_col2 = st.columns(2)

            bill_col1.metric(
                "Monthly Charges",
                f"₹{monthly_charges_inr:,.2f}"
            )

            bill_col2.metric(
                "Estimated Total Charges",
                f"₹{total_charges_inr:,.2f}"
            )

        except Exception as error:

            st.error(
                f"Prediction could not be completed: {error}"
            )


# ============================================================
# BULK CUSTOMER PREDICTION
# ============================================================

st.divider()

st.header("Bulk Customer Prediction")

st.write(
    "Upload a CSV file containing customer information. "
    "MonthlyCharges and TotalCharges must be entered in INR."
)

uploaded_file = st.file_uploader(
    "Upload Customer CSV File",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        bulk_data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Customer Data")

        st.dataframe(
            bulk_data.head(),
            use_container_width=True
        )

        missing_columns = [
            column for column in REQUIRED_COLUMNS
            if column not in bulk_data.columns
        ]

        if missing_columns:

            st.error(
                "Missing required columns: "
                + ", ".join(missing_columns)
            )

        else:

            if st.button(
                "Predict Churn for All Customers",
                use_container_width=True
            ):

                bulk_customer = bulk_data[
                    REQUIRED_COLUMNS
                ].copy()

                for column in NUMERIC_COLUMNS:

                    bulk_customer[column] = pd.to_numeric(
                        bulk_customer[column],
                        errors="coerce"
                    )

                if bulk_customer.isnull().any().any():

                    st.error(
                        "The CSV contains missing or invalid values. "
                        "Check the numerical columns."
                    )

                elif (
                    (bulk_customer["tenure"] <= 0).any()
                    or (bulk_customer["MonthlyCharges"] <= 0).any()
                    or (bulk_customer["TotalCharges"] < 0).any()
                ):

                    st.error(
                        "Tenure and MonthlyCharges must be greater "
                        "than zero, and TotalCharges cannot be negative."
                    )

                else:

                    try:

                        senior_values = (
                            bulk_customer["SeniorCitizen"]
                            .astype(str)
                            .str.strip()
                            .str.lower()
                        )

                        senior_mapping = {
                            "yes": 1,
                            "no": 0,
                            "1": 1,
                            "0": 0
                        }

                        converted_senior = senior_values.map(
                            senior_mapping
                        )

                        if converted_senior.isnull().any():

                            st.error(
                                "SeniorCitizen must contain Yes/No "
                                "or 1/0 values."
                            )

                            st.stop()

                        bulk_customer["SeniorCitizen"] = (
                            converted_senior.astype(int)
                        )

                        bulk_customer["MonthlyCharges"] = (
                            bulk_customer["MonthlyCharges"]
                            / USD_INR_RATE
                        )

                        bulk_customer["TotalCharges"] = (
                            bulk_customer["TotalCharges"]
                            / USD_INR_RATE
                        )

                        processed_bulk = prepare_model_input(
                            bulk_customer,
                            original_data,
                            scaler,
                            feature_columns
                        )

                        probabilities = predict_churn(
                            model,
                            processed_bulk
                        )

                        results = bulk_data.copy()

                        results["Churn Probability (%)"] = (
                            probabilities * 100
                        ).round(2)

                        results["Prediction"] = np.where(
                            probabilities >= 0.5,
                            "Likely to Churn",
                            "Likely to Stay"
                        )

                        st.divider()

                        st.subheader(
                            "Bulk Prediction Results"
                        )

                        st.dataframe(
                            results,
                            use_container_width=True
                        )

                        likely_to_churn = int(
                            (probabilities >= 0.5).sum()
                        )

                        likely_to_stay = int(
                            (probabilities < 0.5).sum()
                        )

                        summary_col1, summary_col2, summary_col3 = (
                            st.columns(3)
                        )

                        summary_col1.metric(
                            "Total Customers",
                            len(results)
                        )

                        summary_col2.metric(
                            "Likely to Churn",
                            likely_to_churn
                        )

                        summary_col3.metric(
                            "Likely to Stay",
                            likely_to_stay
                        )

                        chart_data = pd.DataFrame({
                            "Prediction": [
                                "Likely to Churn",
                                "Likely to Stay"
                            ],
                            "Customers": [
                                likely_to_churn,
                                likely_to_stay
                            ]
                        })

                        fig = px.bar(
                            chart_data,
                            x="Prediction",
                            y="Customers",
                            title="Bulk Customer Prediction Summary",
                            text="Customers"
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                        csv_data = results.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="Download Prediction Results CSV",
                            data=csv_data,
                            file_name="customer_churn_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    except Exception as error:

                        st.error(
                            f"Bulk prediction failed: {error}"
                        )

    except Exception as error:

        st.error(
            f"Unable to read the uploaded CSV file: {error}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Customer Churn Prediction System"
)
