
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
import plotly.graph_objects as go


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# =====================================================
# 2. LOAD MODEL AND PREPROCESSING FILES
# =====================================================

@st.cache_resource
def load_files():

    model = tf.keras.models.load_model(
        "model/churn_model.keras"
    )

    scaler = joblib.load(
        "dataset/scaler.pkl"
    )

    feature_columns = joblib.load(
        "dataset/feature_columns.pkl"
    )

    csv_path = "dataset/Telco-Customer-Churn.csv"

    if not os.path.exists(csv_path):
        csv_path = "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    original_data = pd.read_csv(csv_path)

    original_data["TotalCharges"] = pd.to_numeric(
        original_data["TotalCharges"],
        errors="coerce"
    ).fillna(0)

    original_data = original_data.drop(
        columns=["customerID", "Churn"]
    )

    return model, scaler, feature_columns, original_data


try:

    model, scaler, feature_columns, original_data = load_files()

except Exception as e:

    st.error(f"Error loading model or dataset: {e}")
    st.stop()


# =====================================================
# 3. APPLICATION HEADER
# =====================================================

st.title("📊 Customer Churn Prediction System")

st.markdown(
    """
    ### Deep Neural Network Based Prediction

    This application predicts whether a customer
    is likely to leave the company using a trained
    Deep Neural Network.
    """
)

st.divider()


# =====================================================
# 4. CUSTOMER INFORMATION FORM
# =====================================================

with st.form("customer_form"):

    st.header("👤 Personal Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

    with col2:

        senior = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    with col3:

        partner = st.selectbox(
            "Has Partner?",
            ["Yes", "No"]
        )

    col1, col2 = st.columns(2)

    with col1:

        dependents = st.selectbox(
            "Has Dependents?",
            ["Yes", "No"]
        )

    with col2:

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12,
            step=1
        )

    st.divider()

    # =================================================
    # SERVICE INFORMATION
    # =================================================

    st.header("📞 Services Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        phone = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

    with col2:

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No phone service", "No", "Yes"]
        )

    with col3:

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

    internet_options = [
        "No internet service",
        "No",
        "Yes"
    ]

    col1, col2 = st.columns(2)

    with col1:

        online_security = st.selectbox(
            "Online Security",
            internet_options
        )

        online_backup = st.selectbox(
            "Online Backup",
            internet_options
        )

        device_protection = st.selectbox(
            "Device Protection",
            internet_options
        )

    with col2:

        tech_support = st.selectbox(
            "Tech Support",
            internet_options
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            internet_options
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            internet_options
        )

    st.divider()

    # =================================================
    # BILLING INFORMATION
    # =================================================

    st.header("💳 Billing Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

    with col2:

        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

    with col3:

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    col1, col2 = st.columns(2)

    with col1:

        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            value=70.0,
            step=1.0
        )

    # AUTOMATIC TOTAL CHARGES CALCULATION

    total_charges = monthly_charges * tenure

    with col2:

        st.metric(
            "Calculated Total Charges",
            f"${total_charges:.2f}"
        )

        st.caption(
            "Calculated using Monthly Charges × Tenure"
        )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Predict Churn",
        use_container_width=True
    )


# =====================================================
# 5. PREDICTION
# =====================================================

if submitted:

    # Collect customer information

    customer = {

        "gender": gender,

        "SeniorCitizen": senior,

        "Partner": partner,

        "Dependents": dependents,

        "tenure": tenure,

        "PhoneService": phone,

        "MultipleLines": multiple_lines,

        "InternetService": internet,

        "OnlineSecurity": online_security,

        "OnlineBackup": online_backup,

        "DeviceProtection": device_protection,

        "TechSupport": tech_support,

        "StreamingTV": streaming_tv,

        "StreamingMovies": streaming_movies,

        "Contract": contract,

        "PaperlessBilling": paperless,

        "PaymentMethod": payment,

        "MonthlyCharges": monthly_charges,

        "TotalCharges": total_charges

    }

    customer_df = pd.DataFrame([customer])

    # =================================================
    # 6. DATA PREPROCESSING
    # =================================================

    combined = pd.concat(
        [original_data, customer_df],
        ignore_index=True
    )

    combined_encoded = pd.get_dummies(
        combined,
        drop_first=True,
        dtype=int
    )

    X_customer = combined_encoded.iloc[[-1]]

    # Match training feature columns

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

    # =================================================
    # 7. NEURAL NETWORK PREDICTION
    # =================================================

    prediction = model.predict(
        X_customer,
        verbose=0
    )

    churn_probability = float(prediction[0][0])

    stay_probability = 1 - churn_probability

    # =================================================
    # 8. DISPLAY PREDICTION RESULT
    # =================================================

    st.divider()

    st.header("📈 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Churn Probability",
            f"{churn_probability * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Stay Probability",
            f"{stay_probability * 100:.2f}%"
        )

    if churn_probability >= 0.5:

        st.error(
            "⚠️ CUSTOMER LIKELY TO CHURN"
        )

        st.write(
            "The model predicts a higher likelihood "
            "that this customer will leave the company."
        )

    else:

        st.success(
            "✅ CUSTOMER LIKELY TO STAY"
        )

        st.write(
            "The model predicts a higher likelihood "
            "that this customer will remain with the company."
        )

    # =================================================
    # 9. CHURN PROBABILITY CHART
    # =================================================

    st.subheader("Churn Probability Visualization")

    fig = go.Figure(
        data=[
            go.Pie(
                labels=[
                    "Churn Risk",
                    "Stay Probability"
                ],

                values=[
                    churn_probability * 100,
                    stay_probability * 100
                ],

                hole=0.55,

                textinfo="label+percent"
            )
        ]
    )

    fig.update_layout(
        height=400,
        title="Customer Churn Analysis"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =================================================
    # 10. CUSTOMER RETENTION RECOMMENDATIONS
    # =================================================

    st.subheader(
        "💡 Customer Retention Recommendations"
    )

    recommendations = []

    if churn_probability >= 0.5:

        if contract == "Month-to-month":

            recommendations.append(
                "Offer a suitable long-term contract plan."
            )

        if payment == "Electronic check":

            recommendations.append(
                "Offer automatic payment options."
            )

        if tech_support == "No":

            recommendations.append(
                "Consider offering technical support services."
            )

        if online_security == "No":

            recommendations.append(
                "Introduce online security packages."
            )

        if tenure <= 12:

            recommendations.append(
                "Provide onboarding assistance and loyalty benefits."
            )

        if monthly_charges > 80:

            recommendations.append(
                "Review pricing and offer suitable plans."
            )

        if not recommendations:

            recommendations.append(
                "Offer personalized support and customer satisfaction reviews."
            )

        for item in recommendations:

            st.write("👉", item)

    else:

        st.success(
            "Customer retention risk is relatively low. "
            "Continue providing quality service and regular support."
        )

    # =================================================
    # 11. CUSTOMER BILLING SUMMARY
    # =================================================

    st.subheader("💰 Customer Billing Summary")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Tenure",
            f"{tenure} Months"
        )

    with col2:

        st.metric(
            "Monthly Charges",
            f"${monthly_charges:.2f}"
        )

    with col3:

        st.metric(
            "Total Charges",
            f"${total_charges:.2f}"
        )

    st.divider()

    st.caption(
        "This prediction is an estimate generated by "
        "a trained Deep Neural Network. It should be "
        "used as decision-support information."
    )

# ============================================================
# BULK CUSTOMER CHURN PREDICTION
# ============================================================

st.header("Bulk Customer Churn Prediction")

st.write(
    "Upload a CSV file containing multiple customer records "
    "to predict churn for all customers at once."
)

uploaded_file = st.file_uploader(
    "Upload Customer CSV",
    type=["csv"],
    key="bulk_customer_csv"
)

if uploaded_file is not None:

    try:
        bulk_data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Customer Data")
        st.dataframe(bulk_data.head())

        # Keep customer IDs for the final results
        if "customerID" in bulk_data.columns:
            customer_ids = bulk_data["customerID"].astype(str)
        else:
            customer_ids = pd.Series(
                [f"Customer_{i+1}" for i in range(len(bulk_data))]
            )

        # Remove columns that are not model inputs
        bulk_data = bulk_data.drop(
            columns=["customerID", "Churn"],
            errors="ignore"
        )

        # Required input columns
        required_columns = [
            "gender", "SeniorCitizen", "Partner", "Dependents",
            "tenure", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling",
            "PaymentMethod", "MonthlyCharges", "TotalCharges"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in bulk_data.columns
        ]

        if missing_columns:
            st.error(
                "Missing columns: " + ", ".join(missing_columns)
            )

        else:

            if st.button("Predict All Customers"):

                with st.spinner("Predicting customer churn..."):

                    # Convert TotalCharges to numeric
                    bulk_data["TotalCharges"] = pd.to_numeric(
                        bulk_data["TotalCharges"],
                        errors="coerce"
                    ).fillna(0)

                    # Clean the training reference data
                    reference_data = original_data.copy()

                    reference_data = reference_data.drop(
                        columns=["customerID", "Churn"],
                        errors="ignore"
                    )

                    reference_data["TotalCharges"] = pd.to_numeric(
                        reference_data["TotalCharges"],
                        errors="coerce"
                    ).fillna(0)

                    # Combine reference data and uploaded customers
                    combined_data = pd.concat(
                        [reference_data, bulk_data],
                        ignore_index=True
                    )

                    # Apply the same one-hot encoding
                    combined_encoded = pd.get_dummies(
                        combined_data,
                        drop_first=True,
                        dtype=int
                    )

                    # Keep only the features used during training
                    combined_encoded = combined_encoded.reindex(
                        columns=feature_columns,
                        fill_value=0
                    )

                    # Select only the uploaded customer rows
                    X_bulk = combined_encoded.iloc[
                        len(reference_data):
                    ].copy()

                    # Scale numerical columns
                    numeric_columns = [
                        "tenure",
                        "MonthlyCharges",
                        "TotalCharges"
                    ]

                    X_bulk[numeric_columns] = scaler.transform(
                        X_bulk[numeric_columns]
                    )

                    X_bulk = X_bulk.astype("float32")

                    # Predict all customers together
                    probabilities = model.predict(
                        X_bulk,
                        verbose=0
                    ).flatten()

                    # Create results
                    results = pd.DataFrame({
                        "Customer ID": customer_ids.values,
                        "Churn Probability": probabilities * 100,
                        "Prediction": [
                            "Likely to Churn"
                            if p >= 0.5
                            else "Likely to Stay"
                            for p in probabilities
                        ]
                    })

                    # Display summary
                    total_customers = len(results)
                    churn_count = (
                        results["Prediction"] == "Likely to Churn"
                    ).sum()
                    stay_count = total_customers - churn_count

                    st.success("Bulk prediction completed!")

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Total Customers",
                        total_customers
                    )

                    col2.metric(
                        "Likely to Churn",
                        int(churn_count)
                    )

                    col3.metric(
                        "Likely to Stay",
                        int(stay_count)
                    )

                    # Display prediction table
                    st.subheader("Prediction Results")

                    results["Churn Probability"] = (
                        results["Churn Probability"].round(2)
                    )

                    st.dataframe(
                        results,
                        use_container_width=True
                    )

                    # Download results
                    csv_data = results.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Download Prediction Results",
                        data=csv_data,
                        file_name="bulk_churn_predictions.csv",
                        mime="text/csv"
                    )

    except Exception as e:
        st.error(f"Error processing file: {e}")
