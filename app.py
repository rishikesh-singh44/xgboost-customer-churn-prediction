from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier


PROJECT_DIR = Path(__file__).resolve().parent

PREPROCESSOR_PATH = (
    PROJECT_DIR
    / "models"
    / "churn_preprocessor.joblib"
)

MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "churn_xgboost.json"
)


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
)


@st.cache_resource
def load_assets():
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    return preprocessor, model


preprocessor, model = load_assets()


st.title("Customer Churn Prediction")

st.write(
    "Enter the customer information and click Predict."
)


with st.form("customer_form"):
    left, right = st.columns(2)

    with left:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"],
        )

        senior_citizen = st.selectbox(
            "Senior citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )

        partner = st.selectbox(
            "Partner",
            ["No", "Yes"],
        )

        dependents = st.selectbox(
            "Dependents",
            ["No", "Yes"],
        )

        tenure = st.number_input(
            "Tenure in months",
            min_value=0,
            max_value=100,
            value=12,
        )

        phone_service = st.selectbox(
            "Phone service",
            ["No", "Yes"],
        )

        multiple_lines = st.selectbox(
            "Multiple lines",
            ["No", "Yes", "No phone service"],
        )

        internet_service = st.selectbox(
            "Internet service",
            ["DSL", "Fiber optic", "No"],
        )

        online_security = st.selectbox(
            "Online security",
            ["No", "Yes", "No internet service"],
        )

        online_backup = st.selectbox(
            "Online backup",
            ["No", "Yes", "No internet service"],
        )

    with right:
        device_protection = st.selectbox(
            "Device protection",
            ["No", "Yes", "No internet service"],
        )

        tech_support = st.selectbox(
            "Technical support",
            ["No", "Yes", "No internet service"],
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"],
        )

        streaming_movies = st.selectbox(
            "Streaming movies",
            ["No", "Yes", "No internet service"],
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )

        paperless_billing = st.selectbox(
            "Paperless billing",
            ["No", "Yes"],
        )

        payment_method = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

        monthly_charges = st.number_input(
            "Monthly charges",
            min_value=0.0,
            value=70.0,
        )

        total_charges = st.number_input(
            "Total charges",
            min_value=0.0,
            value=840.0,
        )

    submitted = st.form_submit_button(
        "Predict churn",
        type="primary",
    )


if submitted:
    customer = pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
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
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }
        ]
    )

    processed_customer = preprocessor.transform(
        customer
    )

    churn_probability = float(
        model.predict_proba(processed_customer)[0, 1]
    )

    st.subheader("Prediction result")

    st.metric(
        "Churn probability",
        f"{churn_probability:.2%}",
    )

    if churn_probability >= 0.70:
        st.error("High churn risk")
    elif churn_probability >= 0.40:
        st.warning("Medium churn risk")
    else:
        st.success("Low churn risk")