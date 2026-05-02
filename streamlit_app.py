import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import os

MODEL_PATH        = "app/models/model.pkl"
SCALER_PATH       = "app/models/scaler.pkl"
FEATURE_PATH      = "app/models/feature_names.pkl"
SHAP_PLOT_PATH    = "app/plots/shap_summary.png"

@st.cache_resource
def load_model():
    model         = joblib.load(MODEL_PATH)
    scaler        = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_PATH)
    return model, scaler, feature_names

def build_input(tenure, monthly_charges, contract,
                internet, senior, partner, dependents,
                phone, paperless, payment,
                online_sec, online_backup, device_prot,
                tech_support, streaming_tv, streaming_movies,
                multiple_lines, feature_names):

    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
    payment_map  = {
        "Bank transfer": 0, "Credit card": 1,
        "Electronic check": 2, "Mailed check": 3
    }
    yes_no = lambda x: 1 if x == "Yes" else 0

    total_charges       = monthly_charges * tenure
    charges_per_tenure  = monthly_charges / (tenure + 1)
    is_new_customer     = 1 if tenure < 6 else 0
    phone_val           = yes_no(phone)
    multiple_val        = yes_no(multiple_lines)
    sec_val             = yes_no(online_sec)
    backup_val          = yes_no(online_backup)
    device_val          = yes_no(device_prot)
    tech_val            = yes_no(tech_support)
    tv_val              = yes_no(streaming_tv)
    movies_val          = yes_no(streaming_movies)
    total_services      = (phone_val + multiple_val + sec_val +
                           backup_val + device_val + tech_val +
                           tv_val + movies_val)

    row = {
        "gender":              0,
        "SeniorCitizen":       1 if senior else 0,
        "Partner":             yes_no(partner),
        "Dependents":          yes_no(dependents),
        "tenure":              tenure,
        "PhoneService":        phone_val,
        "MultipleLines":       multiple_val,
        "InternetService":     internet_map[internet],
        "OnlineSecurity":      sec_val,
        "OnlineBackup":        backup_val,
        "DeviceProtection":    device_val,
        "TechSupport":         tech_val,
        "StreamingTV":         tv_val,
        "StreamingMovies":     movies_val,
        "Contract":            contract_map[contract],
        "PaperlessBilling":    yes_no(paperless),
        "PaymentMethod":       payment_map[payment],
        "MonthlyCharges":      monthly_charges,
        "TotalCharges":        total_charges,
        "charges_per_tenure":  charges_per_tenure,
        "is_new_customer":     is_new_customer,
        "total_services":      total_services
    }
    return pd.DataFrame([row])[feature_names]

def main():
    st.set_page_config(
        page_title="Customer Churn Predictor",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Customer Churn Predictor")
    st.caption("Predict whether a customer is likely to leave — powered by Machine Learning")
    st.markdown("---")

    model, scaler, feature_names = load_model()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Contract & Billing")
        contract  = st.selectbox("Contract type",
                                 ["Month-to-month", "One year", "Two year"])
        internet  = st.selectbox("Internet service",
                                 ["DSL", "Fiber optic", "No"])
        payment   = st.selectbox("Payment method",
                                 ["Electronic check", "Mailed check",
                                  "Bank transfer", "Credit card"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])

    with col2:
        st.subheader("Usage & Charges")
        tenure          = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly charges ($)", 18, 120, 65)
        st.info(f"Estimated total charges: **${monthly_charges * tenure:,.0f}**")
        phone    = st.selectbox("Phone service", ["Yes", "No"])
        multiple = st.selectbox("Multiple lines",  ["Yes", "No"])

    with col3:
        st.subheader("Customer Profile")
        senior     = st.checkbox("Senior citizen")
        partner    = st.selectbox("Partner",    ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        online_sec    = st.selectbox("Online security",   ["Yes", "No"])
        online_backup = st.selectbox("Online backup",     ["Yes", "No"])
        device_prot   = st.selectbox("Device protection", ["Yes", "No"])
        tech_support  = st.selectbox("Tech support",      ["Yes", "No"])
        streaming_tv  = st.selectbox("Streaming TV",      ["Yes", "No"])
        streaming_mov = st.selectbox("Streaming movies",  ["Yes", "No"])

    st.markdown("---")

    if st.button("🔮 Predict Churn Risk", use_container_width=True):
        input_df = build_input(
            tenure, monthly_charges, contract, internet,
            senior, partner, dependents, phone, paperless,
            payment, online_sec, online_backup, device_prot,
            tech_support, streaming_tv, streaming_mov,
            multiple, feature_names
        )

        input_scaled = scaler.transform(input_df)
        prob         = model.predict_proba(input_scaled)[0][1]
        prediction   = model.predict(input_scaled)[0]

        st.subheader("Prediction Result")

        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Churn Probability", f"{prob:.1%}")
        with res_col2:
            st.metric("Tenure", f"{tenure} months")
        with res_col3:
            st.metric("Monthly Charges", f"${monthly_charges}")

        if prob >= 0.6:
            st.error(f"HIGH CHURN RISK — {prob:.1%} probability of leaving")
            st.warning("Recommended action: Offer a discount or upgrade to a longer contract immediately.")
        elif prob >= 0.35:
            st.warning(f"MEDIUM CHURN RISK — {prob:.1%} probability of leaving")
            st.info("Recommended action: Send a satisfaction survey and monitor activity.")
        else:
            st.success(f"LOW CHURN RISK — {prob:.1%} probability of leaving")
            st.info("This customer appears satisfied. Standard retention activities apply.")

        st.markdown("---")
        st.subheader("What drives churn? — SHAP Feature Importance")
        st.caption("This chart shows which features have the most impact on churn predictions overall.")
        if os.path.exists(SHAP_PLOT_PATH):
            st.image(SHAP_PLOT_PATH, use_container_width=True)
        else:
            st.info("SHAP plot not found. Run fix_shap.py to generate it.")

    st.markdown("---")
    st.caption("Built with Python, scikit-learn, MLflow, SHAP and Streamlit")

if __name__ == "__main__":
    main()
