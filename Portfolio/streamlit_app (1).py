"""
Loan Default Prediction – Streamlit Web App
Connects to the AWS SageMaker endpoint for real-time inference
and renders SHAP waterfall explanations.
"""

import streamlit as st
import numpy as np
import pandas as pd
import boto3
import json
import shap
import matplotlib.pyplot as plt
from joblib import load

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Loan Default Risk Predictor")
st.markdown(
    "Enter applicant details below. The model returns the **probability of default** "
    "and a SHAP explanation of which features drove the prediction."
)

# ── Load AWS credentials from Streamlit secrets ──────────────────────────────
@st.cache_resource
def get_sagemaker_client():
    creds = st.secrets["aws_credentials"]
    return boto3.client(
        "sagemaker-runtime",
        aws_access_key_id=creds["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=creds["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=creds["AWS_SESSION_TOKEN"] or None,
        region_name="us-east-1",
    )

@st.cache_resource
def get_explainer():
    creds = st.secrets["aws_credentials"]
    s3 = boto3.client(
        "s3",
        aws_access_key_id=creds["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=creds["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=creds["AWS_SESSION_TOKEN"] or None,
    )
    s3.download_file(creds["AWS_BUCKET"],
                     "explainer/loan_default_shap_explainer.joblib",
                     "/tmp/explainer.joblib")
    with open("/tmp/explainer.joblib", "rb") as f:
        explainer = load(f)
    return explainer

FEATURE_NAMES = [
    "loan_amnt", "int_rate", "installment", "dti", "fico_avg", "term",
    "annual_inc", "emp_length", "open_acc", "revol_bal", "revol_util",
    "pub_rec_bankruptcies", "mort_acc", "loan_income_ratio",
    "installment_income_ratio", "grade", "delinq_2yrs", "inq_last_6mths",
    "total_acc", "home_ownership", "verification_status",
]

GRADE_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
HOME_MAP  = {"RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 3}
VERIF_MAP = {"Not Verified": 0, "Source Verified": 1, "Verified": 2}

# ── Sidebar: input form ───────────────────────────────────────────────────────
st.sidebar.header("Loan Application Details")

loan_amnt  = st.sidebar.number_input("Loan Amount ($)", 500, 40000, 10000, step=500)
int_rate   = st.sidebar.slider("Interest Rate (%)", 5.0, 35.0, 12.0, step=0.1)
term       = st.sidebar.selectbox("Loan Term (months)", [36, 60])
installment = st.sidebar.number_input("Monthly Installment ($)", 10.0, 2000.0, 300.0, step=10.0)
grade      = st.sidebar.selectbox("Loan Grade", list(GRADE_MAP.keys()))

st.sidebar.markdown("---")
st.sidebar.subheader("Borrower Profile")
annual_inc = st.sidebar.number_input("Annual Income ($)", 10000, 500000, 60000, step=1000)
emp_length = st.sidebar.slider("Employment Length (years)", 0, 10, 5)
home_ownership = st.sidebar.selectbox("Home Ownership", list(HOME_MAP.keys()))
verification_status = st.sidebar.selectbox("Income Verification", list(VERIF_MAP.keys()))
dti        = st.sidebar.slider("Debt-to-Income Ratio (%)", 0.0, 50.0, 15.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("Credit History")
fico_low   = st.sidebar.slider("FICO Score Low",  600, 850, 690)
fico_high  = st.sidebar.slider("FICO Score High", 600, 850, 694)
revol_util = st.sidebar.slider("Revolving Utilization (%)", 0.0, 100.0, 45.0, step=0.5)
revol_bal  = st.sidebar.number_input("Revolving Balance ($)", 0, 200000, 10000, step=500)
open_acc   = st.sidebar.slider("Open Accounts", 0, 40, 10)
total_acc  = st.sidebar.slider("Total Accounts", 1, 80, 25)
delinq_2yrs     = st.sidebar.slider("Delinquencies (2 yrs)", 0, 10, 0)
inq_last_6mths  = st.sidebar.slider("Inquiries (last 6 mo)", 0, 10, 1)
pub_rec_bankruptcies = st.sidebar.slider("Public Record Bankruptcies", 0, 5, 0)
mort_acc   = st.sidebar.slider("Mortgage Accounts", 0, 20, 1)

# ── Derive engineered features ────────────────────────────────────────────────
fico_avg  = (fico_low + fico_high) / 2.0
loan_income_ratio        = loan_amnt / (annual_inc + 1)
installment_income_ratio = installment / (annual_inc + 1)
grade_encoded = GRADE_MAP[grade]
home_encoded  = HOME_MAP[home_ownership]
verif_encoded = VERIF_MAP[verification_status]

# Assemble feature vector
feature_vector = np.array([[
    loan_amnt, int_rate, installment, dti, fico_avg, term,
    annual_inc, emp_length, open_acc, revol_bal, revol_util,
    pub_rec_bankruptcies, mort_acc, loan_income_ratio,
    installment_income_ratio, grade_encoded, delinq_2yrs, inq_last_6mths,
    total_acc, home_encoded, verif_encoded,
]], dtype=np.float32)

# ── Predict ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Risk Score")
    if st.button("🔍  Predict Default Risk", type="primary"):
        with st.spinner("Querying SageMaker endpoint..."):
            try:
                client = get_sagemaker_client()
                body = json.dumps(feature_vector.flatten().tolist())
                response = client.invoke_endpoint(
                    EndpointName=st.secrets["aws_credentials"]["AWS_ENDPOINT"],
                    ContentType="application/json",
                    Body=body,
                )
                result = json.loads(response['Body'].read().decode('utf-8'))
                prob = float(result['default_probability'][0])
            except Exception as e:
                st.error(f"Endpoint error: {e}")
                prob = None

        if prob is not None:
            if prob < 0.30:
                risk_label, risk_color = "Low Risk", "green"
            elif prob < 0.55:
                risk_label, risk_color = "Medium Risk", "orange"
            else:
                risk_label, risk_color = "High Risk", "red"

            st.metric("Default Probability", f"{prob:.1%}")
            st.markdown(f"### :{risk_color}[{risk_label}]")

            if prob >= 0.55:
                st.error("⚠️ Recommend: Decline or manual underwriter review.")
            elif prob >= 0.30:
                st.warning("⚠️ Recommend: Manual review or adjusted terms.")
            else:
                st.success("✅ Recommend: Standard approval process.")

            # ── SHAP explanation ─────────────────────────────────────────────
            with col2:
                st.subheader("Prediction Explanation (SHAP)")
                with st.spinner("Computing SHAP values..."):
                    try:
                        explainer = get_explainer()
                        X_df = pd.DataFrame(feature_vector, columns=FEATURE_NAMES)
                        shap_vals = explainer(X_df)
                        fig, ax = plt.subplots()
                        shap.plots.waterfall(shap_vals[0], max_display=12, show=False)
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.warning(f"SHAP explanation unavailable: {e}")
    else:
        st.info("Fill in the sidebar and click **Predict Default Risk**.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Model: Tuned XGBoost | Pipeline: sklearn ColumnTransformer + SMOTE | "
    "Inference: AWS SageMaker | Explanations: SHAP TreeExplainer"
)
