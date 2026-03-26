import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import NumpySerializer
from sagemaker.deserializers import NumpyDeserializer

from imblearn.pipeline import Pipeline
import shap
import importlib

# ── Setup ──────────────────────────────────────────────────────────────────────
warnings.simplefilter("ignore")

# FIX: On Streamlit Cloud the app runs from:
#   /mount/src/stock_prediction/Portfolio/StreamlitApp_HW4.py
# The src/ folder lives one level up at:
#   /mount/src/stock_prediction/src/
# So we must add the PARENT of the Portfolio/ folder to sys.path dynamically.
# This also works on SageMaker where the structure is:
#   /home/ec2-user/SageMaker/HW 4/StreamlitApp_HW4.py
#   /home/ec2-user/SageMaker/src/
current_dir  = os.path.dirname(os.path.abspath(__file__))   # .../Portfolio/
project_root = os.path.abspath(os.path.join(current_dir, '..'))  # one level up

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# FIX: Import PairFeatureEngineer directly — it is needed by the loaded pipeline
# at inference time (it is step 0 inside the saved .joblib file).
# extract_features_pair is NOT imported — confirmed absent from feature_utils.py.
import src.Custom_Classes
import src.feature_utils
importlib.reload(src.Custom_Classes)
importlib.reload(src.feature_utils)
from src.Custom_Classes import FeatureEngineer, PairFeatureEngineer

# ── AWS Secrets ────────────────────────────────────────────────────────────────
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]

# Pulled directly from Cell 61 and Cell 67 of YOUR template
aws_bucket   = st.secrets["aws_credentials"].get("AWS_BUCKET",   "hayden-dodge-s3-bucket")
aws_endpoint = st.secrets["aws_credentials"].get("AWS_ENDPOINT", "logistic-pipeline-endpoint-auto-6")

# ── AWS Session ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session    = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

# ── Pair Configuration ─────────────────────────────────────────────────────────
# Matches Cell 34 of the template:  target_ticker = 'JPM'
# Matches Cell 39 of the template:  valid_partner = valid_partners[0][0]
# Matches Cell 41 of the template:  data_prediction = dataset[[valid_partner, target_ticker]]
# Matches Cell 46 of the template:  X = data_prediction[[valid_partner, target_ticker]]
#
# !! IMPORTANT: Replace PARTNER_TICKER with whatever your notebook printed for valid_partner !!
# Run this in your notebook to confirm:  print(f'Selected pair: {target_ticker} <--> {valid_partner}')
TARGET_TICKER  = "JPM"
PARTNER_TICKER = "GS"    # <-- REPLACE with your actual valid_partner from the notebook output

# Partner goes FIRST — matches the column order in Cell 46
COLUMN_ORDER = [PARTNER_TICKER, TARGET_TICKER]

MODEL_INFO = {
    "endpoint" : aws_endpoint,
    "explainer": "explainer_pair.shap",
    "pipeline" : "finalized_pair_model.tar.gz",
    "keys"     : COLUMN_ORDER,
    "inputs"   : [
        {"name": PARTNER_TICKER, "min": 0.0, "default": 100.0, "step": 1.0},
        {"name": TARGET_TICKER,  "min": 0.0, "default": 100.0, "step": 1.0},
    ]
}

# ── Load Pipeline from S3 ──────────────────────────────────────────────────────
# Matches Cell 64 of the template:
#   s3_path_key = 'sklearn-pipeline-deployment'
#   filename    = 'finalized_pair_model.tar.gz'
@st.cache_resource
def load_pipeline(_session, bucket):
    s3_client = _session.client('s3')
    filename  = MODEL_INFO["pipeline"]

    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=f"sklearn-pipeline-deployment/{os.path.basename(filename)}"
    )

    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]

    return joblib.load(joblib_file)


# ── Load SHAP Explainer from S3 ────────────────────────────────────────────────
# Matches Cell 63 of the template:
#   Key = "explainer/explainer_pair.shap"
@st.cache_resource
def load_shap_explainer(_session, bucket):
    s3_client  = _session.client('s3')
    local_path = os.path.join(tempfile.gettempdir(), "explainer_pair.shap")

    if not os.path.exists(local_path):
        s3_client.download_file(
            Filename=local_path,
            Bucket=bucket,
            Key="explainer/explainer_pair.shap"
        )

    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)


# ── Base Historical Price Data ─────────────────────────────────────────────────
# Replicates Cells 10, 37, 41 of the template:
#   dataset = read_csv('./SP500Data.csv', index_col=0)
#   dataset.fillna(dataset.mean(), inplace=True)
#   data_prediction = dataset[[valid_partner, target_ticker]]
#
# PairFeatureEngineer(window=60) is step 0 of the pipeline and needs >= 60 rows
# of raw price history before it can produce features — so we pass the last 120 rows.
@st.cache_data(ttl=300)
def load_base_data():
    try:
        dataset = pd.read_csv('./SP500Data.csv', index_col=0)
        dataset.fillna(dataset.mean(), inplace=True)   # matches Cell 37
        pair_df = dataset[COLUMN_ORDER].copy()         # matches Cell 41
        return pair_df.iloc[-120:]                     # 2× window=60 for safety
    except FileNotFoundError:
        st.warning("SP500Data.csv not found — using single-row input only.")
        return pd.DataFrame(columns=COLUMN_ORDER)


# ── Prediction via SageMaker Endpoint ─────────────────────────────────────────
def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer()
    )
    try:
        raw_pred = predictor.predict(input_df.values)
        pred_val = pd.DataFrame(raw_pred).values[-1][0]
        # Matches Cell 43 of the template:
        #   choices = [1, -1]  with default=0
        #   1 = BUY, -1 = SELL, 0 = HOLD
        mapping = {-1: "SELL", 0: "HOLD", 1: "BUY"}
        return mapping.get(int(pred_val), str(pred_val)), 200
    except Exception as e:
        return f"Error: {str(e)}", 500


# ── SHAP Explanation ───────────────────────────────────────────────────────────
def display_explanation(input_df):
    """
    Matches Cells 58-59 of the template exactly:

      Cell 58:
        preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-2])
        X_train_transformed    = preprocessing_pipeline.transform(X_train)
        explainer = shap.Explainer(model, X_train_transformed)

      Cell 59:
        X_test_transformed = preprocessing_pipeline.transform(X_test)
        feature_names      = best_pipeline[1:4].get_feature_names_out()
        X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names)
        shap_values        = explainer(X_test_transformed)
        shap.plots.waterfall(shap_values[0, :, 0])
        shap.plots.bar(shap_values[:, :, 0], max_display=10)
    """
    full_pipeline = load_pipeline(session, aws_bucket)
    explainer     = load_shap_explainer(session, aws_bucket)

    # steps[:-2] drops ('sampler', SMOTE) and ('model', LogisticRegression)
    # leaving: pair_ind_5, imputer, scaler, feature_selection
    preprocessing_pipeline = Pipeline(steps=full_pipeline.steps[:-2])
    input_transformed      = preprocessing_pipeline.transform(input_df)

    # [1:4] = imputer, scaler, feature_selection  (index 0 is PairFeatureEngineer)
    feature_names     = full_pipeline[1:4].get_feature_names_out()
    input_transformed = pd.DataFrame(input_transformed, columns=feature_names)

    shap_values = explainer(input_transformed)

    st.subheader("🔍 Decision Transparency (SHAP)")

    # Waterfall — Cell 59 line 1
    fig1, _ = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0, :, 0], show=False)
    st.pyplot(fig1)

    # Bar summary — Cell 59 line 2
    fig2, _ = plt.subplots(figsize=(10, 4))
    shap.plots.bar(shap_values[:, :, 0], max_display=10, show=False)
    st.pyplot(fig2)

    top_feature = (
        pd.Series(
            shap_values[0, :, 0].values,
            index=shap_values[0, :, 0].feature_names
        )
        .abs()
        .idxmax()
    )
    st.info(
        f"**Business Insight:** The most influential factor in this prediction was **{top_feature}**."
    )


# ── Streamlit UI ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Pair Trading — HW4", layout="wide")
st.title("📈 Pairs Trading Signal Predictor")
st.subheader(f"Pair: {PARTNER_TICKER}  ↔  {TARGET_TICKER}  |  Signals: BUY / HOLD / SELL")
st.markdown(
    "Enter today's **closing prices** for both stocks. "
    "The model predicts whether to **BUY (1)**, **HOLD (0)**, or **SELL (-1)** "
    "the pair based on the cointegrated strategy trained in HW4."
)

with st.form("pred_form"):
    st.subheader("Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                inp["name"],
                min_value=float(inp["min"]),
                value=float(inp["default"]),
                step=float(inp["step"]),
                help=f"Today's closing price for {inp['name']}"
            )

    submitted = st.form_submit_button("▶ Run Prediction")

if submitted:
    # Column order: [PARTNER_TICKER, TARGET_TICKER] — matches Cell 46
    new_row = [user_inputs[k] for k in COLUMN_ORDER]

    base_df = load_base_data()

    if base_df.empty:
        input_df = pd.DataFrame([new_row], columns=COLUMN_ORDER)
    else:
        input_df = pd.concat(
            [base_df, pd.DataFrame([new_row], columns=COLUMN_ORDER)],
            ignore_index=True
        )

    with st.spinner("Calling SageMaker endpoint..."):
        res, status = call_model_api(input_df)

    if status == 200:
        color = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}.get(res, "")
        st.metric("Prediction Signal", f"{color} {res}")

        with st.spinner("Computing SHAP explanation..."):
            display_explanation(input_df)
    else:
        st.error(res)
