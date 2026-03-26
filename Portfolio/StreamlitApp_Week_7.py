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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import shap

# ── Setup ──────────────────────────────────────────────────────────────────────
warnings.simplefilter("ignore")

# CHANGED: path resolution updated to match the HW4 folder structure
# src/ lives at /home/ec2-user/SageMaker/src, notebook is in HW 4/
# For Streamlit Cloud the project root containing src/ must be on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))  # one level up = SageMaker/
if project_root not in sys.path:
    sys.path.append(project_root)

# CHANGED: removed extract_features_pair import — that function does not exist in
# src/feature_utils.py (confirmed from HW4 template). Feature engineering is
# handled inline below using PairFeatureEngineer from src.Custom_Classes.
import importlib
import src.Custom_Classes
import src.feature_utils
importlib.reload(src.Custom_Classes)
importlib.reload(src.feature_utils)
from src.Custom_Classes import PairFeatureEngineer

# ── AWS Secrets ────────────────────────────────────────────────────────────────
aws_id       = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret   = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token    = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
# CHANGED: bucket and endpoint now match the values hard-coded in the HW4 template
aws_bucket   = st.secrets["aws_credentials"].get("AWS_BUCKET",   "franck-soh-s3-bucket")
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

# ── Model / Pair Configuration ─────────────────────────────────────────────────
# CHANGED: keys updated to match the HW4 template pair
#   target_ticker = 'JPM'   (was 'AAPL' in Week 7 — forbidden by the assignment)
#   valid_partner  = cointegrated partner found in the notebook (first in sorted list)
#   The column order in X is  [valid_partner, target_ticker]  — matching:
#       X = data_prediction[[valid_partner, target_ticker]]  (Cell 46 of template)
TARGET_TICKER = "JPM"
# Replace PARTNER_TICKER with whichever ticker was printed by
#   print(f'Selected pair: {target_ticker}  <-->  {valid_partner}')
# in your notebook before you run Streamlit.
PARTNER_TICKER = "GS"   # <-- UPDATE THIS to your actual cointegrated partner

MODEL_INFO = {
    "endpoint" : aws_endpoint,
    "explainer": "explainer_pair.shap",
    "pipeline" : "finalized_pair_model.tar.gz",
    # CHANGED: column order must match X = data_prediction[[valid_partner, target_ticker]]
    "keys"     : [PARTNER_TICKER, TARGET_TICKER],
    "inputs"   : [
        {"name": PARTNER_TICKER, "type": "number", "min": 0.0, "default": 100.0, "step": 1.0},
        {"name": TARGET_TICKER,  "type": "number", "min": 0.0, "default": 100.0, "step": 1.0},
    ]
}

# ── Pipeline / Explainer Loaders ───────────────────────────────────────────────
@st.cache_resource
def load_pipeline(_session, bucket, s3_key):
    """Download finalized_pair_model.tar.gz from S3, extract, and load the joblib pipeline."""
    s3_client = _session.client('s3')
    filename  = MODEL_INFO["pipeline"]

    s3_client.download_file(
        Filename=filename,
        Bucket=bucket,
        Key=f"{s3_key}/{os.path.basename(filename)}"
    )

    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]

    return joblib.load(joblib_file)


@st.cache_resource
def load_shap_explainer(_session, bucket, s3_key, local_path):
    """Download explainer_pair.shap from S3 (once) and load it."""
    s3_client = _session.client('s3')

    if not os.path.exists(local_path):
        s3_client.download_file(
            Filename=local_path,
            Bucket=bucket,
            Key=s3_key
        )

    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)


# ── Feature Engineering ────────────────────────────────────────────────────────
# CHANGED: replaced extract_features_pair() (non-existent function) with inline
# replication of what the notebook does in Cells 41-46:
#   data_prediction = dataset[[valid_partner, target_ticker]]
#   X = data_prediction[[valid_partner, target_ticker]]   (just the two price columns)
# PairFeatureEngineer(window=60) is the FIRST step in the pipeline, so we only
# need to pass the raw two-column price DataFrame — the pipeline handles the rest.

@st.cache_data(ttl=300)
def load_base_features():
    """
    Load the last 60+ rows of SP500Data.csv for the pair so that
    PairFeatureEngineer(window=60) inside the pipeline has enough history.
    Returns a DataFrame with columns [PARTNER_TICKER, TARGET_TICKER].
    """
    try:
        dataset = pd.read_csv('./SP500Data.csv', index_col=0)
        # Keep only the two pair columns in the correct order
        pair_df = dataset[[PARTNER_TICKER, TARGET_TICKER]].copy()
        pair_df.fillna(pair_df.mean(), inplace=True)
        # Return the last 120 rows — enough for the window=60 feature engineer
        return pair_df.iloc[-120:]
    except FileNotFoundError:
        # Fallback: return an empty DataFrame with correct columns
        return pd.DataFrame(columns=[PARTNER_TICKER, TARGET_TICKER])


# ── Prediction ─────────────────────────────────────────────────────────────────
def call_model_api(input_df):
    """Send input_df to the SageMaker endpoint and return (signal_label, status_code)."""
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer()
    )

    try:
        raw_pred = predictor.predict(input_df.values)
        pred_val = pd.DataFrame(raw_pred).values[-1][0]
        # CHANGED: mapping matches the template's signal encoding exactly
        # Cell 43: choices = [1, -1]  ->  1=BUY, -1=SELL, 0=HOLD
        mapping = {-1: "SELL", 0: "HOLD", 1: "BUY"}
        return mapping.get(int(pred_val), str(pred_val)), 200
    except Exception as e:
        return f"Error: {str(e)}", 500


# ── SHAP Explanation ───────────────────────────────────────────────────────────
def display_explanation(input_df):
    """
    Load the pipeline + SHAP explainer, transform input, and render waterfall plot.
    CHANGED: preprocessing slice matches the template exactly —
        preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[:-2])
        feature_names = best_pipeline[1:4].get_feature_names_out()
    (steps[:-2] drops SMOTE + LogisticRegression; [1:4] = imputer/scaler/feature_selection)
    """
    explainer_name = MODEL_INFO["explainer"]
    local_explainer = os.path.join(tempfile.gettempdir(), explainer_name)

    explainer     = load_shap_explainer(
        session, aws_bucket,
        posixpath.join('explainer', explainer_name),
        local_explainer
    )
    full_pipeline = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')

    # CHANGED: matches template Cell 58/59 exactly
    preprocessing_pipeline = Pipeline(steps=full_pipeline.steps[:-2])
    input_transformed      = preprocessing_pipeline.transform(input_df)
    feature_names          = full_pipeline[1:4].get_feature_names_out()
    input_transformed      = pd.DataFrame(input_transformed, columns=feature_names)

    shap_values = explainer(input_transformed)

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0, :, 0], show=False)
    st.pyplot(fig)

    top_feature = (
        pd.Series(shap_values[0, :, 0].values, index=shap_values[0, :, 0].feature_names)
        .abs()
        .idxmax()
    )
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")


# ── Streamlit UI ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Pair Trading — JPM", layout="wide")
st.title("📈 Pairs Trading Signal Predictor")
# CHANGED: subtitle reflects the actual pair from the HW4 template
st.subheader(f"Pair: {PARTNER_TICKER}  ↔  {TARGET_TICKER}  |  Signal: BUY / HOLD / SELL")

st.markdown(
    "Enter the **current closing prices** for both stocks. "
    "The model will predict whether to **BUY**, **HOLD**, or **SELL** the pair today."
)

with st.form("pred_form"):
    st.subheader("Stock Price Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'],
                min_value=inp['min'],
                value=inp['default'],
                step=inp['step'],
                help=f"Today's closing price for {inp['name']}"
            )

    submitted = st.form_submit_button("▶ Run Prediction")

if submitted:
    # Build the input row in the correct column order: [PARTNER_TICKER, TARGET_TICKER]
    new_row = [user_inputs[k] for k in MODEL_INFO["keys"]]

    # Append the new row to the base historical data so PairFeatureEngineer has context
    base_df  = load_base_features()

    if base_df.empty:
        # If CSV not found, just use the single input row
        input_df = pd.DataFrame([new_row], columns=MODEL_INFO["keys"])
    else:
        input_df = pd.concat(
            [base_df, pd.DataFrame([new_row], columns=base_df.columns)],
            ignore_index=True
        )

    with st.spinner("Calling model endpoint..."):
        res, status = call_model_api(input_df)

    if status == 200:
        # Colour-code the signal
        color_map = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}
        st.metric("Prediction Signal", f"{color_map.get(res, '')} {res}")

        with st.spinner("Loading SHAP explanation..."):
            display_explanation(input_df)
    else:
        st.error(res)