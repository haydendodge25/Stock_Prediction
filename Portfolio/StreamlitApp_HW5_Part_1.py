import os, sys, warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import posixpath
import json 

import joblib
import tarfile
import tempfile

import boto3
import sagemaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer 
from sagemaker.deserializers import JSONDeserializer

from sklearn.pipeline import Pipeline
import shap

warnings.simplefilter("ignore")

# Access the secrets
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": 'explainer_pca.shap', 
    "pipeline": 'finalized_pca_model.tar.gz', 
    "keys": ["IBM_CR_Cum","NVDA_CR_Cum"], 
    "inputs": [{"name": k, "type": "number", "min": -100.0, "max": 100.0, "default": 0.0, "step": 10.0} for k in ["IBM_CR_Cum","NVDA_CR_Cum"]] 
}

# Built-in replacement for convert_input_pca_regression
def convert_input_pca_regression(raw_json_input, content_type):
    payload = json.loads(raw_json_input)
    keys = list(payload.keys())

    dataset = pd.read_csv('./SP500Data.csv', index_col=0)

    return_period = 5
    target_stock = 'AMZN'

    X = np.log(dataset.drop([target_stock], axis=1)).diff(return_period)
    X = np.exp(X).cumsum()
    X.columns = [name + '_CR_Cum' for name in X.columns]

    val0 = float(payload[keys[0]])
    val1 = float(payload[keys[1]])

    distances = np.sqrt((X[keys[0]] - val0)**2 + (X[keys[1]] - val1)**2)
    closest_idx = distances.idxmin()
    closest_row = X.loc[[closest_idx]].copy()
    closest_row[keys[0]] = val0
    closest_row[keys[1]] = val1

    return closest_row

@st.cache_resource
def get_session(aws_id, aws_secret, aws_token):
    return boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name='us-east-1'
    )

session = get_session(aws_id, aws_secret, aws_token)
sm_session = sagemaker.Session(boto_session=session)

def load_pipeline(_session, bucket, key):
    s3_client = _session.client('s3')
    filename = MODEL_INFO["pipeline"]
    s3_client.download_file(
        Filename=filename, 
        Bucket=bucket, 
        Key=f"{key}/{os.path.basename(filename)}")
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")
        joblib_file = [f for f in tar.getnames() if f.endswith('.joblib')][0]
    return joblib.load(f"{joblib_file}")

def load_shap_explainer(_session, bucket, key, local_path):
    s3_client = _session.client('s3')
    if not os.path.exists(local_path):
        s3_client.download_file(Filename=local_path, Bucket=bucket, Key=key)
    with open(local_path, "rb") as f:
        return shap.Explainer.load(f)

def call_model_api(input_df):
    predictor = Predictor(
        endpoint_name=MODEL_INFO["endpoint"],
        sagemaker_session=sm_session,
        serializer=JSONSerializer(), 
        deserializer=JSONDeserializer()
    )
    try:
        raw_pred = predictor.predict(input_df)
        pred_val = raw_pred["predictions"][0]
        return round(float(pred_val), 4), 200
    except Exception as e:
        return f"Error: {str(e)}", 500

def display_explanation(input_df, session, aws_bucket):
    explainer_name = MODEL_INFO["explainer"]
    explainer = load_shap_explainer(
        session, aws_bucket,
        posixpath.join('explainer', explainer_name),
        os.path.join(tempfile.gettempdir(), explainer_name)
    )

    raw_json_input = json.dumps(input_df)
    input_df = convert_input_pca_regression(raw_json_input, 'application/json')

    best_pipeline = load_pipeline(session, aws_bucket, 'sklearn-pipeline-deployment')
    
    preprocessing_pipeline = Pipeline(steps=best_pipeline.steps[0:2]) 
    input_df_transformed = preprocessing_pipeline.transform(input_df) 
    feature_names = ['KernelPC_%d' % (i+1) for i in range(input_df_transformed.shape[1])]
    input_df_transformed = pd.DataFrame(input_df_transformed, columns=feature_names) 
    shap_values = explainer(input_df_transformed) 

    st.subheader("🔍 Decision Transparency (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(shap_values[0], max_display=10)
    st.pyplot(fig)
    top_feature = pd.Series(shap_values[0].values, index=shap_values[0].feature_names).abs().idxmax()
    st.info(f"**Business Insight:** The most influential factor in this decision was **{top_feature}**.")

# Streamlit UI
st.set_page_config(page_title="ML Deployment", layout="wide")
st.title("👨‍💻 ML Deployment")

with st.form("pred_form"):
    st.subheader(f"Inputs")
    cols = st.columns(2)
    user_inputs = {}
    
    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp['name']] = st.number_input(
                inp['name'].replace('_', ' ').upper(),
                min_value=inp['min'], max_value=inp['max'], value=inp['default'], step=inp['step']
            )
    
    submitted = st.form_submit_button("Run Prediction")

if submitted:
    res, status = call_model_api(user_inputs)
    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(user_inputs, session, aws_bucket)
    else:
        st.error(res)
