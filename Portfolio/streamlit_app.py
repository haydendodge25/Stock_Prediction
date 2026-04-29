# Replace the sagemaker import and get_predictor() with this:
import boto3, json

@st.cache_resource
def get_sagemaker_client():
    creds = st.secrets["aws_credentials"]
    return boto3.client(
        "sagemaker-runtime",
        aws_access_key_id=creds["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=creds["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=creds["AWS_SESSION_TOKEN"],
        region_name="us-east-1",
    )

# Then in your predict button:
client = get_sagemaker_client()
response = client.invoke_endpoint(
    EndpointName=st.secrets["aws_credentials"]["AWS_ENDPOINT"],
    ContentType="application/x-npy",  # or "text/csv" depending on your endpoint
    Body=feature_vector.tobytes(),
)
prob = float(np.frombuffer(response['Body'].read(), dtype=np.float32)[0])
