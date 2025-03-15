import os
import boto3
from dotenv import load_dotenv
from sagemaker import Session
from sagemaker.huggingface import HuggingFaceModel


load_dotenv()

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION_NAME")
)

sagemaker_session = Session()

model_name: str = "finance-news-sentiments-classifier"
local_model_path: str = os.path.abspath("../models/model")
local_tar_path: str = os.path.abspath("../models/model/model.tar.gz")
bucket_name: str = "opixelum"
remote_model_path: str = f"{model_name}/models/model.tar.gz"
full_remote_model_path: str = f"s3://{bucket_name}/{remote_model_path}"

print("Initializing model...")
model = HuggingFaceModel(
    model_data=full_remote_model_path,
    role="arn:aws:iam::203918861682:role/opixelum",
    entry_point='entry_point.py',  # The script we defined above
    transformers_version='4.26.0',
    pytorch_version='1.13.1',
    py_version='py39',
    sagemaker_session=sagemaker_session
)

print("Deploying model...")
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.t2.medium",
    endpoint_name=model_name
)

print(f"Model deployed to endpoint: {predictor.endpoint_name}")