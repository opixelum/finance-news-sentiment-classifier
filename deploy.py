import boto3
import os
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION_NAME")
)

local_model_path: str = "models/model.tar.gz"
bucket_name: str = "opixelum"
remote_model_path: str = "finance-news-sentiments-classifier/models/model.tar.gz"

s3.upload_file(
    local_model_path,
    bucket_name,
    remote_model_path
)
