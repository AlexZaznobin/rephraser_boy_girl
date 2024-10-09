import boto3
import os
from botocore.client import Config
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Yandex Object Storage endpoint
endpoint_url = 'https://storage.yandexcloud.net'

# Yandex credentials
access_key = os.environ.get("YANDEX_KEY_ID")
secret_key = os.environ.get("YANDEX_KEY")

# Initialize a session using Yandex Object Storage
session = boto3.session.Session()
s3_client = session.client(
    service_name='s3',
    endpoint_url=endpoint_url,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    config=Config(signature_version='s3v4')  # Specify the signature version
)

def upload_model_to_s3(local_directory, bucket_name, s3_directory):
    """
    Uploads the contents of a local directory to S3.

    :param local_directory: The path of the local directory containing the model files.
    :param bucket_name: The name of the S3 bucket.
    :param s3_directory: The S3 directory where the model will be uploaded.
    """
    for root, dirs, files in os.walk(local_directory):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, local_directory)
            s3_path = os.path.join(s3_directory, relative_path)

            # Upload the file to the given S3 path
            s3_client.upload_file(local_path, bucket_name, s3_path)
            print(f"Uploaded {local_path} to s3://{bucket_name}/{s3_path}")


# Function to download files from S3
def download_model_from_s3 (s3_bucket_name, s3_model_dir, local_model_dir) :
    """
    Download the model files from S3 to the local directory.

    :param s3_bucket_name: S3 bucket name.
    :param s3_model_dir: Path in S3 where the model files are stored.
    :param local_model_dir: Local directory where the files will be saved.
    """
    os.makedirs(local_model_dir, exist_ok=True)

    for obj in s3_client.list_objects_v2(Bucket=s3_bucket_name, Prefix=s3_model_dir)['Contents'] :
        s3_file_path = obj['Key']
        local_file_path = os.path.join(local_model_dir, os.path.relpath(s3_file_path, s3_model_dir))

        # Create any missing directories
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Download the file
        s3_client.download_file(s3_bucket_name, s3_file_path, local_file_path)
        print(f"Downloaded {s3_file_path} to {local_file_path}")