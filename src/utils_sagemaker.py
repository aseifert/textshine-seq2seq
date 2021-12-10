import boto3
import sagemaker
from dotenv import load_dotenv
from sagemaker.s3 import S3Downloader

load_dotenv()


def init_sagemaker(role_name: str, inside_sagemaker: bool = False):
    role = session = None

    if inside_sagemaker:
        session = sagemaker.Session()
        # sagemaker session bucket -> used for uploading data, models and logs
        # sagemaker will automatically create this bucket if it not exists
        sagemaker_session_bucket = None
        if sagemaker_session_bucket is None and session is not None:
            # set to default bucket if a bucket name is not given
            sagemaker_session_bucket = session.default_bucket()

        role = sagemaker.get_execution_role()
        session = sagemaker.Session(default_bucket=sagemaker_session_bucket)
    else:
        iam_client = boto3.client("iam")
        role = iam_client.get_role(RoleName=role_name)["Role"]["Arn"]
        session = sagemaker.Session()

    return role, session


def download_model(model, sess):
    S3Downloader.download(
        s3_uri=model.model_data,  # S3 URI where the trained model is located
        local_path=".",  # local path where *.tar.gz will be saved
        sagemaker_session=sess,  # Sagemaker session used for training the model
    )
