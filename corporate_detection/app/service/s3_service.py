import json
import boto3
from botocore.exceptions import ClientError
from typing import Dict, Any


class S3Service:
    def __init__(
        self,
        bucket_name: str,
        aws_access_key: str = None,
        aws_secret_key: str = None,
        region_name: str = "us-east-1"
    ):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region_name
        )

    def upload_json(self, data: Dict[str, Any], s3_key: str) -> bool:
        try:
            json_bytes = json.dumps(data, indent=2).encode("utf-8")
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_bytes,
                ContentType="application/json"
            )
            print(f"[S3 Upload Success] Dumped to s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            print(f"[S3 Upload Error] Failed to upload JSON: {e}")
            return False