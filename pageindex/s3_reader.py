import os
import tempfile
import boto3
from botocore.exceptions import ClientError
from pageindex.env_settings import settings

def get_s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_REGION
    )

def list_doc_ids() -> list[str]:
    """Walks the cleaned/ prefix to find all doc_ids in the format cleaned/{stem}/{doc_id}/."""
    s3 = get_s3_client()
    doc_ids = []
    
    try:
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=settings.S3_BUCKET_NAME, Prefix=settings.S3_CLEANED_PREFIX, Delimiter='/')
        
        for page in pages:
            for prefix in page.get('CommonPrefixes', []):
                stem_prefix = prefix['Prefix']
                # List doc_ids under this stem
                sub_pages = paginator.paginate(Bucket=settings.S3_BUCKET_NAME, Prefix=stem_prefix, Delimiter='/')
                for sub_page in sub_pages:
                    for doc_prefix in sub_page.get('CommonPrefixes', []):
                        # Extracts the doc_id from the path e.g. cleaned/stem/doc_id/ -> doc_id
                        parts = doc_prefix['Prefix'].strip('/').split('/')
                        if len(parts) >= 3:
                            doc_ids.append(parts[-1])
                            
    except ClientError as e:
        print(f"Error listing S3 doc_ids: {e}")
        
    return doc_ids

def get_pdf_key(doc_id: str) -> str | None:
    s3 = get_s3_client()
    try:
        # We need to find the stem for this doc_id. 
        # Alternatively, search prefix using list_objects
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=settings.S3_BUCKET_NAME, Prefix=settings.S3_CLEANED_PREFIX)
        for page in pages:
            for obj in page.get('Contents', []):
                key = obj['Key']
                if f"/{doc_id}/" in key and key.endswith(".pdf"):
                    return key
    except ClientError as e:
        print(f"Error finding pdf key: {e}")
    return None

def download_s3_to_tempfile(key: str) -> str:
    """Downloads the file from S3 to a temporary file and returns its path."""
    s3 = get_s3_client()
    fd, temp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        s3.download_file(settings.S3_BUCKET_NAME, key, temp_path)
        return temp_path
    except ClientError as e:
        os.remove(temp_path)
        raise Exception(f"Failed to download S3 key {key}: {e}")
