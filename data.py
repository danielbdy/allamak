from google.cloud import storage
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def get_data_from_bucket(file_name):
    """Retrieve data from a GCS bucket."""
    # Get the bucket name from environment variables
    bucket_name = os.getenv('BUCKET_NAME')

    # Initialize the GCS client
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    # Get blob (object) from the bucket
    blob = bucket.blob(file_name)

    # Download data from the blob
    data = blob.download_as_string()

    return data

