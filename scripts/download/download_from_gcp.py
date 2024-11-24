import os
import pandas as pd
from tqdm import tqdm
import google
from google.cloud import storage

def download_from_gcp(bucket_name, file_name_prefix, destination):
    """
    Downloads files from Google Cloud Storage that match the file name prefix.
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=file_name_prefix))
    assert len(blobs) == 1, f"More than one file found for prefix {file_name_prefix}"
    for blob in blobs:
        destination_file_path = os.path.join(destination, blob.name.split('/')[-1])
        blob.download_to_filename(destination_file_path)

# destination = "/mnt/disks/eo-data/goes/"
# bucket_name = "iti-datasets-eo"
# subfolders = "goes-geoprocessed/2020"
# csv_file = "goes_2020_hourly_subset.csv"

destination = "/mnt/disks/eo-data/msg/"
bucket_name = "iti-datasets-eo"
subfolders = "msg-geoprocessed/2020"
csv_file = "msg_2020_hourly_subset.csv"

df = pd.read_csv(csv_file)
dates = df["datetime"].values

for date in tqdm(dates):
    file_name_prefix = f"{subfolders}/{date}"
    download_from_gcp(
        bucket_name=bucket_name, 
        file_name_prefix=file_name_prefix,
        destination=destination)