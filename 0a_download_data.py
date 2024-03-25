import os
import wget
import zipfile

# Create the "data/all_cts" directories if they don't exist.
all_cts_dir_path = os.path.join('.', 'data', 'all_cts')
if not os.path.exists(all_cts_dir_path):
    os.makedirs(all_cts_dir_path) 

# URL and save path for the ZIP file.
url = "https://classic.clinicaltrials.gov/AllAPIJSON.zip"
file_path = os.path.join(all_cts_dir_path, 'AllAPIJSON.zip')

# Download the ZIP file.
wget.download(url, out=file_path)

# Extract the contents of the ZIP file to the save path.
with zipfile.ZipFile(file_path, 'r') as zip_ref:
    zip_ref.extractall(all_cts_dir_path)