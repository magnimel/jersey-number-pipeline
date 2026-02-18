from SoccerNet.Downloader import SoccerNetDownloader as SNdl
import os
import gdown
import zipfile
import sys
import os

mySNdl = SNdl(LocalDirectory="/content/jersey-number-pipeline/data")
mySNdl.downloadDataTask(task="jersey-2023", split=["train","test","challenge"])


# Define the base path
base_path = '/content/jersey-number-pipeline'

# --- Group 1: Models ---
models_path = os.path.join(base_path, 'models')
os.makedirs(models_path, exist_ok=True)

models_files = [
    # (File ID, Output Filename)
    ('1VKaK2tJI7e-4j7NTWNuLS2ABo6DuUf9g', 'parseq-bb5792a6.pt'),
    ('1DULUhorGHsozOumtSocon0V-kbKwFCWG', 'parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt'),
    ('1QDAqZvIbf0UPP9disdBsqcdIB0e84ZWa', 'legibility_resnet34_soccer_20240215.pth'),
    ('1qki2Ah0xFfbqVE64wwYTy4Ow1RBoTEWW', 'market1501_resnet50_256_128_epoch_120.ckpt'),
    ('1IZghoJUMiq1NeXmPz4fOJBiEeeCjIbtL', 'dukemtmcreid_resnet50_256_128_epoch_120.ckpt'),
]
print(f"Downloading models to {models_path}...")
for file_id, filename in models_files:
    output_file = os.path.join(models_path, filename)
    url = f'https://drive.google.com/uc?id={file_id}'
    try:
      gdown.download(url, output_file, quiet=False)
    except:
      print(f"Skipping {output_file}")

# --- Group 2: Data ---
data_path = os.path.join(base_path, 'data')
os.makedirs(data_path, exist_ok=True)

data_files = [
    # (File ID, Output Filename)
    ('1_EA2lUBYx0TY-Ul0OJc7GTw1Y8kfLKvm', 'SoccerNetLegibility.zip'),
    ('17A4HSSf7IcdrUcj9h_D_T3NEGfH2jUBm', 'soccer_lmdb.zip')
]

print(f"\nDownloading data to {data_path}...")
for file_id, filename in data_files:
    output_file = os.path.join(data_path, filename)
    url = f'https://drive.google.com/uc?id={file_id}'
    try:
      gdown.download(url, output_file, quiet=False)
    except:
      print(f"Skipping {output_file}")

print("\nAll downloads complete.")

# Unzip all files in the data directory
for item in os.listdir(data_path):
    if item.endswith(".zip"):
        file_name = os.path.join(data_path, item)
        print(f"Unzipping {item}...")
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            zip_ref.extractall(data_path)

jersey_data_path = '/content/jersey-number-pipeline/data/jersey-2023'
for item in os.listdir(jersey_data_path):
  if item.endswith('.zip'):
    file_name = os.path.join(jersey_data_path, item)
    print(f"Unzipping {item}...")
    with zipfile.ZipFile(file_name, 'r') as zip_ref:
        zip_ref.extractall(jersey_data_path)

print("Unzipping complete.")

sys.path.insert(0, '/content/jersey-number-pipeline/reid/centroids-reid')
# Add the directory containing the 'datasets' module to PYTHONPATH for the subprocess
os.environ['PYTHONPATH'] = '/content/jersey-number-pipeline/reid/centroids-reid/' + ':' + os.environ.get('PYTHONPATH', '')