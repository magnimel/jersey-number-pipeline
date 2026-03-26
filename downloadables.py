import os
import gdown
import zipfile
import shutil
import sys

base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, 'data')
models_path = os.path.join(base_path, 'models')
os.makedirs(data_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)


def is_valid_checkpoint(path):
    """Return True if the file looks like a real model checkpoint (not an HTML error page)."""
    if not os.path.isfile(path):
        return False
    if os.path.getsize(path) < 1024 * 100:  # anything under 100KB is certainly wrong
        return False
    with open(path, 'rb') as f:
        header = f.read(4)
    # HTML pages start with '<'; JSON error responses start with '{'
    # Valid formats: pickle (\x80), ZIP/PyTorch-new (PK = \x50\x4b)
    return header[0:1] not in (b'<', b'{')


def is_valid_zip(path):
    if not os.path.isfile(path):
        return False
    return zipfile.is_zipfile(path)


def download_file(file_id, dest_path, label, validate_fn):
    """Download a file from Google Drive, re-downloading if the existing file is invalid."""
    if os.path.isfile(dest_path):
        if validate_fn(dest_path):
            print(f"  OK (already present): {label}")
            return
        else:
            print(f"  CORRUPT — removing and re-downloading: {label}")
            os.remove(dest_path)

    print(f"  Downloading: {label}")
    url = f'https://drive.google.com/uc?id={file_id}'
    try:
        gdown.download(url, dest_path, quiet=False)
        if not validate_fn(dest_path):
            print(f"  WARNING: {label} still looks invalid after download (quota exceeded?)")
            os.remove(dest_path)
    except Exception as e:
        print(f"  ERROR downloading {label}: {e}")
        if os.path.isfile(dest_path):
            os.remove(dest_path)


def unzip_if_needed(zip_path, extract_to, sentinel=None):
    """
    Unzip zip_path into extract_to, but only if:
    - The sentinel path doesn't exist yet (sentinel = a file/dir that appears after extraction), OR
    - No sentinel is given and the zip hasn't been extracted.
    Skips if the zip itself is missing/corrupt.
    """
    if not is_valid_zip(zip_path):
        print(f"  Skipping unzip (missing or corrupt): {os.path.basename(zip_path)}")
        return
    if sentinel and os.path.exists(sentinel):
        print(f"  Already extracted: {os.path.basename(zip_path)}")
        return
    print(f"  Unzipping {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)


# ---------------------------------------------------------------------------
# 1. Jersey-2023 dataset splits
# ---------------------------------------------------------------------------
jersey_data_path = os.path.join(data_path, 'jersey-2023')
soccernet_path = os.path.join(data_path, 'SoccerNet')
# If already renamed from a previous run, point at SoccerNet directly
active_jersey_path = soccernet_path if os.path.isdir(soccernet_path) else jersey_data_path
os.makedirs(active_jersey_path, exist_ok=True)

jersey_splits = [
    ('119w2kiO5pqw1avKlSdoL5RWpWuD5McBP', 'train.zip',     'train/images'),
    ('1dJ9V6rM-E4x51g5HbkmOYVp9nCf1Bb0w', 'test.zip',      'test/images'),
    ('12XKUXPlD1Tm03BdnIuS_9C0P76oCIcWL', 'challenge.zip', 'challenge/images'),
]

print("--- Jersey-2023 splits ---")
for file_id, filename, sentinel_rel in jersey_splits:
    out = os.path.join(active_jersey_path, filename)
    download_file(file_id, out, filename, is_valid_zip)
    sentinel = os.path.join(active_jersey_path, sentinel_rel)
    unzip_if_needed(out, active_jersey_path, sentinel=sentinel)

# Rename jersey-2023 -> SoccerNet if not done yet
if os.path.isdir(jersey_data_path) and not os.path.isdir(soccernet_path):
    os.rename(jersey_data_path, soccernet_path)
    print("Renamed data/jersey-2023 -> data/SoccerNet")
    active_jersey_path = soccernet_path

# ---------------------------------------------------------------------------
# 2. Model checkpoints
# ---------------------------------------------------------------------------
models_files = [
    ('1VKaK2tJI7e-4j7NTWNuLS2ABo6DuUf9g', 'parseq-bb5792a6.pt'),
    ('1DULUhorGHsozOumtSocon0V-kbKwFCWG', 'parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt'),
    ('1QDAqZvIbf0UPP9disdBsqcdIB0e84ZWa', 'legibility_resnet34_soccer_20240215.pth'),
    ('1qki2Ah0xFfbqVE64wwYTy4Ow1RBoTEWW', 'market1501_resnet50_256_128_epoch_120.ckpt'),
    ('1IZghoJUMiq1NeXmPz4fOJBiEeeCjIbtL', 'dukemtmcreid_resnet50_256_128_epoch_120.ckpt'),
]

print("\n--- Model checkpoints ---")
for file_id, filename in models_files:
    download_file(file_id, os.path.join(models_path, filename), filename, is_valid_checkpoint)

# ---------------------------------------------------------------------------
# 3. Extra data files
# ---------------------------------------------------------------------------
data_files = [
    ('1_EA2lUBYx0TY-Ul0OJc7GTw1Y8kfLKvm', 'SoccerNetLegibility.zip', 'SoccerNetLegibility'),
    ('17A4HSSf7IcdrUcj9h_D_T3NEGfH2jUBm', 'soccer_lmdb.zip',         'lmdb'),
]

print("\n--- Extra data files ---")
for file_id, filename, sentinel_rel in data_files:
    out = os.path.join(data_path, filename)
    download_file(file_id, out, filename, is_valid_zip)
    sentinel = os.path.join(data_path, sentinel_rel)
    unzip_if_needed(out, data_path, sentinel=sentinel)

# ---------------------------------------------------------------------------
# 4. Copy reid checkpoints into centroids-reid/models/
# ---------------------------------------------------------------------------
print("\n--- Reid model copy ---")
reid_dst = os.path.join(base_path, 'reid', 'centroids-reid', 'models')
os.makedirs(reid_dst, exist_ok=True)
for fname in ['market1501_resnet50_256_128_epoch_120.ckpt',
              'dukemtmcreid_resnet50_256_128_epoch_120.ckpt']:
    src = os.path.join(models_path, fname)
    dst = os.path.join(reid_dst, fname)
    if not is_valid_checkpoint(src):
        print(f"  Source not ready (download may have failed): {fname}")
        continue
    if is_valid_checkpoint(dst):
        print(f"  OK (already present): {fname}")
    else:
        if os.path.isfile(dst):
            os.remove(dst)
        shutil.copy2(src, dst)
        print(f"  Copied {fname} -> {reid_dst}")

# Remove macOS metadata files that break os.listdir() checks in the pipeline
os.system('find . -name ".DS_Store" -type f -delete')

print("\nDone.")

sys.path.insert(0, os.path.join(base_path, 'reid', 'centroids-reid'))
os.environ['PYTHONPATH'] = (
    os.path.join(base_path, 'reid', 'centroids-reid')
    + ':' + os.environ.get('PYTHONPATH', '')
)
