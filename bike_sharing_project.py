import os
import zipfile
from pathlib import Path
import pandas as pd

# Constants
URL = "www.kaggle.com/datasets/hmavrodiev/london-bike-sharing-dataset"
DATASET = "london_merged.csv"
DATA_FOLDER = Path("datasets")
IMAGES_FOLDER = Path("images")

def download_and_extract(url: str, folder: Path):
    zip_file = folder / f"{url.split('/')[-1]}.zip"

    # Download the dataset if not already done
    if not zip_file.is_file():
        folder.mkdir(parents=True, exist_ok=True)
        os.system(f"kaggle datasets download -d {url.split('/')[-1]} -p {folder}")

    # Extract ZIP if not already extracted
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(folder)

def load_csv(file_name: str, folder: Path):
    return pd.read_csv(folder / file_name)

# Download, extract, and load the dataset
download_and_extract(URL, DATA_FOLDER)
df = load_csv(DATASET, DATA_FOLDER)

# Display the first few rows to confirm
df.head()

