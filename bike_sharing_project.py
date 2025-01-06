import os
import zipfile

import pathlib
import sys
assert sys.version_info >= (3, 7)

from pathlib import Path
import tarfile
import urllib.request

from urllib.parse import urlparse
import pandas as pd
import numpy as np
URL = "www.kaggle.com/datasets/hmavrodiev/london-bike-sharing-dataset"
DATASET = "london_merged.csv"

DATEN_ORDNER = Path() / Path("datasets")
IMAGES_PATH = Path() / Path("images")
def load_data(url: str, ziel_ordner: Path):
    parsed_url = urlparse(url)
    path_parts = parsed_url.path.split('/')

    # Suchen nach dem Teil, der mit 'datasets' beginnt
    if "datasets" in path_parts:
        index = path_parts.index("datasets") + 1
        quelle = '/'.join(path_parts[index:len(path_parts)])
        zip_name = path_parts[len(path_parts)-1] + ".zip"
        # Pfad zur ZIP-Datei
        zip_file = ziel_ordner / Path(zip_name)

        if not zip_file.is_file():
            # Erstelle das Verzeichnis, falls es nicht existiert
            ziel_ordner.mkdir(parents=True, exist_ok=True)
            os.system(f"kaggle datasets download -d {quelle} -p {ziel_ordner.name}")

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Auflisten der Dateien im ZIP-Archiv
            file_list = zip_ref.namelist()
            for file_name in file_list:
                csv_file = ziel_ordner / Path(file_name)

                if not csv_file.is_file():
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                            zip_ref.extractall(path=ziel_ordner.name)
    else:
        print("ungueltige Url")

def pd_openCSVFile(csv_name: str, ordner: Path, encoding='utf-8'):
    csv_file = ordner / Path(csv_name)
    # Fuer die Voruntersuchung mittels panda, da heterogene Datentypen
    return pd.read_csv(csv_file,encoding=encoding)
def np_openCSVFile(csv_name: str, ordner: Path, encoding='utf-8'):
    csv_file = ordner / Path(csv_name)
    return np.genfromtxt(csv_file, delimiter=',', dtype=None, encoding=encoding, skip_header=1)
    # dtype=None: NumPy bestimmt den Datentyp automatisch .
    # encoding='utf-8': Daten werden korrekt gelesen, auch Sonderzeichen
    # skip_header=1: Überspringt die erste Zeile der Datei
def save_fig(fig_id: str, ziel_ordner: Path, tight_layout=True, fig_extension="png", resolution=300):
    if not ziel_ordner.is_dir():
        ziel_ordner.mkdir(parents=True, exist_ok=True)

    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        #anpassen des Layouts
        plt.tight_layout()
    #speichert die Datei, mit einer Aufloeseung dpi=resolution
    plt.savefig(path, format=fig_extension, dpi=resolution)
load_data(url=URL, ziel_ordner=DATEN_ORDNER)
df =  pd_openCSVFile(csv_name=DATASET, ordner=DATEN_ORDNER)
