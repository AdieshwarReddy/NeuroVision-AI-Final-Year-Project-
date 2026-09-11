"""
Figshare Brain Tumor Dataset (Jun Cheng) Downloader and Extractor.
Fetches the 4 official zip archives from Figshare API (Article ID: 1512427)
and extracts 1.mat ... 3064.mat into data/raw_mat/.
"""

import os
import sys
import zipfile
import json
import urllib.request
from pathlib import Path
import requests
from tqdm import tqdm


FIGSHARE_ARTICLE_URL = "https://api.figshare.com/v2/articles/1512427"
DATA_DIR = Path("data")
RAW_MAT_DIR = DATA_DIR / "raw_mat"
ZIP_DIR = DATA_DIR / "zips"


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def get_figshare_file_list() -> list:
    """Queries Figshare API for the download URLs of Jun Cheng dataset files."""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = requests.get(FIGSHARE_ARTICLE_URL, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch Figshare metadata. HTTP Status: {resp.status_code}")
    data = resp.json()
    files = data.get("files", [])
    return files


def download_and_extract_all(target_mat_count: int = 3064):
    """Downloads all Figshare dataset zip parts and extracts .mat files."""
    RAW_MAT_DIR.mkdir(parents=True, exist_ok=True)
    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if raw_mat already has 3064 files
    existing_mats = list(RAW_MAT_DIR.glob("*.mat"))
    # Exclude cvind.mat if present
    data_mats = [f for f in existing_mats if f.name != "cvind.mat"]
    if len(data_mats) >= target_mat_count:
        print(f"[OK] Dataset already present: {len(data_mats)} .mat files found in {RAW_MAT_DIR}")
        return
        
    print(f"[*] Querying Figshare API for dataset files (Article ID: 1512427)...")
    files = get_figshare_file_list()
    print(f"[*] Found {len(files)} files on Figshare.")
    
    for f_info in files:
        name = f_info["name"]
        download_url = f_info["download_url"]
        size_mb = f_info.get("size", 0) / (1024 * 1024)
        
        # We need the zip files and cvind.mat
        dest_path = ZIP_DIR / name if name.endswith(".zip") else RAW_MAT_DIR / name
        
        if dest_path.exists() and dest_path.stat().st_size > 0:
            print(f"[OK] Already downloaded: {name}")
        else:
            print(f"[*] Downloading {name} ({size_mb:.2f} MB)...")
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=name) as t:
                urllib.request.urlretrieve(download_url, filename=str(dest_path), reporthook=t.update_to)
                
        # If it's a zip file, extract .mat files into RAW_MAT_DIR
        if name.endswith(".zip"):
            print(f"[*] Extracting {name} into {RAW_MAT_DIR}...")
            with zipfile.ZipFile(dest_path, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    if member.endswith(".mat"):
                        filename = os.path.basename(member)
                        if filename:
                            source = zip_ref.open(member)
                            target = open(RAW_MAT_DIR / filename, "wb")
                            with source, target:
                                target.write(source.read())
                                
    total_mats = len([f for f in RAW_MAT_DIR.glob("*.mat") if f.name != "cvind.mat"])
    print(f"[OK] Extraction complete! Total data .mat files verified: {total_mats}")


if __name__ == "__main__":
    download_and_extract_all()
