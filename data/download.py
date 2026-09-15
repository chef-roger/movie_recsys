"""
Downloads and unzips the MovieLens Small dataset into data/ml-latest-small/.
We don't commit the raw CSVs to git — run this script instead after cloning.

Usage: python data/download.py
"""
import os
import zipfile
import urllib.request

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DEST_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(DEST_DIR, "ml-latest-small.zip")


def main():
    if os.path.exists(os.path.join(DEST_DIR, "ml-latest-small", "ratings.csv")):
        print("Dataset already present, skipping download.")
        return

    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, ZIP_PATH)

    print("Unzipping...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DEST_DIR)

    os.remove(ZIP_PATH)
    print(f"Done. Data is in {os.path.join(DEST_DIR, 'ml-latest-small')}")


if __name__ == "__main__":
    main()
