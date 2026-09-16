"""
Downloads and unzips the MovieLens Small dataset into data/ml-latest-small/.
We don't commit the raw CSVs to git — run this script instead after cloning.

Usage: python data/download.py
"""
import os
import ssl
import zipfile
import urllib.request
import certifi

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DEST_DIR = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(DEST_DIR, "ml-latest-small.zip")


def main():
    if os.path.exists(os.path.join(DEST_DIR, "ml-latest-small", "ratings.csv")):
        print("Dataset already present, skipping download.")
        return

    print(f"Downloading {URL} ...")
    # Use certifi's certificate bundle explicitly — fixes
    # "SSL: CERTIFICATE_VERIFY_FAILED" on some Windows Python installs.
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler(context=ssl_context)
    )
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(URL, ZIP_PATH)

    print("Unzipping...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DEST_DIR)

    os.remove(ZIP_PATH)
    print(f"Done. Data is in {os.path.join(DEST_DIR, 'ml-latest-small')}")


if __name__ == "__main__":
    main()