import os
import requests
import zipfile


__data_root = "data/"
__zip_filename = __data_root + "UCI%20HAR%20Dataset.zip"
__data_directory = __data_root + "UCI HAR Dataset"


def download():
    directory = os.path.abspath(__zip_filename)
    if not os.path.exists(directory):
        print("Downloading data")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
        r = requests.get(url, allow_redirects=True)
        open(__zip_filename, "wb").write(r.content)


def unzip():
    directory = os.path.abspath(__data_directory)
    if not os.path.exists(directory):
        print("Unzipping data")
        with zipfile.ZipFile(__zip_filename, "r") as zip_ref:
            zip_ref.extractall(__data_root)
        os.remove(__zip_filename)


def get_data():
    directory = os.path.abspath(__data_directory)
    if not os.path.exists(directory):
        download()
        unzip()
    else:
        print("Data already prepared")


if __name__ == "__main__":
    get_data()
