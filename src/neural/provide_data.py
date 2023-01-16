from collections import defaultdict
import os
import requests
import pandas as pd
from numpy import uint8

TEST_CSV = 'fashion-mnist_test.csv'
TRAIN_CSV = 'fashion-mnist_train.csv'
PATH_DATA = './dataSources'


def load_train_and_test_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(PATH_DATA):
        print('Create Data Directory ...')
        os.mkdir(PATH_DATA)
    # Download data if it is unavailable.
    if not all(csv in os.listdir(PATH_DATA) for csv in (TEST_CSV, TRAIN_CSV)):
        download_data()
    return read_csv_as_bytes(TRAIN_CSV), read_csv_as_bytes(TEST_CSV)


def read_csv_as_bytes(filename: str) -> pd.DataFrame:
    # as data describe pixel intensities in the range of 0..255, it is efficient to load them as unsigned bytes
    return pd.read_csv(PATH_DATA + '/' + filename, dtype=defaultdict(lambda: uint8))


def download_data():
    download_csv("https://www.dropbox.com/s/5vg67ndkth17mvc/", TRAIN_CSV)
    download_csv("https://www.dropbox.com/s/9bj5a14unl5os6a/", TEST_CSV)


def download_csv(dropbox_folder: str, filename: str):
    print(f'Dataset {filename} loading ... Please wait...')
    r = requests.get(dropbox_folder + filename + "?dl=1", allow_redirects=True)
    open(PATH_DATA + '/' + filename, 'wb').write(r.content)
    print('Loaded.')
