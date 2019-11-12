import pandas as pd
import numpy as np


def inspect(csv_path):
    ds = pd.read_csv(csv_path, index_col=0, header=0)
    print('columns: \t{}'.format(ds.shape[1]))
    print('rows: \t\t{}'.format(ds.shape[0]))
    print(ds.head())


def load(csv_path):
    return pd.read_csv(csv_path, index_col=0, header=0).values.T