from utils import Load, Preprocess, EDA


import pandas as pd
import numpy as np

def run():
    feature_path = 'tmp/test1_feature_t.csv'
    # Load.EDA(feature_path, 'feature')
    # label_path = 'dataset/test1_label.csv'
    # Load.EDA(label_path, 'label')
    print(EDA.sparse_feature(feature_path, threshold=0.5))
    ds = pd.read_csv(feature_path, encoding='utf-8', header=0, index_col=0)


if __name__ == '__main__':
    run()