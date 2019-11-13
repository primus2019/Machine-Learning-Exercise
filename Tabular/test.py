from utils import Load, Preprocess, EDA


import pandas as pd
import numpy as np

def run():
    feature_path = 'tmp/test2_feature_t.csv'
    EDA.EDA(feature_path, 'feature', largeset=True, nrows=10)
    # label_path = 'dataset/test1_label.csv'
    # Load.EDA(label_path, 'label')
    # data_split_path = ['split/test2_feature_s_{}.csv'.format(i) for i in range(4)]
    # Preprocess.split(feature_path, data_split_path, fraction=80, largeset=True)
    # for data_split_item in data_split_path:
    #     EDA.EDA(data_split_item, 'feature')
    

if __name__ == '__main__':
    run()