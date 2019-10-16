import numpy as np


def info(train_data, train_data_idx, train_data_col, train_target, test):
    if test:
        dataset_name = 'test'
    else:
        dataset_name = 'train'
    print(dataset_name + ' set info: ')
    print(dataset_name + ' set shape:     ' + (str)(train_data.shape))
    print(dataset_name + ' input labels:  ' + (str)(train_data_col.shape[0]))
    print(dataset_name + ' input size:    ' + (str)(train_data_idx.shape[0]))
    print(dataset_name + ' targets:       ' + (str)(np.unique(train_target)))