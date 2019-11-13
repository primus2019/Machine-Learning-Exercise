import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def EDA(ds_path, data_type, encoding='utf-8', header=0, index_col=0, largeset=False, chunksize=1000):
    types = ['feature', 'label']
    assert data_type in types, 'Types are not valid'
    print('---------------------EDA of {}---------------------'.format(os.path.basename(ds_path)))
    if not largeset:
        ds_raw = pd.read_csv(ds_path, encoding='utf-8', index_col=index_col, header=header)
        if   data_type == 'feature':
            print('sample(row):     {}'.format(ds_raw.shape[0]))
            print('feature(line):   {}'.format(ds_raw.shape[1]))
            print('head of feature: \n{}'.format(ds_raw.head()))
        elif data_type == 'label':
            print('sample(row):     {}'.format(ds_raw.shape[0]))
            print('label(line):     {}'.format(ds_raw.shape[1]))
            print('label:           {}'.format(list(set(np.ravel(ds_raw.values)))))
            print('head of label:   \n{}'.format(ds_raw.head()))
        plt.title(os.path.basename(ds_path))
        #########################################
        plt.scatter()
    elif largeset:
        ds_raw = pd.read_csv(ds_path, chunksize=chunksize)
        pass


def header(ds, head=5, largeset=False, encoding='utf-8', header=0, index_col=0):
    if type(ds) == str:
        ds = pd.read_csv(ds_path, encoding=encoding, header=header, index_col=index_col)
    print(ds.head())


def shape(ds, largeset=False, encoding='utf-8', header=0, index_col=0):
    '''
    Params:

    ds: pd.Dataframe or str of dataset path
    '''
    if type(ds) == str:
        ds = pd.read_csv(ds, encoding=encoding, header=0, index_col=0)
    print('sample(row):           {}'.format(ds.shape[0]))
    print('feature/label(column): {}'.format(ds.shape[1]))    


def sparse_feature(ds, measure='std', threshold=0.01, largeset=False, encoding='utf-8', header=0, index_col=0):
    '''
    Params: 

    ds: pandas.Dataframe, numpy.ndarray or str of dataset path shaped [n of samples, n of features]

    measure(default 'std'): either 'mean' or 'std', deciding the calculation of feature performance
    (mean threshold are compared with features' absolute means)

    Return:
    
    pandas.Series of [n of features] of boolean values. 
    
    True represent the feature is not sparse; False represent sparse
    '''
    if type(ds) == str:
        ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col)
    if measure == 'mean':
        return (ds != 0).abs().mean() > threshold
    elif measure == 'std':
        return (ds != 0).std() > threshold

