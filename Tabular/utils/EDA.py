import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


def EDA(ds, data_type, encoding='utf-8', header=0, index_col=0, largeset=False, nrows=1000):
    '''
    Params:

    ds_path: str/pd.Dataframe , dataset path or dataset

    data_type: str, either 'feature' or 'label'; decides EDA mode

    encoding(default 'utf-8'): str, encoding of dataset

    header(default 0): int/list of int, works on pandas.read_csv()
    (learn more at: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    index_col(default 0): int/list of int, works on pandas.read_csv()
    (learn more at: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    largeset(default False): boolean, whether to apply low-memory method for EDA

    nrows(default 1000): int, works on pandas.read_csv(), work only when largeset is True
    (learn more at: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    '''
    types = ['feature', 'label']
    assert data_type in types, 'Types are not valid; should be \'feature\' or \'label\''

    if type(ds) is str:
        print('---------------------EDA of {}---------------------'.format(os.path.basename(ds)))
    if not largeset:
        if type(ds) is str:
            ds_raw = pd.read_csv(ds, encoding='utf-8', index_col=index_col, header=header)
            print('[{} sample(row) * {} feature(column)]'.format(ds_raw.shape[0], ds_raw.shape[1]))
        else:
            ds_raw = ds
        if data_type == 'feature':
            print('head of feature: \n{}'.format(ds_raw.head()))
        elif data_type == 'label':
            print('label:           {}'.format(list(set(np.ravel(ds_raw.values)))))
            print('head of label:   \n{}'.format(ds_raw.head()))
        # print(ds_raw.describe(include='all'))
        # ds_raw.isnull().sum()
        # sns.pairplot(ds_raw)
        # plt.show()
        check_null(ds_raw, 'tmp/check_null.csv')
    elif largeset:
        if type(ds) is str:
            rows, columns = 0, 0
            with open(ds, encoding=encoding) as file:
                for line in file:
                    rows += 1
                    columns = len(line.split(',')) - 1
            rows -= 1
            print('[{} sample(row) * {} feature(column)]'.format(rows, columns))
            ds_raw = pd.read_csv(ds, nrows=nrows, encoding=encoding, header=header, index_col=index_col)
        else: 
            ds_raw = ds
        EDA(ds_raw, data_type, encoding=encoding, header=header, index_col=index_col)



def header(ds, head=5, largeset=False, encoding='utf-8', header=0, index_col=0):
    if type(ds) == str:
        ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col)
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

    measure(default 'std'): str, either 'mean' or 'std', deciding the calculation of feature performance
    (mean threshold are compared with features' absolute means)

    threshold(default 0.01): float, threshold for deciding whether a feature is sparse

    largeset(default Faslse): boolean, whether to apply low-memory method for sparse detection

    encoding(default 'utf-8'): str, encoding of dataset

    header(default 0): int, works on pandas.read_csv()
    (learn more at: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    index_col(default 0 ): int, works on pandas.read_csv()
    (learn more at: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    Return:
    
    pandas.Series, boolean values of shape [n of features]
    (true represent the feature is not sparse; False represent sparse)

    '''
    if type(ds) == str:
        ds = pd.read_csv(ds, encoding=encoding, header=header, index_col=index_col)
    if measure == 'mean':
        return (ds != 0).abs().mean() > threshold
    elif measure == 'std':
        return (ds != 0).std() > threshold


def check_null(ds, file_path=None, encoding='utf-8'):
    series_null = ds.isnull().sum()
    print('{} out of {} features contains NA data, totally {}'.format((series_null > 0).sum(), series_null.size, series_null.sum()))
    if file_path:
        series_null.to_csv(file_path, encoding='utf-8', header=True)
