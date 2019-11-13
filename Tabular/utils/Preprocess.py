import pandas as pd
import numpy as np
import random

## todo
def transpose(ds_path, ds_t_path=False, largeset=False, encoding='utf-8', header=0, index_col=0):
    '''
    Params:

    ds_path: str, dataset path

    ds_t_path(default False): boolean, transposed dataset path
    (should be different from ds_path)

    largeset(default False): boolean, whether to apply low-memory method

    encoding(default 'utf-8'): str, encoding of dataset

    header(default 0): int/list of int, works on pandas.read_csv()
    (learn more at: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    index_col(default 0): int/list of int, works on pandas.read_csv()
    (learn more at: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    Return: 

    pd.Dataframe, newly transposed dataset

    '''
    assert ds_path != ds_t_path, 'You replace original dataset with processed one(s); assign processed one(s) to new position(s), or delete old dataset with delete.'
    if not largeset:
        if not ds_t_path:
            return pd.read_csv(ds_path, encoding=encoding, header=header, index_col=index_col).T
        else:
            pd.read_csv(ds_path, encoding=encoding, header=header, index_col=index_col).T.to_csv(ds_t_path, encoding=encoding)
            return pd.read_csv(ds_t_path, encoding=encoding, header=header, index_col=index_col)
    elif largeset:
        pass


def split(ds_path, ds_split_path, chunksize=None, fraction=None, shuffle=True, largeset=False, encoding='utf-8', header=0, index_col=0):
    '''
    Params:

    ds_path: str, dataset path

    ds_split_path: str/list of str, split dataset path(s)
    (should be different from ds_path)

    chunksize(default None): int, batch for split dataset(s)
    (strictly one of chunksize and fraction should be valid)

    fraction(default None): float, proportion for split dataset(s)
    (strictly one of chunksize and fraction should be valid)

    shuffle(default True): boolean, whether to randomly shuffle samples when splitting

    largeset(default False): boolean, whether to apply low-memory method for splitting

    encoding(default 'utf-8'): str, encoding of dataset

    header(default 0): int/list of int, works on pandas.read_csv()
    (learn more at: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    index_col(default 0): int/list of int, works on pandas.read_csv()
    (learn more at: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

    '''
    assert ds_path != ds_split_path, 'You replace original dataset with processed one(s); assign processed one(s) to new position(s), or delete old dataset with delete.'
    assert not (chunksize is not None and fraction is not None), 'One and only one of chunksize and fraction should be valid; both valid now.'
    assert chunksize or fraction, 'Only and only one of chunksize and fraction should be valid; both invalid now.'
    assert fraction < 1, 'Fraction is over 1; you may mistake it for chunksize, or else you can change it to 1.'
    assert chunksize > 1, 'Chunksize is below 1; you may mistake it for faction.'


    if not largeset:
        ds_raw = pd.read_csv(ds_path, encoding=encoding, header=header, index_col=index_col)
        if type(ds_split_path) is not list:
            if chunksize and shuffle:
                ds_raw.sample(n=chunksize).to_csv(ds_split_path, encoding=encoding)
            elif fraction and shuffle:
                ds_raw.sample(frac=fraction).to_csv(ds_split_path, encoding=encoding)
            elif chunksize and not shuffle:
                ds_raw.iloc[:chunksize, :].to_csv(ds_split_path, encoding=encoding)
            elif fraction and not shuffle:
                ds_raw.iloc[:round(ds_raw.shape[0] * fraction), :].to_csv(ds_split_path, encoding=encoding)
            else:
                raise 'Split function meets unworking input.'
        elif type(ds_split_path) is list:
            if chunksize:
                if shuffle:
                    ds_raw = ds_raw.sample(frac=1)
                total_size, remain_size = ds_raw.shape[0], ds_raw.shape[0]
                chunk_cnt = 0
                while remain_size > chunksize:
                    assert chunk_cnt <= len(ds_split_path), 'There is not enough name for split datasets.'
                    ds_raw.iloc[(total_size - remain_size):(total_size - remain_size + chunksize), :].to_csv(ds_split_path[chunk_cnt], encoding=encoding)
                    chunk_cnt += 1
                    remain_size -= chunksize
                assert chunk_cnt <= len(ds_split_path), 'There is not enough name for split datasets.'
                ds_raw.iloc[(total_size - remain_size):total_size, :].to_csv(ds_split_path[chunk_cnt], encoding=encoding)
            elif fraction:
                if shuffle:
                    ds_raw = ds_raw.sample(frac=1)
                total_size = ds_raw.shape[0]
                total_fraction, remain_fraction = 1.0, 1.0
                chunk_cnt = 0
                while remain_fraction > fraction:
                    assert chunk_cnt < len(ds_split_path), 'There is not enough name for split datasets.'
                    ds_raw.iloc[(total_size - round(remain_fraction * total_size)):(total_size - round((remain_fraction - fraction) * total_size)), :].to_csv(ds_split_path[chunk_cnt], encoding=encoding)
                    chunk_cnt += 1
                    remain_fraction -= fraction
                assert chunk_cnt < len(ds_split_path), 'There is not enough name for split datasets.'
                ds_raw.iloc[(total_size - round(remain_fraction * total_size)):total_size, :].to_csv(ds_split_path[chunk_cnt], encoding=encoding)
            else:
                raise 'Split function meets unworking input.'
        else:
            raise 'Split function meets unworking input.'
    elif largeset:
        if type(ds_split_path) is not list:
            if chunksize:
                ds_raw = pd.read_csv(ds_path, nrows=chunksize, encoding=encoding, header=header, index_col=index_col)
                if shuffle:
                    ds_raw = ds_raw.sample(frac=1)
                ds_raw.to_csv(ds_split_path, encoding=encoding)
            elif fraction:
                total_size = 0
                with open(ds_path, encoding=encoding) as file:
                    for line in file:
                        total_size += 1
                total_size -= 1
                ds_raw = pd.read_csv(ds_path, nrows=round(total_size * fraction), encoding=encoding, header=header, index_col=index_col)
                if shuffle:
                    ds_raw = ds_raw.sample(frac=1)
                ds_raw.to_csv(ds_split_path, encoding=encoding)
            else:
                raise 'Split function meets unworking input.'
        elif type(ds_split_path) is list:
            if chunksize:
                ds_itr = pd.read_csv(ds_path, chunksize=chunksize, encoding=encoding, header=header, index_col=index_col)
                for i, ds_item in enumerate(ds_itr):
                    assert i < len(ds_split_path), 'There is not enough name for split datasets.'
                    if shuffle:
                        ds_item = ds_item.sample(frac=1)
                    ds_item.to_csv(ds_split_path[i], encoding=encoding)
            elif fraction:
                total_size = 0
                with open(ds_path, encoding=encoding) as file:
                    for line in file:
                        total_size += 1
                total_size -= 1
                ds_itr = pd.read_csv(ds_path, chunksize=round(total_size * fraction), encoding=encoding, header=header, index_col=index_col)
                for i, ds_item in enumerate(ds_itr):
                    assert i < len(ds_split_path), 'There is not enough name for split datasets.'
                    if shuffle:
                        ds_item = ds_item.sample(frac=1)
                    ds_item.to_csv(ds_split_path[i], encoding=encoding)
            else:
                raise 'Split function meets unworking input.'
        else:
            raise 'Split function meets unworking input.'
    else:
        raise 'Split function meets unworking input.'
        
                
                        




