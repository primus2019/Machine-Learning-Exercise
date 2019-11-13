import pandas as pd
import numpy as np
import random


def transpose(ds_path, ds_t_path=False, largeset=False, encoding='utf-8', header=0, index_col=0):
    assert ds_path != ds_t_path, 'You replace original dataset with processed one(s); assign processed one(s) to new position(s), or delete old dataset with delete.'
    if not largeset:
        if not ds_t_path:
            return pd.read_csv(ds_path, encoding=encoding, header=header, index_col=index_col).T
        else:
            pd.read_csv(ds_path, encoding=encoding, header=header, index_col=index_col).T.to_csv(ds_t_path, encoding=encoding)
            return pd.read_csv(ds_t_path, encoding=encoding, header=header, index_col=index_col)
    elif largeset:
        pass


def split(ds_path, ds_split_path, chunksize=None, fraction=None, shuffle=True, largeset=True, encoding='utf-8', header=0, index_col=0):
    assert ds_path != ds_split_path, 'You replace original dataset with processed one(s); assign processed one(s) to new position(s), or delete old dataset with delete.'
    assert not (chunksize is not None and fraction is not None), 'One and only one of chunksize and fraction should be valid; both valid now.'
    assert chunksize or fraction, 'Only and only one of chunksize and fraction should be valid; both invalid now.'

    ds_raw = pd.read_csv(ds_path, encoding=encoding, header=header, index_col=index_col)

    if type(ds_split_path) is not list:
        if chunksize and shuffle:
            ds_raw.sample(n=chunksize).to_csv(ds_split_path, encoding=encoding)
            return pd.read_csv(ds_split_path)
        elif fraction and shuffle:
            ds_raw.sample(frac=fraction).to_csv(ds_split_path, encoding=encoding)
            return pd.read_csv(ds_split_path)
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


