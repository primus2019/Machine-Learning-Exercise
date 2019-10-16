import numpy as np
import pandas as pd


def load_dataset(file_path, test_indices=None):
    '''
        load dataset

        param:
        
        file_path: path
        
        indices: default none, compared with indices of the dataset, assert warning if not identical
    '''
    # train_data = []
    # with open(file_path) as file:
    #     for idx, line in enumerate(file):
    #         train_data.append(line.split(','))
    # train_data = pd.DataFrame(train_data).T
    # print(train_data.head(5))

    train_data = pd.read_csv(file_path, header=0, index_col=0)
    indices = train_data.index.values
    columns = train_data.columns.values
    train_data = train_data.T.to_numpy()
    if test_indices is not None:
        assert test_indices is indices, 'indices of data and target set are not identical'
    return train_data, indices, columns


if __name__ == '__main__':
    file_path = r'data\train_1_target.csv'
    # file_path = r'data\train_1_data.csv'
    data, indices, columns = load_dataset(file_path)
    print('file path:     ' + file_path)
    print('data shape:    ' + (str)(data.shape))
    print('indices shape: ' + (str)(indices.shape))
    print('columns shape: ' + (str)(columns.shape))