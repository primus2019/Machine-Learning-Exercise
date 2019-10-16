import numpy as np
import pandas as pd


def load_dataset(file_path, test_indices=None):
    '''
        load dataset

        param:
        
        file_path: path
        
        indices: default none, compared with indices of the dataset, assert warning if not identical
    '''

    train_data = pd.read_csv(file_path, header=0, index_col=0)
    if train_data.shape[0] < train_data.shape[1]:                   # if the dataset has input labels as index(row), then transpose
        train_data = train_data.T
    indices = train_data.index.values
    columns = train_data.columns.values
    train_data = train_data.to_numpy()
    if test_indices is not None:
        assert (test_indices == indices).all(), 'indices of data ' + (str)(test_indices.shape) + ' and target set ' + (str)(indices.shape) + ' are not identical'
    return train_data, indices, columns


def set_targets(dataset_target):
    dataset_target_unique = np.squeeze(np.unique(dataset_target))
    print('mapping table: ')
    for idx, target in enumerate(dataset_target_unique):
        print('[' + (str)(idx * 2 - 1) + ',' + (str)(target) + ']')
    return np.stack([(2 * np.where(dataset_target_unique==target[0])[0][0] - 1) for target in dataset_target])


def __test(file_path):
    data, indices, columns = load_dataset(file_path)
    print('file path:     ' + file_path)
    print('data shape:    ' + (str)(data.shape))
    print('indices shape: ' + (str)(indices.shape))
    print('columns shape: ' + (str)(columns.shape))



if __name__ == '__main__':
    file_path_1 = r'data\train_1_target.csv'
    file_path_2 = r'data\train_1_data.csv'
    file_path_3 = r'data\test_1_target.csv'
    file_path_4 = r'data\test_1_data.csv'
    __test(file_path_1)
    __test(file_path_2)
    __test(file_path_3)
    __test(file_path_4)