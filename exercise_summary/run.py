from utils.Load import *
from utils.Info import *
from utils.Preprocess import *

def run(train_data_path, train_target_path, test_data_path, test_target_path):
    train_data, train_data_idx, train_data_col       = load_dataset(train_data_path)
    train_target, train_target_idx, train_target_col = load_dataset(train_target_path, train_data_idx)
    info(train_data, train_data_idx, train_data_col, train_target, False)
    train_target = set_targets(train_target)
    train_data = scale(train_data, -1, 1)
    w, b = initialize(train_data_col.shape[0])
    



if __name__ == '__main__':
    train_data_path   = r'data\train_1_data.csv'
    train_target_path = r'data\train_1_target.csv'
    test_data_path    = r'data\test_1_data.csv'
    test_target_path  = r'data\test_1_target.csv'
    run(
        train_data_path=train_data_path,
        train_target_path=train_target_path,
        test_data_path=test_data_path,
        test_target_path=test_target_path
    )