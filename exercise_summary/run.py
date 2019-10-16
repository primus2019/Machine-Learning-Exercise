from utils.Load import *


def run(train_data_path, train_target_path):
    train_data, train_data_idx, train_data_col       = load_dataset(train_data_path)
    train_target, train_target_idx, train_target_col = load_dataset(train_target_path, train_data_col)


if __name__ == '__main__':
    train_data_path   = r'data\train_1_data.csv'
    train_target_path = r'data\train_1_target.csv'
    run(
        train_data_path=train_data_path,
        train_target_path=train_target_path
    )