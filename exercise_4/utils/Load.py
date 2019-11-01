import pandas as pd


def loadSample(file_name):
    file = pd.read_csv(file_name, header=0, index_col=0)
    return file.T, file.columns.shape[0]


def loadLabel(file_name):
    file = pd.read_csv(file_name, header=0, index_col=0)
    return file, file.index.shape[0]


def test():
    test_sample  = r'dataset/test_sample.csv'
    test_samples, sample_size = loadSample(test_sample)
    test_label   = r'dataset/test_label.csv'
    test_labels, label_size = loadLabel(test_label)


if __name__ == '__main__':
    test()