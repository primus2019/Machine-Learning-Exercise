import pandas as pd
import numpy as np


def normalize(sample_data):
    max_value = sample_data.max().max()
    min_value = sample_data.min().min()
    # print('max_value: {}'.format(max_value))
    # print('min_value: {}'.format(min_value))
    assert max_value != min_value, 'max_value in sample data equals min_value'

    return (sample_data - min_value) / (max_value - min_value)


def labelMap(sample_label, label_to_index):
    return [label_to_index[label] for label in sample_label.transpose().values.tolist()[0]]