import pandas as pd
import numpy as np


def normalize(sample_data, sign=False):
    max_value = sample_data.max().max()
    min_value = sample_data.min().min()
    assert max_value != min_value, 'max_value in sample data equals min_value'

    if sign:
        return ((sample_data - min_value) / (max_value - min_value)) * 2 - 1
    else:
        return (sample_data - min_value) / (max_value - min_value)


def labelMap(sample_label, label_to_index):
    return [label_to_index[label] for label in sample_label.transpose().values.tolist()[0]]