import random
import time

import pandas as pd
import numpy as np


from utils import Load, Preprocess, Log, Graph


filename = 'o_tr2_te2'                          # 'tr': train; 'te': test; '1': small; '2': large
graph_name = 'knn in large trainset & large testset'
                                                # naming graphs
train_sample = r'dataset/train_sample.csv'      # small trainset
train_label  = r'dataset/train_label.csv'       # small trainset
train_sample = r'dataset/train2_sample.csv'     # large trainset
train_label  = r'dataset/train2_label.csv'      # large trainset
test_sample  = r'dataset/test_sample.csv'       # small testset
test_label   = r'dataset/test_label.csv'        # small testset
test_sample  = r'dataset/test2_sample.csv'      # large testset
test_label   = r'dataset/test2_label.csv'       # large testset


def run(train_sample, train_label, test_sample, test_label, k):
    train_sample, train_sample_size = Load.loadSample(train_sample)  
    train_label,  train_label_size  = Load.loadLabel(train_label)
    assert train_sample_size == train_label_size, 'train_sample_size does not match train_label_size'

    test_sample, test_sample_size = Load.loadSample(test_sample)
    test_label,  test_label_size  = Load.loadLabel(test_label)
    assert test_sample_size == test_label_size, 'test_sample_size does not match test_label_size'

    train_sample = Preprocess.normalize(train_sample).values.tolist()   # list
    test_sample  = Preprocess.normalize(test_sample).values.tolist()    # list

    label_to_index = {label: index for index, label in enumerate(set(train_label['x'].tolist()))}
    train_index = Preprocess.labelMap(train_label, label_to_index)      # list
    test_index  = Preprocess.labelMap(test_label,  label_to_index)      # list

    correct_count = 0

    for i, one in enumerate(test_sample):
        euclid_dist = np.linalg.norm(np.array(one) - np.array(train_sample), axis=1)
        nn_idx = euclid_dist.argsort()[:k]

        nn_vote = []
        nn_decision = 0
        for idx in nn_idx:
            nn_vote.append(train_index[idx])     # for there are only 1 or 0
        if sum(nn_vote) > k / 2:
            # print(list(label_to_index.keys())[1])
            nn_decision = 1
        else:
            # print(list(label_to_index.keys())[0])
            nn_decision = 0
        # print(test_label.values.tolist()[i][0])
        if test_label.values.tolist()[i][0] == list(label_to_index.keys())[nn_decision]:
            # right
            correct_count += 1
    test_correct = correct_count / test_sample_size
    Log.log(filename, 'k: {}; correct rate: {}\n'.format(k, test_correct))
    return test_correct


if __name__ == '__main__':
    Log.clearLog(filename)

    rec = []
    for k in range(1, 101):
        rec.append(run(
            train_sample, 
            train_label,
            test_sample,
            test_label,
            k
        ))
        
    maxi = max(rec)
    mini = min(rec)
    avrg = sum(rec) / len(rec)
    stdv = np.std(rec, ddof=1)
    Log.log(filename, 'max: {}; min: {}; average: {}; std: {}'.format(maxi, mini, avrg, stdv))
    Graph.draw(filename, [i for i in range(1, 101)], rec, 100, 1.0, graph_name)