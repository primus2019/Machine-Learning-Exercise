import time
import numpy as np
import pandas as pd


from utils import Load, Preprocess, Calc, Log, Graph


title = 'random forest in large trainset & large testset'
                                                # naming graphs
# train_sample = r'dataset/train1_sample.csv'      # small trainset
# train_label  = r'dataset/train1_label.csv'       # small trainset
train_sample = r'dataset/train2_sample.csv'     # large trainset
train_label  = r'dataset/train2_label.csv'      # large trainset
# test_sample  = r'dataset/test1_sample.csv'       # small testset
# test_label   = r'dataset/test1_label.csv'        # small testset
test_sample  = r'dataset/test2_sample.csv'      # large testset
test_label   = r'dataset/test2_label.csv'       # large testset
M = 500                                         # number of weak classifiers
thresh = 0.45                                   # threshold for CART as weak classifier
CART_step = 10                                  # particle size of CART stumping, larger the smaller


filename = 'rf_o_tr{}_te{}_M{}_thr{}_stp{}'.format(train_sample[13], test_sample[12], M, thresh, CART_step)
                                                # 'tr': train; 'te': test; '1': small; '2': large


def run(filename, train_sample, train_label, test_sample, test_label, title, M, thresh):
    train_sample, train_sample_size = Load.loadSample(train_sample)  
    train_label,  train_label_size  = Load.loadLabel(train_label)
    assert train_sample_size == train_label_size, 'train_sample_size does not match train_label_size'

    test_sample, test_sample_size = Load.loadSample(test_sample)
    test_label,  test_label_size  = Load.loadLabel(test_label)
    assert test_sample_size == test_label_size, 'test_sample_size does not match test_label_size'

    train_sample = Preprocess.normalize(train_sample, True).values.tolist()     # list
    test_sample  = Preprocess.normalize(test_sample,  True).values.tolist()     # list

    label_to_index = {label: index for index, label in enumerate(set(train_label['x'].tolist()))}
    train_index = Preprocess.labelMap(train_label, label_to_index)              # list
    test_index  = Preprocess.labelMap(test_label,  label_to_index)              # list

    input_size = len(train_sample[0])
    sample_size = len(train_sample)
    sample_weights = [1 / sample_size for _ in range(sample_size)]
    classifier_weights    = []
    classifier_thresholds = []
    threshold_positions   = []
    test_corrs            = []
    test_times            = [i + 1 for i in range(M)]
    
    for i in range(M):
        threshold, position, errors = Calc.CART(train_sample, train_index, sample_weights, thresh, CART_step)
        total_error = Calc.gentleError(np.array(sample_weights), np.array(errors))
        classifier_weights.append(round(Calc.classifierError(total_error), 3))
        classifier_thresholds.append(threshold)
        threshold_positions.append(position)
        sample_weights = Calc.updateVariableWeights(np.array(sample_weights), total_error, errors)
        # print('errors: {}'.format(errors))
        # print('sample_weights: {}'.format(sample_weights))
        # print('classifier_threshold: {} in {}'.format(threshold, position))
        # print('total_error: {}'.format(total_error))
        print('threshold_positions:   {}'.format(threshold_positions))
        print('classifier_thresholds: {}'.format(classifier_thresholds))
        print('classifier_weights:    {}'.format(classifier_weights))
    
        test_corr = 0
        test_size = len(test_sample)
        for sample, index in zip(test_sample, test_index):
            vote = 0
            for threshold, position, weight in zip(classifier_thresholds, threshold_positions, classifier_weights):
                if  sample[position] >= threshold:
                    vote += weight
                elif sample[position] < threshold:
                    vote -= weight
            if  vote >= 0 and index == 1:
                test_corr += 1
            elif vote < 0 and index == 0:
                test_corr += 1
        test_corrs.append(round(test_corr / test_size, 3))
        Log.log(filename, 'M: {}; correction: {}\n'.format(M, test_corrs[-1]))
        print('-----------------{}-----------------'.format(i + 1))
        time.sleep(1.0)
        

    Graph.draw(filename, test_times, test_corrs, test_times[-1], 1.0, title)
    return test_corrs


if __name__ == '__main__':
    Log.clearDefaultLog()
    Log.clearLog(filename)

    rec = []

    run(
        filename,
        train_sample,
        train_label,
        test_sample,
        test_label,
        title,
        M,
        thresh
    )