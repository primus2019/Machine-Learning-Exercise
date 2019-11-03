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
M = 100                                         # number of weak classifiers

thresh = 0.40                                   # threshold for CART as weak classifier
CART_step = 7                                  # number of steps in modifying CART stumping, negative to CART diversity
hypervariant = 'thresh'
hyperrange   = np.arange(0.30, 0.50, 0.01)

filename = 'default'


def run(filename, train_sample, train_label, test_sample, test_label, title, M, thresh, CART_step):
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
        print('total_error: {}'.format(total_error))
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
        print('-----------------thresh: {}; CART_step: {}; iter: {}-----------------'.format(thresh, CART_step, i + 1))
        

    Graph.draw(filename, test_times, test_corrs, test_times[-1], 1.0, title)
    return test_corrs


def setFileName():
    return 'rf-{}/'.format(hypervariant) + 'rf-o-tr{}-te{}-M{}-thr{}-stp{}'.format(train_sample[13], test_sample[12], M, thresh, CART_step)
                                                    # 'tr': train; 'te': test; '1': small; '2': large

if __name__ == '__main__':
    rec    = []
    maxi   = []
    maxi_M = []
    mini   = []
    mini_M = []
    avrg   = []
    stdv   = []
    # modify the variant after 'for'
    for i, thresh in enumerate(hyperrange):
        filename = setFileName()
        Log.clearDefaultLog()
        Log.clearLog(filename)
        
        rec.append(run(
            filename,
            train_sample,
            train_label,
            test_sample,
            test_label,
            title,
            M,
            thresh,
            CART_step
        ))
        maxi.append(np.max(rec[-1]))
        mini.append(np.min(rec[-1]))
        maxi_M.append(np.argmax(rec[-1]))
        mini_M.append(np.argmin(rec[-1]))
        avrg.append(np.average(rec[-1]))
        stdv.append(np.std(rec[-1], ddof=1))
        Log.log(filename, 'max: {}; min: {}; max M: {}; min M: {}; avg: {}; std: {}'.format(maxi[-1], mini[-1], maxi_M[-1], mini_M[-1], avrg[-1], stdv[-1]))

    Graph.drawHyper('hypervariant-maxM-{}-{}-{}'.format(hypervariant, hyperrange[0], hyperrange[-1]), hyperrange, maxi_M, title='max M when varying {}'.format(hypervariant))
    Graph.drawHyper('hypervariant-maxCorr-{}-{}-{}'.format(hypervariant, hyperrange[0], hyperrange[-1]), hyperrange, maxi, title='max correction when varying {}'.format(hypervariant))
