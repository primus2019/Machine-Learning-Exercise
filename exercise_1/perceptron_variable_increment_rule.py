# perceptron with fixed increment rule
import pandas as pd
from pandas import Series, DataFrame
import numpy as np

def train():
    labels   = pd.read_csv(r'C:\Primus\Codes\Python\Machine-Learning-Exercise\exercise_1\data1\train_10gene_label_sub.csv')
    features = pd.read_csv(r'C:\Primus\Codes\Python\Machine-Learning-Exercise\exercise_1\data1\train_10gene_sub.csv')

    samples     = features.iloc[:,1:]
    max_sample_num = len(features.columns) - 1

    for index, row in labels.iterrows():
        if row[1] == 'fibroblast':
            if row[0] in samples.columns:
                samples[row[0]] = -samples[row[0]]
                print(row[0] + ' changed')

    weight = Series(np.random.rand((10)))
    memo = [0 for _ in range(max_sample_num)]
    sample_num  = 0
    threshold = 0
    while True:
        print('----------------------------')
        print('iteration ' + (str)(sample_num))
        # print('max_sample_sum: ' + (str)(max_sample_num))
        # print('aT`y = ' + (str)(np.dot(weight, samples.iloc[:,sample_num % max_sample_num])))
        temp = weight
        if np.abs(np.dot(weight, samples.iloc[:,sample_num % max_sample_num])) > np.zeros(shape=(1,1)):
            memo[sample_num % max_sample_num] = 1
            test = True
            for sample in memo:
                if sample == 0:
                    test = False
            if test:
                print('solution vector: \n' + (str)(weight))
                break
        else:
            # print('samples.iloc[:,sample_num % max_sample_num] = \n' + (str)(samples.iloc[:,sample_num % max_sample_num]))
            memo = [0 for _ in range(max_sample_num)]
            weight += samples.iloc[:,sample_num % max_sample_num]*np.abs(np.dot(weight, samples.iloc[:,sample_num % max_sample_num]))/Euclid(samples.iloc[:,sample_num % max_sample_num])**2
            weight[0]  += samples.iloc[:]
        # print(weight)
        # print('sample_num: ' + (str)(sample_num % max_sample_num))
        sample_num += 1
    
    return weight


def Euclid(vector):
    print(np.sqrt(np.sum(np.square(Series(vector)))))
    return(np.sqrt(np.sum(np.square(Series(vector)))))



def test(weight, features_path, labels_path):
    features = pd.read_csv(features_path)
    labels   = pd.read_csv(labels_path)
    max_sample_num = len(features.columns) - 1

    samples     = features.iloc[:,1:]
    
    for index, row in labels.iterrows():
        if row[1] == 'endothelial cell':
            if row[0] in samples.columns:
                samples[row[0]] = -samples[row[0]]
                # print(row[0] + ' changed')
    
    sample_num = 0
    error, total = 0, 0
    while True:
        total += 1
        # print('----------------------------')
        # print('iteration ' + (str)(sample_num))


        if np.dot(weight, samples.iloc[:,sample_num]) > np.zeros(shape=(1,1)):
            for index, row in labels.iterrows():
                if row[0] == samples.columns[sample_num]:
                    if row[1] == 'endothelial cell':
                        break
                    else:
                        error += 1
                        break
        else:
            for index, row in labels.iterrows():
                if row[0] == samples.columns[sample_num]:
                    if row[1] == 'endothelial cell':
                        error += 1
                        break
                    else:
                        break

        sample_num += 1
        if sample_num == max_sample_num:
            print('error rate: ' + (str)(error / total))
            break
    



if __name__ == '__main__':
    weight = train()
    test(
        weight, 
        r'C:\Primus\Codes\Python\Machine-Learning-Exercise\exercise_1\data1\train_10gene.csv',
        r'C:\Primus\Codes\Python\Machine-Learning-Exercise\exercise_1\data1\train_label.csv'
        )
    test(
        weight,
        r'C:\Primus\Codes\Python\Machine-Learning-Exercise\exercise_1\data1\test_10gene.csv',
        r'C:\Primus\Codes\Python\Machine-Learning-Exercise\exercise_1\data1\test_label.csv'
        )
    