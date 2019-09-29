# perceptron with fixed increment rule
import pandas as pd
from pandas import Series, DataFrame
import numpy as np

def train():
    labels   = pd.read_csv(r'C:\Primus\Codes\Python\Machine-Learning-Exercise\exercise_1\data1\train_label.csv')
    features = pd.read_csv(r'C:\Primus\Codes\Python\Machine-Learning-Exercise\exercise_1\data1\train_10gene.csv')

    samples     = features.iloc[:,1:]
    max_sample_num = len(features.columns) - 1
    activations = Series(np.zeros(shape=(max_sample_num)))

    # for index, row in labels.iterrows():
    #     if row[1] == 'fibroblast':
    #         if row[0] in samples.columns:
    #             samples[row[0]] = samples[row[0]]
    #             print(row[0] + ' changed')

    for index, col in enumerate(features.columns):
        for sub_index, row in labels.iterrows():
            if row[0] == col and row[1] == 'endothelial cell':
                activations.iloc[index - 1] = 1
                break

    weight = Series(np.ones(shape=(11)))
    memo = [0 for _ in range(40)]
    learning_rate = 0.005
    sample_num  = 0
    record = [0 for _ in range(max_sample_num)]
    while sample_num < 10000:
        temp = weight
        discriminant = np.dot(weight[1:], samples.iloc[:, sample_num % max_sample_num]) + weight[0]
        if sample_num % 2000 == 0:
            print('----------------------------')
            print('iteration '      + (str)(sample_num))
            print('discriminant: '  + (str)(discriminant))
            print('weight: '        + (str)(weight))
        # check if valid
        if np.abs(discriminant) < 0.01:
            record[sample_num % max_sample_num] = 1
            valid = True
            for i in record:
                if i == 0:
                    valid = False
            if valid:
                print('solution vector: \n' + (str)(weight))
                return weight
        else:
            record = [0 for _ in range(max_sample_num)]
        if discriminant > np.zeros(shape=(1,1)):
            prediction = 1
        else:
            prediction = 0
        weight[1:] += learning_rate*np.dot((activations[sample_num % max_sample_num] - prediction), samples.iloc[:, sample_num % max_sample_num])
        weight[0]  += learning_rate*(activations[sample_num % max_sample_num] - prediction)

        # if np.dot(weight, samples.iloc[:,sample_num % 40]) > np.zeros(shape=(1,1)):
        #     memo[sample_num % 40] = 1
        #     test = True
        #     for sample in memo:
        #         if sample == 0:
        #             test = False
        #     if test:
        #         print('solution vector: \n' + (str)(weight))
        #         break
        # else:
        #     memo = [0 for _ in range(40)]
        #     weight = weight+Series(samples.iloc[:,sample_num % 40])
        # print(weight)
        # print('sample_num: ' + (str)(sample_num % 40))
        sample_num += 1
    
    return weight


def test(weight, features_path, labels_path):
    features = pd.read_csv(features_path)
    labels   = pd.read_csv(labels_path)
    max_sample_num = len(features.columns) - 1

    samples     = features.iloc[:,1:]
    
    for index, row in labels.iterrows():
        if row[1] == 'endothelial cell':
            if row[0] in samples.columns:
                samples[row[0]] = samples[row[0]]
                # print(row[0] + ' changed')
    
    sample_num = 0
    error, total = 0, 0
    while True:
        total += 1
        # print('----------------------------')
        # print('iteration ' + (str)(sample_num))


        if np.dot(weight[1:], samples.iloc[:,sample_num]) + weight[0] > np.zeros(shape=(1,1)):
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
    