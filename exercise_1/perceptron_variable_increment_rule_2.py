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
    with open('variable_increment_train_2.log', 'a+') as file:
        while sample_num < 10001:
            temp = weight
            discriminant = np.dot(weight[1:], samples.iloc[:, sample_num % max_sample_num]) + weight[0]
            if sample_num % 2000 == 0:
                file.write('----------------------------\n')
                file.write('iteration '      + (str)(sample_num) + '\n')
                file.write('discriminant: '  + (str)(discriminant) + '\n')
                file.write('weight: '        + (str)(weight) + '\n')
            # check if valid
            if np.abs(discriminant) < 0.01:
                record[sample_num % max_sample_num] = 1
                valid = True
                for i in record:
                    if i == 0:
                        valid = False
                if valid:
                    file.write('final iteration: ' + (str)(sample_num) + '\n')
                    file.write('solution vector: \n' + (str)(weight) + '\n')
                    return weight
            else:
                record = [0 for _ in range(max_sample_num)]
            if discriminant > np.zeros(shape=(1,1)):
                prediction = 1
            else:
                prediction = 0
            weight[1:] += learning_rate*np.dot((activations[sample_num % max_sample_num] - prediction), samples.iloc[:, sample_num % max_sample_num])
            weight[0]  += learning_rate*(activations[sample_num % max_sample_num] - prediction)

            sample_num += 1
    
    return weight


def test(weight, features_path, labels_path):
    features = pd.read_csv(features_path)
    labels   = pd.read_csv(labels_path)
    max_sample_num = len(features.columns) - 1

    samples  = features.iloc[:,1:]
    
    for index, row in labels.iterrows():
        if row[1] == 'endothelial cell':
            if row[0] in samples.columns:
                samples[row[0]] = samples[row[0]]
                # print(row[0] + ' changed')
    
    sample_num = 0
    error, total = 0, 0
    with open('variable_increment_test_2.log', 'a+') as file:
        file.write('features_path: ' + (str)(features_path) + '\n')
        file.write('labels_path: ' + (str)(labels_path) + '\n')
        while True:
            total += 1

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
                file.write('error rate: ' + (str)(error / total) + '\n')
                break
    



if __name__ == '__main__':
    open('variable_increment_train_2.log', 'w').close()
    open('variable_increment_test_2.log', 'w').close()
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
    