import random
import time
import uuid

import numpy as np


from utils import Log


def euclideanDisatance(node_1, node_2):
    return np.linalg.norm(node_1 - node_2)


def CART(train_sample, train_index, variable_error_weights, threshold, step=40):
    label_size = len(train_sample[0])
    sample_size = len(train_sample)
    Log.defaultLog('train input size: {}'.format(label_size))

    random.seed(uuid.uuid4().hex)
    position = random.randint(0, label_size - 1)
    weights = random.random() * 2 - 1

    Log.defaultLog('try the weight on {}'.format(position))
    step = 40
    gain_ginis = []

    # for i in range(step):
    #     larger_samples  = []
    #     larger_0 = []
    #     larger_1 = []
        
    #     smaller_samples = []
    #     smaller_0 = []
    #     smaller_1 = []

    #     for y, (sample, index) in enumerate(zip(train_sample, train_index)):
    #         if sample[position] >= weights[position]:
    #             if index == 0:
    #                 larger_0.append(y)
    #             else:
    #                 larger_1.append(y)
    #         else:
    #             if index == 0:
    #                 smaller_0.append(y)
    #             else:
    #                 smaller_1.append(y)
        
    #     gain_ginis.append(gini(len(larger_0), len(larger_1), sample_size))
    #     weights[position] += 2.0 / step
    
    # smallest_gain_gini_idx = np.argmin(gain_ginis)
    while(True):
        for _ in range(label_size):
            for i in range(step):
                error_rate = 0.0
                error_positions = [0 for _ in range(sample_size)]
                Log.defaultLog('try threshold {}'.format(weights))
                for y, (sample, label) in enumerate(zip(train_sample, train_index)):
                    # Log.defaultLog('sample: {}'.format(sample))
                    # Log.defaultLog('label:  {}'.format(label))
                    if sample[position] > weights and label == 0:
                        # error_rate += variable_error_weights[position]
                        error_rate += 1
                        # Log.defaultLog('{}   get one error: type 1; error_rate: {}'.format(y, error_rate))
                        error_positions[y] = 1
                    elif sample[position] < weights and label == 1:
                        # error_rate += variable_error_weights[position]
                        error_rate += 1
                        # Log.defaultLog('{}   get one error: type 2; error_rate: {}'.format(y, error_rate))
                        error_positions[y] = 1
                # error_rate /= np.sum(variable_error_weights)
                error_rate /= sample_size
                Log.defaultLog('error_rate: {}'.format(error_rate))
                if error_rate < threshold:
                    Log.defaultLog('get right CART: {} in {}'.format(weights, position))
                    return round(weights, 3), position, error_positions
                else:
                    weights += 2.0 / step
            Log.defaultLog('--------------------------------')
            Log.defaultLog('false weight on {}, loop again'.format(position))
            position = (position + 1) % label_size
            weights = -1
        print('seemingly no feasible CART for this threshold, increase threshold by 0.02 and loop again')
        threshold += 0.02
    assert False, 'no feasible CART for this threshold!'
        

def gini(class_1, class_2, total_number):
    class_number = class_1 + class_2
    return (class_number / total_number) * (1 - (class_1 / class_number) ** 2 - (class_2 / class_number) ** 2)

# epsilon
def gentleError(variable_error_weights, errors):
    return np.dot(variable_error_weights, errors.T) / np.sum(variable_error_weights)


def classifierError(total_error):
    return np.log((1 - total_error) / total_error)


def updateVariableWeights(sample_weights, total_error, errors):
    new_sample_weights = []
    for weight, error in zip(sample_weights, errors):
        if error == 1:
            new_sample_weights.append(weight * (1 - total_error) / total_error)
        else:
            new_sample_weights.append(weight)
    # return variable_error_weights * (1 - total_error) / total_error
    return np.array(new_sample_weights)