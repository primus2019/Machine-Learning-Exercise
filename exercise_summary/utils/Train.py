import numpy as np
import math


def train(input, target, w, b, epochs, learning_rate=0.01, fixed_increment=True):
    error = False
    if not fixed_increment:
        lr = learning_rate / math.log10(epochs)
    else:
        lr=learning_rate
    if (np.dot(w, input.T) + b) * target <= 0:
        w += lr * target * input
        b += lr * target
        error = True
    return w, b, error


def sign(num):
    if num > 0:
        return 1
    else:
        return -1


def report(w, b, epochs, error, i):
    print('------------------------------')
    print('epochs:       \n  ' + (str(i)))
    print('learning rate:\n  ' + (str)(1.0 - error / epochs))
    print('w:            \n  ' + (str)(w))
    print('b:            \n  ' + (str)(b))


def test(w, b, test_data, test_target, test_error, test_time):
    for data, target in zip(test_data, test_target):
        if (np.dot(w, data) + b) * target <= 0:
            test_error += 1
        test_time += 1
    return test_error, test_time
        