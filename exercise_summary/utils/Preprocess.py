import numpy as np


def scale(X, x_min=-1, x_max=1):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom


def initialize(input_size):
    w = np.zeros(input_size)
    b = np.zeros(1)
    return w, b


def preprocess(data, target, min=-1, max=1):
    assert data.shape[0] == target.shape[0], "size doesn't match "
    processed_data = scale(np.array(data),min,max)
    categories = []
    for i,v in enumerate(target):
        if (v not in categories):
            categories.append(v)
    categories.sort()
    # print(categories) 
    nc = len(categories)
    nt = len(target)
    processedtarget = np.zeros((nc, nt))
    for i,v in enumerate(target):
        processedtarget[categories.index(v),i] = 1
    return processed_data.T, processedtarget, categories