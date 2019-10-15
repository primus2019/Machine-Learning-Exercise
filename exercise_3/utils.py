import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def transpose(file_name):
    dataset = pd.read_csv(file_name).T
    dataset.to_csv('transposed_' + file_name, header=False)

    return 'transposed_' + file_name

def data_detail(file_name):
    dataset = pd.read_csv(file_name)
    print(dataset.info())

def isFloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def parellize_target(file_name):
    target = []
    with open(transpose(file_name), 'r') as file:
        for indices, line in enumerate(file):
            if indices == 0:
                continue
            return list(set(line[:-1].split(',')[1:]))

def load_dataset(file_name, parell_table=None):

    store   = []
    indices = []
    data    = []
    with open(file_name, 'r') as file:
        for indix, line in enumerate(file):
            line = line[:-1].split(',')
            if indix == 0:
                continue    # pass the column line
            if parell_table is not None:
                line[1] = parell_table.index(line[1])
            store.append(line)
        for line in store:
            indices.append(line[0])
            # print(line[0])
            if parell_table is None:        # is data
                data.append(list(map(float, line[1:])))
            else:                           # is target
                data.append(int(line[1]))
    
    return indices, data


def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom

def preprocessing(data, target):
    assert len(data) == len(target), "size doesn't match "
    processeddata = scale(np.array(data),-1,1)
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
    return processeddata.T, processedtarget, categories

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def unquote(file_name):
    pd.read_csv(file_name, quotechar='"', header=None, index_col=None).to_csv('unquoted_' + file_name, header=False, index=False)
    return 'unquoted_' + file_name

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt