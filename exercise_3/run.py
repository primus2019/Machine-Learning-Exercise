import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import validation_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from utils import *

def run(train_data, train_target, test_data, test_target, core_function, epochs=10):
    # np.random.seed(8)

    parell_table = parellize_target(train_target)
    indices_1, target = load_dataset(train_target, parell_table)
    indices_2, data   = load_dataset(train_data)
    # print(pd.DataFrame(indices_1).shape)
    # print(pd.DataFrame(indices_2).shape)
    assert indices_1 == indices_2, 'label does not match data'
    pdata, ptarget, categories = preprocessing(data, target)
    
    parell_table_test = parellize_target(test_target)
    indices_1, target = load_dataset(test_target, parell_table_test)
    indices_2, data   = load_dataset(test_data)
    # print(pd.DataFrame(indices_1).shape)
    # print(pd.DataFrame(indices_2).shape)
    assert indices_1 == indices_2, 'label does not match data'
    pdata_test, ptarget_test, categories_test = preprocessing(data, target)

    pdata = pdata.reshape(-1, 10)
    ptarget = ptarget[0].reshape(-1, 1)
    pdata_test = pdata_test.reshape(-1, 10)
    ptarget_test = ptarget_test[0].reshape(-1, 1)
    sample_size = pdata.shape[0]
    print(pdata.shape[0])
    print(ptarget.shape)
    assert parell_table == parell_table_test, "parell table not identical"
    for e in range(epochs):
        clf = SVC(kernel=core_function)
        tmp = list(zip(pdata, ptarget))
        random.shuffle(tmp)
        tdata, ttarget = zip(*tmp)
        tdata_train   = list(tdata)[ :int(0.8*sample_size)]
        ttarget_train = list(ttarget)[ :int(0.8*sample_size)]
        tdata_test    = list(tdata)[int(0.8*sample_size):]
        ttarget_test  = list(ttarget)[int(0.8*sample_size):]

        clf.fit(tdata_train, ttarget_train)
        with open(core_function + '_trainset_2_testset_2.log', 'a+') as file:
            file.write('training error:         ' + (str)(clf.score(tdata_train, ttarget_train)) + '\n')
            file.write('cross-validation error: ' + (str)(clf.score(tdata_test, ttarget_test)) + '\n')
            file.write('test error:             ' + (str)(clf.score(pdata_test, ptarget_test)) + '\n')
        
    # pdata        = np.array(pdata).reshape((len(pdata[0]), len(pdata)))
    # ptarget      = np.array(ptarget).reshape((-1, 1))
    # # pdata_test   = np.array(pdata_test).reshape((len(pdata_test[0]), len(pdata_test)))
    # ptarget_test = np.array(ptarget_test).reshape((-1, 1))
    # # print(pd.DataFrame(pdata).shape)
    # train_input, vali_input, train_target, vali_target = train_test_split(pdata, ptarget, test_size=0.1)
    # test_input, test_target = pdata_test, ptarget_test
    # print(pd.DataFrame(train_input).shape)

    # train_target = np.array([targets[0] for targets in train_target]).reshape((-1,1))
    # vali_target  = np.array([targets[0] for targets in vali_target]).reshape((-1,1))
    # test_target  = np.array([targets[0] for targets in test_target]).reshape((-1,1))
    
    # errors = []
    # accuracies_of_train = []
    # accuracies_of_vali  = []
    # accuracies_of_test  = []
    # # pdata = np.array(pdata).reshape((len(pdata[0]), len(pdata)))
    # # ptarget = np.array([ptarget[0]]).reshape(-1,1)
    # linear_svm = SVC(kernel='linear')
    # tdata, ttarget = [], []
    # for idx, (data, target) in enumerate(zip(train_input, train_target)):
    #     print(pd.DataFrame(tdata).shape)
    #     print(pd.DataFrame(ttarget).shape)
    #     tdata.append(data)
    #     ttarget.append(target)
    #     linear_svm.fit(tdata, ttarget, core_function)
    #     prediction = linear_svm.predict(pdata_test)
    #     print(prediction)


    # # train_scores, test_scores = validation_curve(linear_svm, pdata.tolist()
    # , ptarget, param_name='alpha', param_range=np.logspace(0,1,2))

    
    # train_sizes, train_scores, test_scores = plot_learning_curve(linear_svm, pdata, ptarget)
    # print(train_sizes.shape)
    # print(train_scores.shape)
    # print(test_scores.shape)
    # linear_svm.fit(train_input, train_target)
    # result = linear_svm.predict(test_input)
    # print(result)
    # print(np.array(test_target))

    
    # title = "Learning Curves (Naive Bayes)"
    # # Cross validation with 100 iterations to get smoother mean test and train
    # # score curves, each time with 20% data randomly selected as a validation set.
    # cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

    # # estimator = GaussianNB()
    # # plot_learning_curve(estimator, title, pdata, ptarget, ylim=(0, 1), cv=cv, n_jobs=4)

    # title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
    # # SVC is more expensive so we do a lower number of CV iterations:
    # cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # estimator = SVC(gamma=0.001)
    # plot_learning_curve(estimator, title, pdata, ptarget, (0.7, 1.01), cv=cv, n_jobs=4)

    # plt.show()

    


if __name__ == '__main__':
    
    train_data   = transpose(unquote('train_10gene.csv'))    # trainset-1
    train_target = unquote('train_label.csv')
    
    test_data    = transpose(unquote('test2_10gene.csv'))            # testset-1
    test_target  = unquote('test2_label.csv')
    run(
        train_data      = train_data,
        train_target    = train_target,
        test_data       = test_data,
        test_target     = test_target,
        core_function   = 'poly'
    )

    # fuck = load_digits()
    # print(pd.DataFrame(fuck.data).shape, pd.DataFrame(fuck.target).shape)
    # print(pd.DataFrame(fuck.data).head(50), pd.DataFrame(fuck.target).head(50))
    
    # title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
    # # SVC is more expensive so we do a lower number of CV iterations:
    # cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # estimator = SVC(gamma=0.001)
    # plt = plot_learning_curve(estimator, title, fuck.data, fuck.target, (0, 1), cv=cv, n_jobs=4)
    # plt.show()