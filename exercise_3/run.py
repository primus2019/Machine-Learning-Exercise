import numpy as np
import pandas as pd
import matplotlib.pyplot
# from sklearn import svm
from sklearn.model_selection import train_test_split
from utils import *

def run(train_data, train_target, test_data, test_target):
    # np.random.seed(8)
    train_target    = unquote(train_target)
    train_data      = transpose(unquote(train_data))
    test_target     = unquote(test_target)
    test_data       = transpose(unquote(test_data))

    parell_table = parellize_target(train_target)
    indices_1, target = load_dataset(train_target, parell_table)
    indices_2, data   = load_dataset(train_data)
    assert indices_1 == indices_2, 'label does not match data'
    pdata, ptarget, categories = preprocessing(data, target)
    
    parell_table_test = parellize_target(test_target)
    indices_1, target = load_dataset(test_target, parell_table_test)
    indices_2, data   = load_dataset(test_data)
    assert indices_1 == indices_2, 'label does not match data'
    pdata_test, ptarget_test, categories_test = preprocessing(data, target)

    assert parell_table == parell_table_test, "parell table not identical"
    
    pdata_train, pdata_vali, ptarget_train, ptarget_vali = train_test_split(pdata, ptarget, test_size=0.1)
    print(pd.DataFrame(pdata_train).shape)
    print(pd.DataFrame(pdata_test).shape)
    print(pd.DataFrame(ptarget_train).shape)
    print(pd.DataFrame(ptarget_test).shape)
    # lvm = svm.SVC(kernel='lin
    # ear')
    


if __name__ == '__main__':
    
    train_data=transpose(unquote('train_10gene_sub.csv'))    # trainset-1
    train_target=unquote('train_label_sub.csv')
    
    test_data=transpose(unquote('test_10gene.csv'))            # testset-1
    test_target=unquote('test_label.csv')
    run(
        train_data = train_data,
        train_target = train_target,
        test_data = test_data,
        test_target = test_target
    )