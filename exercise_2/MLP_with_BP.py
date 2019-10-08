import numpy as np
import matplotlib.pyplot as plt
from utils import *
from network import MLP


def train(
    train_data, train_target, test_data, test_target
):
    bpn = MLP(input_size=10,output_size=2,learning_rate=0.01)
    # x = np.array([1.52101,13.64,4.49,1.1,71.78,0.06,8.75,0,0], dtype=np.float64).reshape(-1,1)
    # print(bpn.forward(x))
    # t = np.array([0,1,0,0,0,0]).reshape(-1,1)
    # bpn.backward(t)

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
        
    epochs = 1000
    # data, target, _ = LoadData("./GlassData.csv")
    # pdata, ptarget, categories = Preprocessing(data, target)
    train_size = pdata.shape[1]
    test_size  = pdata_test.shape[1]
    traininput = pdata[:,:int(train_size)]
    traintarget = ptarget[:,:int(train_size)]
    validationinput = pdata[:,:int(train_size)]
    validationtarget = ptarget[:,:int(train_size)]
    testinput = pdata_test
    testtarget = ptarget_test
    errors = []
    accuracies_of_vali = []
    for e in range(epochs):
        correct_cnt_train = 0
        for i in range(traininput.shape[1]):

            out = bpn.forward(traininput[:,i:i+1])
            bpn.backward(traintarget[:,i:i+1], learning_rate=1-0.1*e/epochs)
            if np.argmax(out) == np.argmax(traintarget[:,i:i+1]):
                correct_cnt_train += 1
        print("accuracy of trainset = {}".format(correct_cnt_train/traininput.shape[1]))
        error = 0
        correct_cnt = 0
        for i in range(validationinput.shape[1]):
            out = bpn.forward(validationinput[:,i:i+1])
            error += np.sum(np.abs(out-validationtarget[:,i:i+1]) )
            if np.argmax(out) == np.argmax(validationtarget[:,i:i+1]):
                correct_cnt += 1
        # print("error = {}".format(error))
        errors.append(error)
        accuracy = correct_cnt/testinput.shape[1]
        print("accuracy of validationset = {}".format(accuracy))
        accuracies_of_vali.append(accuracy) 
        # if accuracy > 0.78:
        #     break
    test_correct_cnt = 0
    for i in range(testinput.shape[1]):
        out = bpn.forward(testinput[:,i:i+1])
        if np.argmax(out) == np.argmax(testtarget[:,i:i+1]):
            print(np.argmax(out))
            test_correct_cnt += 1
    print("testset accuracy is = {}".format(test_correct_cnt/testinput.shape[1])) 
    print("max accuracy of validation set is {}".format(max(accuracies_of_vali)))
    x = range(len(accuracies_of_vali))
    y = accuracies_of_vali
    print(min(y))
    plt.plot(x,y)
    plt.show()

if __name__ == '__main__':
    train(
        train_data=transpose('train_10gene_sub.csv'),
        train_target='train_10gene_label_sub.csv',
        test_data=transpose(unquote('test_10gene.csv')),
        test_target=unquote('test_label.csv')
    )
    # train_target = 'train_10gene_label_sub.csv'
    # train_data   = transpose('train_10gene_sub.csv')

    # parell_table = parellize_target(train_target)

    # indices_1, target = load_dataset(train_target, parell_table)
    # # print(indices_1)
    # # print(target)
    # indices_2, data   = load_dataset(train_data)
    # assert indices_1 == indices_2, 'label does not match data'
    # pdata, ptarget, categories = preprocessing(data, target)
    # print(pd.DataFrame(ptarget).info())

    
    