import numpy as np
import matplotlib.pyplot as plt
from utils import *
from network import MLP


def train(train_data, train_target, test_data, test_target, learning_rate=0.1, epochs=1000, file_name=None):
    bpn = MLP(input_size=10,output_size=2,hidden_size_1=40,hidden_size_2=20,learning_rate=learning_rate)
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

    assert parell_table == parell_table_test, "parell table not identical"
        
    # data, target, _ = LoadData("./GlassData.csv")
    # pdata, ptarget, categories = Preprocessing(data, target)
    train_size = pdata.shape[1]
    test_size  = pdata_test.shape[1]
    traininput = pdata[:,:int(0.9*train_size)]
    traintarget = ptarget[:,:int(0.9*train_size)]
    validationinput = pdata[:,int(0.9*train_size):int(train_size)]
    validationtarget = ptarget[:,int(0.9*train_size):int(train_size)]
    testinput = pdata_test
    testtarget = ptarget_test
    errors = []
    accuracies_of_train = []
    accuracies_of_vali = []
    accuracies_of_test = []
    with open('log/' + file_name + '.log', 'w+') as file:
        for e in range(epochs):
            train_correct_cnt = 0
            for i in range(traininput.shape[1]):

                out = bpn.forward(traininput[:,i:i+1])
                bpn.backward(traintarget[:,i:i+1], learning_rate=1-0.1*e/epochs)
                if np.argmax(out) == np.argmax(traintarget[:,i:i+1]):
                    train_correct_cnt += 1
            accuracy = train_correct_cnt / traininput.shape[1]
            if e % 50 == 0:
                file.write("train accuracy = {}".format(accuracy) + '\n')
            accuracies_of_train.append(accuracy)
            
            error = 0
            vali_correct_cnt = 0
            for i in range(validationinput.shape[1]):
                out = bpn.forward(validationinput[:,i:i+1])
                error += np.sum(np.abs(out-validationtarget[:,i:i+1]) )
                if np.argmax(out) == np.argmax(validationtarget[:,i:i+1]):
                    vali_correct_cnt += 1
            # print("error = {}".format(error))
            errors.append(error)
            accuracy = vali_correct_cnt / validationinput.shape[1]
            if e % 50 == 0:
                file.write("cross-validation accuracy = {}".format(accuracy) + '\n')
            accuracies_of_vali.append(accuracy) 
            # if accuracy > 0.78:
            #     break
            test_correct_cnt = 0
            for i in range(testinput.shape[1]):
                out = bpn.test(testinput[:,i:i+1])
                if np.argmax(out) == np.argmax(testtarget[:,i:i+1]):
                    # print(np.argmax(out))
                    test_correct_cnt += 1
            accuracy = test_correct_cnt/testinput.shape[1]
            if e % 50 == 0:
                file.write("test accuracy= {}".format(accuracy) + '\n')
            accuracies_of_test.append(accuracy) 
        # print("max accuracy of validation set is {}".format(max(accuracies_of_vali)))
        # x = range(len(accuracies_of_vali))
        # y = accuracies_of_vali
        # print(min(y))
        # plt.plot(x,y)
        # plt.savefig('lr_train_1_test_1' + (str)(learning_rate) + '.png')
        # plt.show()

        plt.title('accuracies')

        plt.plot(accuracies_of_train, label='train_acc')
        plt.plot(accuracies_of_vali,  label='vali_acc')
        plt.plot(accuracies_of_test,  label='test_acc')
        x_axis_len = max(len(accuracies_of_train), len(accuracies_of_vali), len(accuracies_of_test))
        plt.text(0, 1, 'max train accuracy: ' + (str)(max(accuracies_of_train)), transform=plt.gcf().transFigure)
        plt.text(0, 0.9, 'max vali  accuracy: ' + (str)(max(accuracies_of_vali)), transform=plt.gcf().transFigure)
        plt.text(0, 0.8, 'max test  accuracy: ' + (str)(max(accuracies_of_test)), transform=plt.gcf().transFigure)
        plt.legend(loc='lower right')    
        plt.subplots_adjust(left=0.3)
        plt.savefig('result/' + file_name + '_lr' + (str)(learning_rate) + '_e' + (str)(epochs) + '.png')
        plt.close()

if __name__ == '__main__':
    train(
        # train_data=transpose(unquote('train_10gene_sub.csv')),    # trainset-1
        # train_target=unquote('train_label_sub.csv'),
        train_data=transpose(unquote('train_10gene.csv')),          # trainset-2
        train_target=unquote('train_label.csv'),
        # test_data=transpose(unquote('test_10gene.csv')),            # testset-1
        # test_target=unquote('test_label.csv'),
        test_data=transpose(unquote('test2_10gene.csv')),            # testset-2
        test_target=unquote('test2_label.csv'),
        learning_rate=0.001,
        epochs=100,
        file_name='tr2_te2'
    )
    train(
        # train_data=transpose(unquote('train_10gene_sub.csv')),    # trainset-1
        # train_target=unquote('train_label_sub.csv'),
        train_data=transpose(unquote('train_10gene.csv')),          # trainset-2
        train_target=unquote('train_label.csv'),
        # test_data=transpose(unquote('test_10gene.csv')),            # testset-1
        # test_target=unquote('test_label.csv'),
        test_data=transpose(unquote('test2_10gene.csv')),            # testset-2
        test_target=unquote('test2_label.csv'),
        learning_rate=0.001,
        epochs=500,
        file_name='tr2_te2'
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

    
    