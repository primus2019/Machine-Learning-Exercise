from utils.Load import *
from utils.Info import *
from utils.Train import *
from utils.Preprocess import *
import matplotlib.pyplot as plt

def run(
    train_data_path, train_target_path, test_data_path, test_target_path,
    epochs=1000, learning_rate=0.01,
    file_name='temp', fixed_increment=True
):
    train_data, train_data_idx, train_data_col       = load_dataset(train_data_path)
    train_target, train_target_idx, train_target_col = load_dataset(train_target_path, train_data_idx)
    info(train_data, train_data_idx, train_data_col, train_target, False)
    train_target = set_targets(train_target)
    train_data = scale(train_data, -1, 1)

    w, b = initialize(train_data_col.shape[0])
    train_error, train_time, train_accuracies, test_accuracies = 0.0, 0.0, [], []
    all_classified = ''
    for i in range(epochs):
        temp = train_error
        for data, target in zip(train_data, train_target):
            w, b, error = train(data, target, w, b, i, learning_rate, True)
            if error:
                train_error += 1
            train_time += 1
        train_accuracies.append(1 - train_error / train_time)
        test_accuracies.append(1 - test_error / test_size)

        if i % 300 == 0:
            report(w, b, train_time, train_error, i)
            # add test acc to report
        if temp == train_error:
            all_classified = ' all_classified'
            report(w, b, train_time, train_error, i)
            break
        # if train_error / train_time < 0.05:
        #     report(w, b, train_time, train_error)
        #     break
    
    x = range(len(train_accuracies))
    y = train_accuracies
    max_acc = max(train_accuracies)
    plt.plot(x,y)
    plt.title('train accuracies' + all_classified)
    plt.text(100, max_acc - 0.01, 'max traning accuracy: ' + (str)(max(train_accuracies)))
    plt.text(100, max_acc - 0.02, train_data_path)
    plt.text(100, max_acc - 0.03, train_target_path)
    if fixed_increment:
        plt.savefig('result/f_'+ file_name + '_lr' + (str)(learning_rate) + '_e' + (str)(epochs) + '.png')
    else:
        plt.savefig('result/v_'+ file_name + '_lr' + (str)(learning_rate) + '_e' + (str)(epochs) + '.png')
    plt.show()


if __name__ == '__main__':
    train_data_path   = r'data\train_1_data.csv'            # trainset-1
    train_target_path = r'data\train_1_target.csv'
    # train_data_path   = r'data\train_2_data.csv'            # trainset-2
    # train_target_path = r'data\train_2_target.csv'
    test_data_path    = r'data\test_1_data.csv'             # testset-1
    test_target_path  = r'data\test_1_target.csv'
    # test_data_path    = r'data\test_2_data.csv'             # testset-2
    # test_target_path  = r'data\test_2_target.csv'
    run(
        train_data_path=train_data_path,
        train_target_path=train_target_path,
        test_data_path=test_data_path,
        test_target_path=test_target_path,
        learning_rate=0.001,
        epochs=300,
        file_name='tr1_te1',
        fixed_increment=True
    )
    run(
        train_data_path=train_data_path,
        train_target_path=train_target_path,
        test_data_path=test_data_path,
        test_target_path=test_target_path,
        learning_rate=0.001,
        epochs=500,
        file_name='tr1_te1',
        fixed_increment=True
    )
    run(
        train_data_path=train_data_path,
        train_target_path=train_target_path,
        test_data_path=test_data_path,
        test_target_path=test_target_path,
        learning_rate=0.1,
        epochs=300,
        file_name='tr1_te1',
        fixed_increment=False
    )
    run(
        train_data_path=train_data_path,
        train_target_path=train_target_path,
        test_data_path=test_data_path,
        test_target_path=test_target_path,
        learning_rate=0.1,
        epochs=500,
        file_name='tr1_te1',
        fixed_increment=False
    )