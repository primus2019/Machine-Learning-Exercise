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

    test_data, test_data_idx, test_data_col       = load_dataset(test_data_path)
    test_target, test_target_idx, test_target_col = load_dataset(test_target_path, test_data_idx)
    info(test_data, test_data_idx, test_data_col, test_target, False)
    test_target = set_targets(test_target)
    test_data = scale(test_data, -1, 1)

    w, b = initialize(train_data_col.shape[0])
    mlp = MLP(
        input_size=train_data_col.shape[0],
        output_size=test_data_col.shape[0],
        hidden_size_1=20,
        hidden_size_2=10,
        learning_rate=learning_rate
    )
    all_classified = ''
    for i in range(epochs):
        train_error, test_error, train_time, test_time, train_accuracies, test_accuracies = 0.0, 0.0, 0.0, 0.0, [], []
        for data, target in zip(train_data, train_target):
            out = mlp.forward(np.reshape(data, (data.shape[0], 1)))
            mlp.backward(target)
            if np.argmax(out) != np.argmax(target):
                train_error += 1
            print('train_acc: ' + (str)(1 - train_error / train_data.shape[0]))
    #         if error:
    #             train_error += 1
    #         train_time += 1
    #     train_accuracies.append(1 - train_error / train_time)
    #     test_error, test_time = test(w, b, test_data, test_target, test_error, test_time)
    #     test_accuracies.append(1 - test_error / test_time)

    #     if i % 300 == 0:
    #         report(w, b, train_time, train_error, i)
    #         # add test acc to report
    #     if temp == train_error:
    #         all_classified = ' all_classified'
    #         report(w, b, train_time, train_error, i)
    #         break
    #     # if train_error / train_time < 0.05:
    #     #     report(w, b, train_time, train_error)
    #     #     break
    

    # x_axis_len = max(len(train_accuracies), len(test_accuracies))
    # # plt.axis(xmin=0, xmax=epochs)
    # plt.title('accuracies' + all_classified) ### notice

    # x = range(len(train_accuracies))
    # y = train_accuracies
    # plt.plot(y, label='train_acc')
    
    # x = range(len(test_accuracies))
    # y = test_accuracies
    # plt.plot(y, label='test_acc')

    # plt.text(x_axis_len / 5, 0.9 - 0.01, 'max train accuracy: ' + (str)(max(train_accuracies)))
    # plt.text(x_axis_len / 5, 0.9 - 0.02, train_data_path)
    # plt.text(x_axis_len / 5, 0.9 - 0.03, train_target_path)
    # # if fixed_increment:
    # #     plt.savefig(r'result\f_'+ file_name + '_lr' + (str)(learning_rate) + '_e' + (str)(epochs) + '.png')
    # # else:
    # #     plt.savefig(r'result\v_'+ file_name + '_lr' + (str)(learning_rate) + '_e' + (str)(epochs) + '.png')
    # # plt.show()
    # plt.text(x_axis_len / 5, 0.9 - 0.04, 'max test accuracy: ' + (str)(max(test_accuracies)))
    # plt.text(x_axis_len / 5, 0.9 - 0.05, test_data_path)
    # plt.text(x_axis_len / 5, 0.9 - 0.06, test_target_path)
    # plt.legend(loc='lower right')

    # if fixed_increment:
    #     plt.savefig(r'result\f_'+ file_name + '_lr' + (str)(learning_rate) + '_e' + (str)(epochs) + '.png')
    # else:
    #     plt.savefig(r'result\v_'+ file_name + '_lr' + (str)(learning_rate) + '_e' + (str)(epochs) + '.png')
        
    # plt.close()


if __name__ == '__main__':
    # train_data_path   = r'data\train_1_data.csv'            # trainset-1
    # train_target_path = r'data\train_1_target.csv'
    train_data_path   = r'data\train_2_data.csv'            # trainset-2
    train_target_path = r'data\train_2_target.csv'
    # test_data_path    = r'data\test_1_data.csv'             # testset-1
    # test_target_path  = r'data\test_1_target.csv'
    test_data_path    = r'data\test_2_data.csv'             # testset-2
    test_target_path  = r'data\test_2_target.csv'
    file_name = 'tr2_te2'
    run(
        train_data_path=train_data_path,
        train_target_path=train_target_path,
        test_data_path=test_data_path,
        test_target_path=test_target_path,
        learning_rate=0.001,
        epochs=10,
        file_name=file_name,
        fixed_increment=True
    )
    # run(
    #     train_data_path=train_data_path,
    #     train_target_path=train_target_path,
    #     test_data_path=test_data_path,
    #     test_target_path=test_target_path,
    #     learning_rate=0.001,
    #     epochs=500,
    #     file_name=file_name,
    #     fixed_increment=True
    # )
    # run(
    #     train_data_path=train_data_path,
    #     train_target_path=train_target_path,
    #     test_data_path=test_data_path,
    #     test_target_path=test_target_path,
    #     learning_rate=0.1,
    #     epochs=100,
    #     file_name=file_name,
    #     fixed_increment=False
    # )
    # run(
    #     train_data_path=train_data_path,
    #     train_target_path=train_target_path,
    #     test_data_path=test_data_path,
    #     test_target_path=test_target_path,
    #     learning_rate=0.1,
    #     epochs=500,
    #     file_name=file_name,
    #     fixed_increment=False
    # )