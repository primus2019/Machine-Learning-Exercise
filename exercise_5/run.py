from utils import Load
import numpy as np
import sklearn
from sklearn.preprocessing import minmax_scale
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import xgboost as xgb


def run():
    fibro = 'data2/fibroblast.csv'
    myofi = 'data2/myofibroblast.csv'
    ## EDA
    ## Load.inspect(fibro)
    fibro_fe = Load.load(fibro)
    myofi_fe = Load.load(myofi)

    ## standardlize: MinMaxScaler
    fibro_fe = minmax_scale(fibro_fe)
    myofi_fe = minmax_scale(myofi_fe)
    print(np.shape(fibro_fe))
    print(np.shape(myofi_fe))
    save_var_mean(fibro_fe, 'before selection')
    save_var_mean(myofi_fe, 'before selection')

    ## feature selection: variance threshold
    fibro_fe = VarianceThreshold(threshold=0.01).fit_transform(fibro_fe)
    myofi_fe = VarianceThreshold(threshold=0.03).fit_transform(myofi_fe)
    print(np.shape(fibro_fe))
    print(np.shape(myofi_fe))
    save_var_mean(fibro_fe, 'after selection')
    save_var_mean(myofi_fe, 'after selection')

    ## dimension reduction: PCA
    fibro_fe = PCA(n_components=100).fit_transform(fibro_fe)
    myofi_fe = PCA(n_components=100).fit_transform(myofi_fe)
    print(np.shape(fibro_fe))
    print(np.shape(myofi_fe))
    save_var_mean(fibro_fe, 'after decomposition')
    save_var_mean(myofi_fe, 'after decomposition')
    
    ## build union dataset of two classes
    union_fe = np.concatenate((fibro_fe, myofi_fe), axis=0)
    union_lb = np.concatenate((np.ones(len(fibro_fe)), np.zeros(len(myofi_fe))), axis=0)
    union_ds = np.c_[union_fe, union_lb]
    print(np.shape(union_ds))
    np.random.shuffle(union_ds)
    print(np.shape(union_ds))

    ## split testset and trainset
    train_fe, test_fe, train_lb, test_lb = train_test_split(union_fe, union_lb, train_size=0.9, shuffle=True)
    print('shape of train feature: \t{}'.format(np.shape(train_fe)))
    print('shape of train label:   \t{}'.format(np.shape(train_lb)))
    print('shape of test  feature: \t{}'.format(np.shape(test_fe)))
    print('shape of test  label:   \t{}'.format(np.shape(test_lb)))

    ## train classifier
    model = xgb.XGBClassifier(learning_rate=0.01)
    model.fit(train_fe, train_lb)
    print(model.score(test_fe, test_lb))


def save_var_mean(fe, process):
    var = np.var(fe, 0, ddof=1)
    mean = np.mean(fe, 0)
    plt.scatter(mean, var)
    plt.xlabel('mean')
    plt.ylabel('variance')
    plt.savefig('img/var-mean curve_{}.png'.format(process))
    plt.close()


def save_max_min(x, y, title, file_name, xlabel, ylabel):
    plt.plot(x, y, '-o-')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    max_idx = np.argmax(y) + 1
    max_val = round(np.max(y), 2)
    min_idx = np.argmin(y) + 1
    min_val = round(np.min(y), 2)
    plt.plot(max_idx, max_val, marker='^')
    plt.plot(min_idx, min_val, marker='v')
    plt.annotate('({}, {})'.format(max_idx, max_val), (max_idx, max_val))
    plt.annotate('({}, {})'.format(min_idx, min_val), (min_idx, min_val))
    plt.savefig('img/{}.png'.format(file_name))
    plt.close()


if __name__ == '__main__':
    run()