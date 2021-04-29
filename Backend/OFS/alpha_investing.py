import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import Backend.OFS.Saola as sa
import Backend.OFS.OSFSn as osfs
def alpha_investing(X, y, w0, dw):
    """
    This function implements streamwise feature selection (SFS) algorithm alpha_investing for binary regression or
    univariate regression
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, assume feature arrives one at each time step
    y: {numpy array}, shape (n_samples,)
        input class labels or regression target
    Output
    ------
    F: {numpy array}, shape (n_selected_features,)
        index of selected features in a streamwise way
    Reference
    ---------
    Zhou, Jing et al. "Streaming Feature Selection using Alpha-investing." KDD 2006.
    """

    n_samples, n_features = X.shape
    w = w0
    F = []  # selected features
    for i in range(n_features):
        x_can = X[:, i]  # generate next feature
        alpha = w / 2 / (i + 1)
        X_old = X[:, F]
        if i is 0:
            X_old = np.ones((n_samples, 1))
            linreg_old = linear_model.LinearRegression()
            linreg_old.fit(X_old, y)
            error_old = 1 - linreg_old.score(X_old, y)
        if i is not 0:
            # model built with only X_old
            linreg_old = linear_model.LinearRegression()
            linreg_old.fit(X_old, y)
            error_old = 1 - linreg_old.score(X_old, y)

        # model built with X_old & {x_can}
        X_new = np.concatenate((X_old, x_can.reshape(n_samples, 1)), axis=1)
        logreg_new = linear_model.LinearRegression()
        logreg_new.fit(X_new, y)
        error_new = 1 - logreg_new.score(X_new, y)

        # calculate p-value
        pval = np.exp((error_new - error_old) / (2 * error_old / n_samples))
        if pval < alpha:
            F.append(i)
            w = w + dw - alpha
        else:
            w -= alpha

        if i==0 and len(F) == 0:
            F.append(i)
    return np.array(F)

def run_AI(X,Y,param):
    dw=param['dw']
    w0=param['alpha']
    selected_features = alpha_investing(X,Y,w0,dw)
    print(selected_features)
    return selected_features,param


def read_doth(file_name,dim,type=int):
    all_records = []
    with open(f'C:/Users/Roi/Documents/Degree/Semester 8/FinalProject/data/validation/{file_name}','r') as f:
        for line in f:
            line = line.rstrip().lstrip()
            dim_list = [0]*dim
            for index_tup in line.split(" "):
                index_tup = index_tup.split(":")
                dim_list[int(index_tup[0]) -1] = type(index_tup[1])

            all_records.append(np.array(dim_list))

    return np.array(all_records)


if __name__ == '__main__':

    # ds_name = 'MADELON'
    # x_train = np.loadtxt(r'C:\Users\Roi\Documents\Degree\Semester 8\FinalProject\data\validation\MADELON\madelon_train.data',dtype=np.int)
    # y_train = np.loadtxt(r'C:\Users\Roi\Documents\Degree\Semester 8\FinalProject\data\validation\MADELON\madelon_train.labels',dtype=np.int)
    # x_test = np.loadtxt(
    #     r'C:\Users\Roi\Documents\Degree\Semester 8\FinalProject\data\validation\MADELON\madelon_valid.data',
    #     dtype=np.int)
    # y_test = np.loadtxt(
    #     r'C:\Users\Roi\Documents\Degree\Semester 8\FinalProject\data\validation\MADELON\madelon_valid.labels',
    #     dtype=np.int)

    # ds_name = 'DEXTER'
    # typ = np.int
    # x_train = read_doth('DEXTER\dexter_train.data',20000,type=typ)
    # y_train = np.loadtxt(r'C:\Users\Roi\Documents\Degree\Semester 8\FinalProject\data\validation\DEXTER\dexter_train.labels',dtype=int)
    # x_test = read_doth('DEXTER\dexter_valid.data',20000,type=typ)
    # y_test = np.loadtxt(
    #     r'C:\Users\Roi\Documents\Degree\Semester 8\FinalProject\data\validation\DEXTER\dexter_valid.labels',
    #     dtype=int)
    ds_name = 'LEUKEMIA'
    typ = np.float
    x_train = read_doth('LEUKEMIA\leukemia_train.data', 7129, type=typ)
    y_train = np.loadtxt(
        r'C:\Users\Roi\Documents\Degree\Semester 8\FinalProject\data\validation\LEUKEMIA\leukemia_train.labels', dtype=int)
    x_test = read_doth('LEUKEMIA\leukemia_valid.data', 7129, type=typ)
    y_test = np.loadtxt(
        r'C:\Users\Roi\Documents\Degree\Semester 8\FinalProject\data\validation\LEUKEMIA\leukemia_valid.labels',
        dtype=int)


    print(f'X_TRAIN:{x_train.shape}')
    print(f'Y_TRAIN:{y_train.shape}')
    print(f'X_TEST:{x_test.shape}')
    print(f'Y_TEST:{y_test.shape}')


    # selected_features = alpha_investing(x_train, y_train, 0.05, 0.05)
    # selected_features, l = sa.run_saola(x_train, y_train, 0.05, 0.05)
    selected_features, l = osfs.run_osfs(x_train, y_train, param={'alpha':0.01, 'fast':True})
    # z = sum(selected_features)
    x_ft_train = x_train[:,selected_features]
    x_ft_test = x_test[:, selected_features]

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(x_ft_train, y_train)
    test_predictions = knn.predict(x_ft_test)
    score = accuracy_score(y_test, test_predictions)
    print(f'DS: {ds_name}')
    print(f'Num Of Features:{len(selected_features)}')
    print(f'Accuracy:{score}')