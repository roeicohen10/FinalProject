import pandas as pd
import numpy as np
from skmultiflow.lazy import KNNClassifier
from skmultiflow.neural_networks import PerceptronMask
from skmultiflow.bayes import NaiveBayes
import time

from skmultiflow.drift_detection import DDM
from Backend.OFS.OSFSn import run_osfs
from Backend.OFS.Saola import run_saola
from Backend.OFS.alpha_investing import run_AI
import matplotlib.pyplot as plt
from scipy.io import arff
from io import StringIO
import arff, numpy as np


# def build_new_model():
# new_model = KNNClassifier(n_neighbors=3, max_window_size=window_size)

def init_model(model, model_params, X_train, y_train, ofs_algo=None, ofs_params=None, window_size=30):
    ol_model =  NaiveBayes()

    selected_features = None
    ofs_running_time, ol_running_time = [], []

    if ofs_algo is None:
        start_t = time.perf_counter()
        ol_model = ol_model.fit(X_train, y_train)
        ol_running_time.append(time.perf_counter() - start_t)
    else:
        start_t = time.perf_counter()
        selected_features, params = ofs_algo(X_train, y_train,
                                             param=ofs_params)
        ofs_running_time.append(time.perf_counter() - start_t)

        while len(selected_features) == 0:
            start_t = time.perf_counter()
            ofs_params['alpha'] = ofs_params['alpha'] * 2
            selected_features, params = ofs_algo(X_train, y_train,
                                                 param=ofs_params)
            ofs_running_time.append(time.perf_counter() - start_t)

        print(f"selected features: {len(selected_features)}")

        start_t = time.perf_counter()
        ol_model = ol_model.fit(X_train[:, selected_features], y_train)
        ol_running_time.append(time.perf_counter() - start_t)

    return ol_model, selected_features, ol_running_time, ofs_running_time




def run_simulation(X, y, window_size=3, ofs_algo=None, ofs_params=None, lazy=True):
    corrects, n_samples, samples_counter= 0, 0, 0
    first_size_window, selected_features, prequential_accuracy, ol_running_time, ofs_running_time = [], [], [], [], []
    ddm = DDM()
    for record in range(X.shape[0]):
        record_x, record_y = np.array([X[record, :]]), np.array([y[record]])
        if record < window_size:
            first_size_window.append(record)
            continue

        if record == window_size:
            old_selected_features = selected_features
            ol_model, selected_features, ol_run_time, ofs_run_time = init_model(None, None, X[first_size_window, :],
                                                                                y[first_size_window], ofs_algo=ofs_algo,
                                                                                ofs_params=ofs_params,
                                                                                window_size=window_size)
            print(old_selected_features == selected_features)

            ol_running_time.extend(ol_run_time)
            if ofs_algo is not None:
                ofs_running_time.extend(ofs_run_time)

            continue

        my_pred = ol_model.predict(record_x) if ofs_algo is None else ol_model.predict(
            record_x[:, selected_features])
        samples_counter += 1
        if record_y[0] == my_pred[0]:
            corrects += 1

        ddm.add_element(corrects / samples_counter)
        prequential_accuracy.append(corrects / samples_counter)

        # if ddm.detected_change():
        if lazy:
            ol_model.partial_fit(record_x[:,selected_features], record_y)
        elif ddm.detected_change():
            old_selected_features = selected_features
            ol_model, selected_features, ol_run_time, ofs_run_time = init_model(None, None,
                                                                                X[record - window_size:record, :],
                                                                                y[record - window_size:record],
                                                                                ofs_algo=ofs_algo,
                                                                                ofs_params=ofs_params,
                                                                                window_size=window_size)
            ol_running_time.extend(ol_run_time)
            if ofs_algo is not None:
                ofs_running_time.extend(ofs_run_time)
            print(old_selected_features == selected_features)


    print('KNNClassifier usage example')
    print('{} samples analyzed.'.format(samples_counter))
    print("KNNClassifier's performance: {}".format(corrects / samples_counter))
    return prequential_accuracy, ol_running_time, ofs_running_time

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
    from sklearn.utils import shuffle

    # ds_name = 'MADELON'
    # x_train = np.loadtxt(r'C:\Users\Roi\Documents\Degree\Semester 8\FinalProject\data\validation\MADELON\madelon_train.data',dtype=np.int)
    # y_train = np.loadtxt(r'C:\Users\Roi\Documents\Degree\Semester 8\FinalProject\data\validation\MADELON\madelon_train.labels',dtype=np.int)
    # print(x_train.shape)
    # ds_name = 'MADELON'
    # data = pd.read_csv(r'E:\data\SPECT.csv', dtype=np.int)
    # data = shuffle(data)
    # x = data.iloc[:, []]
    # x_train = data.iloc[:, :22].to_numpy()
    # y_train = data.iloc[:, 22:].to_numpy()
    # x_test = np.loadtxt(
    #     r'C:\Users\Roi\Documents\Degree\Semester 8\FinalProject\data\validation\MADELON\madelon_valid.data',
    #     dtype=np.int)
    # y_test = np.loadtxt(
    #     r'C:\Users\Roi\Documents\Degree\Semester 8\FinalProject\data\validation\MADELON\madelon_valid.labels',
    #     dtype=np.int)
    # test_multi(x_train, y_train)
    # acc = test_multi(x_train, y_train, window_size=30, ofs_algo=run_saola, ofs_params={'alpha': 0.01, 'fast': True})

    # f = StringIO(r'C:\Users\Roi\Documents\Degree\Semester 8\פרוייקט גמר\datasets\new\ChlorineConcentration\ChlorineConcentration_TRAIN.arff')

    # dataset = arff.load(open(
    #     r'C:\Users\Roi\Documents\Degree\Semester 8\פרוייקט גמר\datasets\new\SmallKitchenAppliances\SmallKitchenAppliances_TRAIN.arff'))
    # data = np.array(dataset['data'])
    # print(data.shape)
    # x_train, y_train = data[:, :-1].astype(np.float), data[:, -1].astype(np.int)
    # ds_name = 'DEXTER'
    # typ = np.int
    x_train =  pd.read_csv('E:/data/X_train.csv').to_numpy()
    y_train = pd.read_csv('E:/data/y_train.csv').to_numpy()
    # y_train = np.loadtxt(r'C:\Users\Roi\Documents\Degree\Semester 8\FinalProject\data\validation\DEXTER\dexter_train.labels',dtype=int)



    print(x_train.shape)
    prequential_accuracy, ol_running_time, ofs_running_time = run_simulation(x_train, y_train, window_size=200,
                                                                         ofs_algo=run_osfs,
                                                                         ofs_params={'alpha': 0.05, 'dw': 0.05,
                                                                                     'fast': True})
    # prequential_accuracy, ol_running_time, ofs_running_time = run_simulation(x_train, y_train, window_size=150)

    print(f"Last accuracy: {prequential_accuracy[-1]}")
    print(f"Mean OL algorithm runtime: {np.mean(np.array(ol_running_time)) * 1000} ms")
    print(f"Mean OFS algorithm runtime: {np.mean(np.array(ofs_running_time)) * 1000} ms")

    iterations = list(range(len(prequential_accuracy)))
    plt.plot(iterations, prequential_accuracy)
    plt.show()
