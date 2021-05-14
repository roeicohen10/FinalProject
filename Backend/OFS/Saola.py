import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score
from scipy import stats
import math



SAOLA_PARAMS = {"alpha": 0.05}


#Path has to lead to an .npy file
def load_target(path):
    target = np.load(path)
    return target


def run_saola(X,Y,w=0,param=0):

    #prepare class attribute
    label = pd.DataFrame(Y).astype(float)

    #prepare attributes df
    data = pd.DataFrame(X).astype(float)

    return np.array(saola(data, label,alpha=param['alpha'])), param


def saola(data,label,alpha = 0.01):
    selected_features_indexes = set()
    N = data.shape[0]
    for i in range(data.shape[1]):
        column_data = data.iloc[:,i]

        corrC = column_data.corr(label.iloc[:,0])
        if math.isnan(corrC):
            continue
        zTrans = np.arctanh(corrC)
        zScore = zTrans*np.sqrt(N - 3)

        alpha_z_score = abs(stats.norm.ppf(alpha))
        if zScore < alpha_z_score:
            continue

        if sum(selected_features_indexes) == 0:
           selected_features_indexes.add(i)
        else:
            selected_features_indexes = correlationF(data, label, selected_features_indexes, i,N)
    return list(selected_features_indexes)



def correlationF(data, label, selected_features_indexes, current_selected_feature_index,N):
    selected_features_indexes_copy = selected_features_indexes.copy()
    stop = False
    for i in selected_features_indexes:
        corr_yc = data.iloc[:, i].corr(label.iloc[:, 0])
        xTrans_yc = np.arctanh(corr_yc)
        Zy_c = xTrans_yc * np.sqrt(N - 3)

        corr_fc = data.iloc[:,current_selected_feature_index].corr(label.iloc[:, 0])
        xTrans_fc = np.arctanh(corr_fc)
        Zf_c = xTrans_fc * np.sqrt(N - 3)

        corr_fy = data.iloc[:, i].corr(data.iloc[:,current_selected_feature_index])
        xTrans_fy = np.arctanh(corr_fy)
        Zf_y = xTrans_fy * np.sqrt(N - 3)

        if Zy_c > Zf_c and Zf_y >= Zf_c:
            stop = True
            break

        if Zf_c > Zy_c and Zf_y >= Zy_c:
            selected_features_indexes_copy.remove(i)

    if not stop:
        selected_features_indexes_copy.add(current_selected_feature_index)

    return selected_features_indexes_copy
