import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Path has to lead to an .npy file
def load_target(path):
    target = np.load(path)
    return target


def run_saola(X,Y,w=0,param=0):

    #prepare class attribute
    label = pd.DataFrame(Y).astype(float)

    #prepare attributes df
    data = pd.DataFrame(X).astype(float)

    return np.array(saola(data, label)), param

def saola(data,label):
    print(data.shape)
    selected_features_indexes = [False for ind in range(data.shape[1])]

    for i in range(data.shape[1]):
        corrC = data.iloc[:,i].corr(label.iloc[:,0])
        zTrans = np.arctanh(corrC)
        zScore = zTrans*np.sqrt(11300 - 3)
        if zScore < 1.96:
                continue
        else:
           if not any(selected_features_indexes):
               selected_features_indexes[i] = True
           else:
               selected_features_indexes = correlationF(data, label, selected_features_indexes, i)
    return [index for index in range(len(selected_features_indexes)) if selected_features_indexes[index]]



def correlationF(data, label, selected_features_indexes, current_selected_feature_index):
    for i in range(len(selected_features_indexes)):
        if not selected_features_indexes[i]:
            continue

        corr_yc = data.iloc[:, i].corr(label.iloc[:, 0])
        xTrans_yc = np.arctanh(corr_yc)
        Zy_c = xTrans_yc * np.sqrt(11300 - 3)

        corr_fc = data.iloc[:,current_selected_feature_index].corr(label.iloc[:, 0])
        xTrans_fc = np.arctanh(corr_fc)
        Zf_c = xTrans_fc * np.sqrt(11300 - 3)

        corr_fy = data.iloc[:, i].corr(data.iloc[:,current_selected_feature_index])
        xTrans_fy = np.arctanh(corr_fy)
        Zf_y = xTrans_fy * np.sqrt(11300 - 3)

        if Zy_c > Zf_c and Zf_y >= Zf_c:
            return selected_features_indexes

        if Zf_c > Zy_c and Zf_y >= Zy_c:
            selected_features_indexes[i] = False

    selected_features_indexes[current_selected_feature_index] = True
    return selected_features_indexes




#
# def saola(data,label):
#     selected_features_indexes = [0 for ind in range(data.shape[1])]
#     selected_features = []
#     df_list = pd.DataFrame()
#     for i in data:
#         corrC = data.iloc[:,i].corr(label.iloc[:,0])
#         zTrans = np.arctanh(corrC)
#         zScore = zTrans*np.sqrt(11300 - 3)
#         if zScore < 1.96:
#                 continue
#         else:
#            # selected_features.append(str(i))
#            df = data.iloc[:,i]
#            if df_list.empty:
#                df_list = pd.concat([df_list, df], axis=1, ignore_index=True)
#            else:
#                df, df_list = correlationF(df, df_list,label,selected_features)
#     return df_list
#
#
#
# def correlationF(df, df_list,label,selected_features):
#     i = 0
#     while i < len(df_list.columns):
#         corr_yc = df_list.iloc[:, i].corr(label.iloc[:, 0])
#         xTrans_yc = np.arctanh(corr_yc)
#         Zy_c = xTrans_yc * np.sqrt(11300 - 3)
#
#         corr_fc = df.corr(label.iloc[:, 0])
#         xTrans_fc = np.arctanh(corr_fc)
#         Zf_c = xTrans_fc * np.sqrt(11300 - 3)
#
#         corr_fy = df_list.iloc[:, i].corr(df)
#         xTrans_fy = np.arctanh(corr_fy)
#         Zf_y = xTrans_fy * np.sqrt(11300 - 3)
#
#         if Zy_c > Zf_c and Zf_y >= Zf_c:
#             return df, df_list
#         if Zf_c > Zy_c and Zf_y >= Zy_c:
#             df_list = df_list.drop(df_list.columns[i], axis=1)
#             i = 0
#         else:
#             i = i + 1
#     df_list = pd.concat([df_list, df], axis=1, ignore_index=True)
#     return df, df_list