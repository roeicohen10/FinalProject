import pandas as pd
import numpy as np
from scipy import stats
import math
from Model.OFS.ofs_ac import OnlineFeatureSelectionAC



class Saola(OnlineFeatureSelectionAC):
    """
    implementation of Saola algorithm
    """
    DEFAULT_PARAMS = {"alpha": 0.05}

    def __init__(self):
        super().__init__(name='SAOLA', parameters=Saola.DEFAULT_PARAMS)

    def run(self, X, Y):
        # prepare class attribute
        label = pd.DataFrame(Y).astype(float)
        # prepare attributes df
        data = pd.DataFrame(X).astype(float)

        return Saola.saola(data, label, self.parameters['alpha'])

    @classmethod
    def saola(cls, data, label, alpha=0.01):
        """
        main algorithm method
        :param data: pandas dataframe of features data
        :param label: pandas dataframe of target data
        :param alpha: significance level
        :return: numpy array of selected features
        """
        selected_features_indexes = set()
        N = data.shape[0]  # number of records in ds
        for i in range(data.shape[1]):
            column_data = data.iloc[:, i]
            # check if the feature is correlated to the target
            corrC = column_data.corr(label.iloc[:, 0])
            # non correlation situation
            if math.isnan(corrC):
                continue
            zTrans = np.arctanh(corrC)
            zScore = zTrans * np.sqrt(N - 3)

            alpha_z_score = abs(stats.norm.ppf(alpha))
            if zScore < alpha_z_score:
                continue

            if len(selected_features_indexes) == 0:
                selected_features_indexes.add(i)
            else:
                # remove uncorrelated features
                selected_features_indexes = cls.correlationF(data, label, selected_features_indexes, i, N)
        return list(selected_features_indexes)

    @classmethod
    def correlationF(cls, data, label, selected_features_indexes, current_selected_feature_index, N):
        """

        :param data: pandas dataframe of features data
        :param label: pandas dataframe of target data
        :param selected_features_indexes: list of selected features indexes
        :param current_selected_feature_index: newly added selected feature
        :param N: number or records in data
        :return: update list of selected features indexes
        """
        selected_features_indexes_copy = selected_features_indexes.copy()
        stop = False
        for i in selected_features_indexes:
            # correlation between newly added feature to label
            corr_yc = data.iloc[:, i].corr(label.iloc[:, 0])
            xTrans_yc = np.arctanh(corr_yc)
            Zy_c = xTrans_yc * np.sqrt(N - 3)

            # correlation between current feature to label
            corr_fc = data.iloc[:, current_selected_feature_index].corr(label.iloc[:, 0])
            xTrans_fc = np.arctanh(corr_fc)
            Zf_c = xTrans_fc * np.sqrt(N - 3)

            # correlation between newly added feature to current feature
            corr_fy = data.iloc[:, i].corr(data.iloc[:, current_selected_feature_index])
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
