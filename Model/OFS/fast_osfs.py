import numpy as np
import pandas as pd
import math
from scipy import stats
from .osfs import OSFS
from .ofs_ac import OnlineFeatureSelectionAC




class FastOSFS(OnlineFeatureSelectionAC):
    """
    implementation of Fast OSFS algorithm
    """
    DEFAULT_PARAMS = {"alpha": 0.05}
    def __init__(self):
        super().__init__(name='Fast OSFS', parameters=FastOSFS.DEFAULT_PARAMS)

    def run(self, X, Y):
        # prepare class attribute
        label = pd.DataFrame(Y).astype(float)
        # prepare attributes df
        data = pd.DataFrame(X).astype(float)

        return FastOSFS.fast_osfs(data, label, self.parameters['alpha'])

    @classmethod
    def fast_osfs(cls, data, label, alpha):
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
            # check if the feature is correlated to the target
            column_data = data.iloc[:, i]
            pearson_corr = column_data.corr(label.iloc[:, 0])
            # non correlation situation
            if math.isnan(pearson_corr):
                continue
            z_score = 0.5 * np.log((1 + pearson_corr) / (1 - pearson_corr))
            z_score = np.sqrt(N - 3) * z_score
            alpha_z_score = stats.norm.ppf(alpha)

            if abs(z_score) < abs(alpha_z_score):
                continue

            # check if the new feature is dependent (difference from OSFS)
            if OSFS.check_dep_features(data, label, alpha, i, selected_features_indexes):
                continue

            # add feature
            selected_features_indexes.add(i)
            # remove depended features
            selected_features_indexes_copy = selected_features_indexes.copy()
            for index in selected_features_indexes:
                selected_features_indexes_copy.remove(index)

                if not OSFS.check_dep_features(data, label, alpha, index, selected_features_indexes_copy):
                    selected_features_indexes_copy.add(index)

            selected_features_indexes = selected_features_indexes_copy

        return list(selected_features_indexes)
