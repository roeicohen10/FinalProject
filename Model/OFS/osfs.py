import numpy as np
import pandas as pd
import math
from scipy import stats
from itertools import combinations
from Model.OFS.ofs_ac import OnlineFeatureSelectionAC




class OSFS(OnlineFeatureSelectionAC):
    """
    implementation of OSFS algorithm
    """
    DEFAULT_PARAMS = {"alpha": 0.05}
    def __init__(self):
        super().__init__(name='OSFS', parameters=OSFS.DEFAULT_PARAMS)

    def run(self, X, Y):
        # prepare class attribute
        label = pd.DataFrame(Y).astype(float)
        # prepare attributes df
        data = pd.DataFrame(X).astype(float)

        return OSFS.osfs(data, label, self.parameters['alpha'])

    @classmethod
    def osfs(cls, data, label, alpha):
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

            # add feature
            selected_features_indexes.add(i)
            # remove depended features
            selected_features_indexes_copy = selected_features_indexes.copy()
            for index in selected_features_indexes:
                selected_features_indexes_copy.remove(index)

                if not cls.check_dep_features(data, label, alpha, index, selected_features_indexes_copy):
                    selected_features_indexes_copy.add(index)

            selected_features_indexes = selected_features_indexes_copy

        return list(selected_features_indexes)

    @classmethod
    def check_dep_features(cls, data, label, alpha, feature_index, selected_features_indexes):
        """
        method to remove deepened feature from the selected features list
        :param data: pandas dataframe of features data
        :param label: pandas dataframe of target data
        :param alpha: significance level
        :param feature_index: current feature index to check if deepened
        :param selected_features_indexes: list of current selected features
        :return: boolean indicator if the feature is deepened or not
        """
        N = data.shape[0]  # number of records in ds
        for L in range(1, len(selected_features_indexes) + 1):  # creates all possible subsets of the selected features
            for subset in combinations(selected_features_indexes, L):
                x1 = data.iloc[:, feature_index].to_frame()
                x2 = label.iloc[:, 0].to_frame()
                x3 = data.iloc[:, list(subset)]

                # calculate covariance matrix
                cov_df = pd.concat([x1, x2, x3], axis=1)
                cov_df = cov_df.corr(method='pearson')

                pearson_corr = OSFS.get_pearson_corr(cov_df.to_numpy(), feature_id=0, label_id=1,
                                                     subset_ids=list(range(2, cov_df.shape[1])))

                # check if the feature is significance
                z_score = 0.5 * np.log((1 + pearson_corr) / (1 - pearson_corr));
                z_score = np.sqrt(N - L - 3) * z_score
                alpha_z_score = stats.norm.ppf(alpha)

                if abs(z_score) < abs(alpha_z_score):
                    return True
        return False

    @staticmethod
    def get_pearson_corr(cov_np, feature_id=0, label_id=1, subset_ids=[]):
        """
        method which calculates the pearson correlation between matrix to vector
        :param cov_np: covariance matrix
        :param feature_id: index of feature to check correlation in matrix
        :param label_id: index of target in matrix
        :param subset_ids: indexes of feature subsets in matrix
        :return: correlation value - float
        """
        feature_id_2, label_id_2 = 0, 1
        X = [feature_id, label_id]
        S2 = cov_np[np.ix_(X, X)] - cov_np[np.ix_(X, subset_ids)] @ np.linalg.inv(
            cov_np[np.ix_(subset_ids, subset_ids)]) @ cov_np[np.ix_(subset_ids, X)]
        c = S2[feature_id_2, label_id_2]
        r = c / np.sqrt(S2[feature_id_2, feature_id_2] * S2[label_id_2, label_id_2])
        return r
