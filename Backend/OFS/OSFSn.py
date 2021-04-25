from scipy import stats
import pandas as pd
import numpy as np
from itertools import combinations
import math
def run_osfs(X,Y,w=0,param=0):

    #prepare class attribute
    label = pd.DataFrame(Y).astype(float)

    #prepare attributes df
    data = pd.DataFrame(X).astype(float)
    if not param['fast']:
        return np.array(osfs(data, label,param['alpha'])), param
    return np.array(fast_osfs(data, label, param['alpha'])), param

def osfs(data,label,alpha=0.01):
    selected_features_indexes = set()
    N = data.shape[0]
    for i in range(data.shape[1]):
        print(i)
        print(f'Number of features:{len(selected_features_indexes)}')
        column_data = data.iloc[:,i]
        pearson_corr = column_data.corr(label.iloc[:,0])
        if math.isnan(pearson_corr):
            continue

        z_score = 0.5 * np.log((1 + pearson_corr) / (1 - pearson_corr))
        z_score = np.sqrt(N - 3) * z_score
        alpha_z_score = stats.norm.ppf(alpha)

        if abs(z_score) < abs(alpha_z_score):
            continue

        #add feature
        selected_features_indexes.add(i)
        # remove depended features
        selected_features_indexes_copy = selected_features_indexes.copy()
        for index in selected_features_indexes:
            selected_features_indexes_copy.remove(index)

            if not check_dep_features(data,label,alpha,index,selected_features_indexes_copy):
                selected_features_indexes_copy.add(index)

        selected_features_indexes = selected_features_indexes_copy

    return list(selected_features_indexes)


def fast_osfs(data,label,alpha=0.01):
    selected_features_indexes = set()
    N = data.shape[0]
    for i in range(data.shape[1]):
        print(i)
        print(f'Number of features:{len(selected_features_indexes)}')
        column_data = data.iloc[:,i]
        pearson_corr = column_data.corr(label.iloc[:,0])
        if math.isnan(pearson_corr):
            continue

        z_score = 0.5 * np.log((1 + pearson_corr) / (1 - pearson_corr));
        z_score = np.sqrt(N - 3) * z_score
        alpha_z_score = stats.norm.ppf(alpha)

        if abs(z_score) < abs(alpha_z_score):
            continue

        if check_dep_features(data, label, alpha, i, selected_features_indexes):
            continue

        #add feature
        selected_features_indexes.add(i)
        # remove depended features
        selected_features_indexes_copy = selected_features_indexes.copy()
        for index in selected_features_indexes:
            if index == i:
                continue
            selected_features_indexes_copy.remove(index)

            if not check_dep_features(data,label,alpha,index,selected_features_indexes_copy):
                selected_features_indexes_copy.add(index)

        selected_features_indexes = selected_features_indexes_copy
    return list(selected_features_indexes)

def check_dep_features(data,label,alpha,feature_index,selected_features_indexes):
    N = data.shape[0]
    for L in range(1,len(selected_features_indexes) + 1):
        for subset in combinations(selected_features_indexes,L):
            x1 = data.iloc[:,feature_index].to_frame()
            x2 = label.iloc[:,0].to_frame()
            x3 = data.iloc[:,list(subset)]

            cov_df = pd.concat([x1,x2,x3],axis=1)
            cov_df = cov_df.corr(method='pearson')

            pearson_corr = get_pearson_corr(cov_df.to_numpy(),feature_id=0,label_id=1,subset_ids=list(range(2,cov_df.shape[1])))

            z_score = 0.5 * np.log((1 + pearson_corr) / (1 - pearson_corr));
            z_score = np.sqrt(N-L- 3) * z_score
            alpha_z_score = stats.norm.ppf(alpha)

            if abs(z_score) < abs(alpha_z_score):
                return True
    return False


def get_pearson_corr(cov_np,feature_id=0,label_id=1,subset_ids =[]):
    feature_id_2, label_id_2 = 0,1
    X = [feature_id,label_id]
    S2 =  cov_np[np.ix_(X, X)] - cov_np[np.ix_(X,subset_ids)]@np.linalg.inv(cov_np[np.ix_(subset_ids,subset_ids)])@cov_np[np.ix_(subset_ids, X)]
    c = S2[feature_id_2,label_id_2]
    r = c / np.sqrt(S2[feature_id_2, feature_id_2] * S2[label_id_2, label_id_2])
    return r

# %     0.8 1.0 -0.56;
# %     -0.4 -0.56 1.0];
# % r(1,3 | 2) = 0.0966
if __name__ == '__main__':
    import numpy as np
    from sklearn.model_selection import KFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    data = pd.read_csv('E:/data/HIVA.csv')
    y = data.iloc[:,1617].to_numpy()
    X = data.iloc[:,0:1617].to_numpy()

    kf = KFold(n_splits=10,shuffle=False)

    accuracy = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        selected_features, l = run_osfs(X_train, y_train, param={'alpha': 0.01, 'fast': False})

        x_ft_train = X_train[:, selected_features]
        x_ft_test = X_test[:, selected_features]

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(x_ft_train, y_train)
        test_predictions = knn.predict(x_ft_test)
        score = accuracy_score(y_test, test_predictions)
        accuracy.append(score)

    print('Accuracy: %.3f (%.3f)' % (np.mean(accuracy), np.std(accuracy)))

