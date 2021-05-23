from skmultiflow.lazy import KNNClassifier
from skmultiflow.neural_networks import PerceptronMask
from skmultiflow.bayes import NaiveBayes
from skmultiflow.meta import AdaptiveRandomForestClassifier



OL_MODELS = [
    {
    "name": "Neural Network",
    "func": PerceptronMask,
    "params": {},
    "fit_params":{"classes":[1,2]},
    "lazy": False,
    "params_str": 'alpha=0.0001, max_iter=1000, tol=0.001, random_state=0, early_stopping=False'
},
    {
    "name": "K-Nearest Neighbors 3",
    "func": KNNClassifier,
    "params": {"n_neighbors":3, "max_window_size":100},
    "fit_params":{},
    "lazy": True,
    "params_str":'K=3, leaf_size=30, metric=Euclidean'
},
    {
    "name": "K-Nearest Neighbors 5",
    "func": KNNClassifier,
    "params": {"n_neighbors":5, "max_window_size":100},
    "fit_params":{},
    "lazy": True,
    "params_str":'K=5, leaf_size=30, metric=Euclidean'
},
    {
    "name": "Naive Bayes",
    "func": NaiveBayes,
    "params": {},
    "fit_params":{},
    "lazy": True,
    "params_str":''
},
    {
    "name": "Random Forest",
    "func": AdaptiveRandomForestClassifier,
    "params": {},
    "fit_params":{},
    "lazy": False,
    "params_str": 'n_estimators=10, lambda_value=6, split_criterion=info_gain, split_confidence=0.01, tie_threshold=0.05, nb_threshold=0 ,random_state=None'
}
]



if __name__ == "__main__":
    # model, model_name = OLModel.get_model(MODELS[2],regression=True)
    # print (model, model_name)
    pass