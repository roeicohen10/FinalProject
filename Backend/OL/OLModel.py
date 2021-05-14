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
    "lazy": False
},
    {
    "name": "K-Nearest Neighbors 3",
    "func": KNNClassifier,
    "params": {"n_neighbors":3, "max_window_size":100},
    "fit_params":{},
    "lazy": True
},
    {
    "name": "K-Nearest Neighbors 5",
    "func": KNNClassifier,
    "params": {"n_neighbors":5, "max_window_size":100},
    "fit_params":{},
    "lazy": True
},
    {
    "name": "Naive Bayes",
    "func": NaiveBayes,
    "params": {},
    "fit_params":{},
    "lazy": True
},
    {
    "name": "Random Forest",
    "func": AdaptiveRandomForestClassifier,
    "params": {},
    "fit_params":{},
    "lazy": False
}
]



if __name__ == "__main__":
    # model, model_name = OLModel.get_model(MODELS[2],regression=True)
    # print (model, model_name)
    pass