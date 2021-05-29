from skmultiflow.lazy import KNNClassifier
from skmultiflow.neural_networks import PerceptronMask
from skmultiflow.bayes import NaiveBayes
from skmultiflow.meta import AdaptiveRandomForestClassifier
from .ol_ac import OnlineLearningAC


class NeuralNetwrok(OnlineLearningAC):
    """
    Neural Netwrok connection class
    """

    DEFAULT_PARAMS = {"alpha": 0.0001,
                      "max_iter": 1000,
                      "random_state": 0}
    DEFAULT_FIT_PARAMS = {"classes": [0, 1]}

    def __init__(self):
        super().__init__(name="Neural Netwrok", model=PerceptronMask, parameters=NeuralNetwrok.DEFAULT_PARAMS,
                         fit_parameters=NeuralNetwrok.DEFAULT_FIT_PARAMS, lazy=False)


class KNN(OnlineLearningAC):
    """
    K-Nearest Neighbors connection class
    """

    DEFAULT_PARAMS = {"n_neighbors": 3,
                      "max_window_size": 100,
                      "leaf_size": 30,
                      "metric": "euclidean"}
    DEFAULT_FIT_PARAMS = {}

    def __init__(self):
        super().__init__(name="K-Nearest Neighbors", model=KNNClassifier, parameters=KNN.DEFAULT_PARAMS,
                         fit_parameters=KNN.DEFAULT_FIT_PARAMS, lazy=True)


class NB(OnlineLearningAC):
    """
    Naive Bayes connection class
    """

    DEFAULT_PARAMS = {}
    DEFAULT_FIT_PARAMS = {}

    def __init__(self):
        super().__init__(name="Naive Bayes", model=NaiveBayes, parameters=NB.DEFAULT_PARAMS,
                         fit_parameters=NB.DEFAULT_FIT_PARAMS, lazy=True)


class RandomForest(OnlineLearningAC):
    """
    Random Forest connection class
    """

    DEFAULT_PARAMS = {"n_estimators": 10,
                      "lambda_value": 6,
                      "performance_metric": "acc",
                      "split_criterion": "info_gain",
                      "split_confidence": 0.01,
                      "tie_threshold": 0.05}
    DEFAULT_FIT_PARAMS = {}

    def __init__(self):
        super().__init__(name="Random Forest", model=AdaptiveRandomForestClassifier,
                         parameters=RandomForest.DEFAULT_PARAMS,
                         fit_parameters=RandomForest.DEFAULT_FIT_PARAMS, lazy=False)
