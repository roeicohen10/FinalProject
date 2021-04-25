from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.neural_network import MLPClassifier,MLPRegressor

MODELS = ["nn","knn","svm","naive_bayes"]

class OLModel:
    def __init__(self):
        pass

    @staticmethod
    def get_knn_model(regression=False, multi_class=False,k=5, **kwargs):
        if regression:
            return KNeighborsRegressor(n_neighbors=k)
        else:
            return KNeighborsClassifier(n_neighbors=k)

    @staticmethod
    def get_nn_model(regression=False, multi_class=False,**kwargs):
        if regression:
            return MLPRegressor(alpha=1e-05, hidden_layer_sizes=(15,), random_state=1,solver='lbfgs')
        else:
            return MLPClassifier(alpha=1e-05, hidden_layer_sizes=(15,), random_state=1,solver='lbfgs')

    @staticmethod
    def get_svm_model(regression=False, multi_class=False, **kwargs):
        if regression:
            return svm.SVR()
        else:
            if not multi_class:
                return svm.SVC()
            else:
                return svm.LinearSVC()

    @staticmethod
    def get_naive_base_model(**kwargs):
        return GaussianNB()


    @staticmethod
    def get_model(model_name, regression=False, multi_class=False,**kwargs):
        if model_name == "nn":
            return OLModel.get_nn_model(regression=regression,multi_class=multi_class),"Neural Network"
        elif model_name == "knn":
            return OLModel.get_knn_model(regression=regression,multi_class=multi_class),"K-Nearest Neighbors"
        elif model_name == "svm":

            return OLModel.get_svm_model(regression=regression,multi_class=multi_class),"Support Vector Machines"
        elif model_name == "naive_base":
            return OLModel.get_naive_base_model(),"Naive Bayes"

        raise Exception("Unknown Model Name")




if __name__ == "__main__":
    model, model_name = OLModel.get_model(MODELS[2],regression=True)
    print (model, model_name)
