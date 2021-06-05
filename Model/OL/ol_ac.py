class OnlineLearningAC:
    """
    Abstract class to ol algorithms
    """
    DEFAULT_PARAMS = {}
    def __init__(self, name='', model=None, parameters={}, fit_parameters={}, lazy=False):
        self.name = name
        self.parameters = parameters
        self.fit_parameters = fit_parameters
        self.lazy = lazy
        self.model = model
        self.created_model = None

    def create(self):
        """
        creates a new OL model instance
        """

        self.created_model = self.model(**self.parameters)

    def get_algorithm_parameters(self):
        """
        :return: list of algorithm hyper-parameters names
        """
        return list(self.parameters.keys())

    def set_algorithm_parameters(self, **kwargs):
        """
        set the algorithm hyper-parameters
        :param kwargs: new hyper-parameters keys-values
        """
        for param_name in self.parameters.keys():
            self.parameters[param_name] = kwargs.get(param_name, self.parameters.get(param_name))

    def parameters_representation(self):
        """
        :return: string representation of the algorithm parameters
        """
        return ','.join([f'{name}={value}' for name, value in self.fit_parameters.items()])

    def get_algorithm_fit_parameters(self):
        """
        :return: list of algorithm fit parameters names
        """
        return list(self.parameters.keys())

    def set_algorithm_fit_parameters(self, **kwargs):
        """
        set the algorithm fit parameters
        :param kwargs: new fit parameters keys-values
        """
        for param_name in self.fit_parameters.keys():
            self.fit_parameters[param_name] = kwargs.get(param_name, self.fit_parameters.get(param_name))

    def __str__(self):
        """
        :return:  string representation of the algorithm
        """
        return f'{self.name}({self.parameters_representation})'

    @classmethod
    def get_model_default_parameters(cls):
        """
        :return: list of algorithm hyper-parameters names
        """
        return cls.DEFAULT_PARAMS

    @classmethod
    def get_all_ol_algo(cls):
        """
        function which returns all the and instance of ol algorithms
        :return: list of ol algorithms
        """
        import importlib, inspect
        models = {}
        for name, clss in inspect.getmembers(importlib.import_module('Model.OL.models'), inspect.isclass):
            if len(clss.__bases__) == 0 or cls.__name__ != clss.__bases__[0].__name__:
                continue
            models[clss.__name__] = clss
        return models


if __name__ == '__main__':
    sub_classes = OnlineLearningAC.get_all_ol_algo()
    print(sub_classes)