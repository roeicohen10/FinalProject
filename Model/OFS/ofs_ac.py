class OnlineFeatureSelectionAC:
    """
    Abstract class to ofs algorithms
    """
    DEFAULT_PARAMS = {}

    def __init__(self, name='', parameters={}):
        self.name = name
        self.parameters = parameters

    def run(self, X, Y):
        """
        method to run the ofs algorithm
        :param X: np.array of features data
        :param Y: np.array of target data
        :param param: hyper parameters to run the ofs
        :return: np.array of selected features
        """

        pass

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
        return ','.join([f'{name}={value}' for name, value in self.parameters.items()])

    def __str__(self):
        """
        :return:  string representation of the algorithm
        """
        return f'{self.name}({self.parameters_representation})'

    @classmethod
    def get_algorithm_default_parameters(cls):
        """
        :return: list of algorithm hyper-parameters names
        """
        return cls.DEFAULT_PARAMS

    @classmethod
    def get_all_ofs_algo(cls):
        """
        function which returns all the and instance of implemented ofs algorithms
        :return: list of ofs algorithms
        """
        import importlib, inspect, os, Model.OFS
        x = OnlineFeatureSelectionAC()
        current_folder_path = os.path.dirname(os.path.abspath(__file__))
        ofs_package_files = os.listdir(current_folder_path)
        files_to_check = []
        for file in ofs_package_files:
            file_name = os.path.basename(file).split(".")
            if len(file_name) != 2 or file_name[1] != "py" or file_name[0] in ['__init__', 'ofs_ac']:
                continue
            files_to_check.append(file_name[0])

        algorithms, files_not_to_check = [], []
        for file in files_to_check:
            for name, clss in inspect.getmembers(importlib.import_module(f'Model.OFS.{file}'), inspect.isclass):
                if len(clss.__bases__) == 0 or cls.__name__ != clss.__bases__[0].__name__:
                    continue
                algorithms.append(clss)
        return algorithms

if __name__ == '__main__':
    print(OnlineFeatureSelectionAC.get_all_ofs_algo())