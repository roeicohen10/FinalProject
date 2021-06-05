from Model.Simulation.experiment import Experiment
from Model.OFS.ofs_ac import OnlineFeatureSelectionAC
from Model.OL.ol_ac import OnlineLearningAC
from Model.Simulation.parse import Parse




class Controller:
    OFS_CONTROLLER = {
        'Alpha Investing': 'AlphaInvesting',
        'SAOLA': 'Saola',
        'OSFS': 'OSFS',
        'F-OSFS': 'FastOSFS',
        'Fires': 'Fires',
        'Without OFS': None
    }

    OL_CONTROLLER = {
        'K-NN': 'KNN',
        'Perceptron Mask (ANN)': 'NeuralNetwork',
        'Random Forest': 'RandomForest',
        'Naive-Bayes': 'NB'
    }


    @classmethod
    def get_relevant_ofs_algorithms(cls, chosen_ofs):
        ofs_instances = []
        ofs_algos = OnlineFeatureSelectionAC.get_all_ofs_algo()
        for ofs_name, ofs_params in chosen_ofs.items():
            if not Controller.OFS_CONTROLLER.get(ofs_name):
                ofs_instances.append(None)  # case without ofs
            else:
                ofs_instance = ofs_algos.get(Controller.OFS_CONTROLLER.get(ofs_name))()
                ofs_instance.set_algorithm_parameters(**ofs_params)
                ofs_instances.append(ofs_instance)
        return ofs_instances

    @classmethod
    def get_relevant_ol_models(cls, chosen_ol):
        ol_instances = []
        ol_models = OnlineLearningAC.get_all_ol_algo()
        for ol_name, ol_params in chosen_ol.items():
            ol_instance = ol_models.get(Controller.OL_CONTROLLER.get(ol_name))()
            ol_instance.set_algorithm_parameters(**ol_params)
            ol_instances.append(ol_instance)
        return ol_instances

    @classmethod
    def run_multi_experiments(cls, file_path, file_name, export_path, ofs_algos, ol_models, window_instance,file_target_index=-1,
                              window_sizes=[300, 500]):
        ol_models = cls.get_relevant_ol_models(ol_models)
        ofs_algos = cls.get_relevant_ofs_algorithms(ofs_algos)
        X, y, classes = Parse.read_ds(file_path, target_index=file_target_index)
        ds_exps = []
        for window_size in window_sizes:
            for ofs_instance in ofs_algos:
                for ol_instance in ol_models:
                    ol_instance.set_algorithm_fit_parameters(classes=classes)
                    experiment = Experiment(ofs=ofs_instance, ol=ol_instance, window_size=window_size, X=X, y=y,
                                            ds_name=file_name, transform_binary=False, special_name='multi')
                    ds_exps.append(experiment)
                    experiment.run()
                    experiment.save(path=export_path)
                    window_instance.increase_pb()
        Experiment.save_graphs(ds_exps)

