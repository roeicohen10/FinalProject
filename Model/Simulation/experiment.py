import pandas as pd
import numpy as np
import time, psutil, os, logging
from datetime import datetime
from skmultiflow.drift_detection import DDM
from Model.Simulation.analysis import Analysis

from Model.OL.ol_ac import OnlineLearningAC
from Model.OFS.ofs_ac import OnlineFeatureSelectionAC
from Model.Simulation.parse import Parse
# path to save results
DIR_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_EXPORT_PATH = os.path.join(DIR_PATH, 'data/Experiments')
DEFAULT_LOGS_EXPORT_PATH = os.path.join(DEFAULT_EXPORT_PATH, 'logs')


class Experiment:
    '''
    main class to simulate experiment on ds
    '''

    def __init__(self, ofs, ol, window_size, X, y, ds_name='', transform_binary=False, special_name=''):
        self.ofs = ofs
        self.ol = ol
        self.window_size = window_size
        self.X = X
        self.y = y if not transform_binary else Experiment.transform_binary(y).reshape((y.shape[0],1))
        self.ds_name = ds_name
        self.stream_records_indexes = list(range(window_size + 1, X.shape[0]))
        self.prequential_accuracy = []
        self.selected_features = []
        self.concept_drift_selected_features = []
        self.concept_drift_stream_indexes = []
        self.memory_usage = []
        self.ofs_runtime = []
        self.ol_runtime = []
        self.current_selected_features = []
        self.start_time = f'{datetime.today().strftime("%d-%m-%Y %H.%M.%S")}({special_name})'
        self.export_path = os.path.join(DEFAULT_EXPORT_PATH,self.start_time)

        if ofs:
            self.base_file_name = f'{self.ds_name}_{str(self.window_size)}_{self.ofs.name}_{self.ol.name}'
        else:
            self.base_file_name = f'{self.ds_name}_{str(self.window_size)}_-_{self.ol.name}'
        Experiment.init_loggings(name=self.base_file_name)

    def init_ofs_ol(self, stream_index):
        '''
        method to create new instances of online learning and online feature selection algorithms
        '''
        X_train = self.X[stream_index - self.window_size :stream_index, :]
        y_train = np.ravel(self.y[stream_index - self.window_size :stream_index])
        selected_features = list(range(X_train.shape[1]))
        self.ol.create()  # create new ol model
        if self.ofs:  # run ofs to get selected features
            start_t = time.perf_counter()
            selected_features = self.ofs.run(X_train, y_train)
            self.ofs_runtime.append(time.perf_counter() - start_t)
            if len(selected_features) == 0:
                logging.error(f"Could not find features for window size {self.window_size}")
                raise Exception("Could not find features")
            self.selected_features.append(selected_features)
            self.concept_drift_selected_features.append(selected_features)
            self.current_selected_features = selected_features
            self.concept_drift_stream_indexes.append(stream_index)

        # fit ol model
        start_t = time.perf_counter()
        self.ol.created_model = self.ol.created_model.fit(X_train[:, selected_features], y_train,
                                                          **self.ol.fit_parameters)
        self.ol_runtime.append(time.perf_counter() - start_t)

    def concept_drift_detection(self, start_window_size, stream_index):
        '''
        method to handle concept drift situation - try to ini ol and ofs to update the model
        :param start_window_size: user given window size - for case where ofs can't find features
        '''
        found_features = False
        while not found_features:
            try:
                self.init_ofs_ol(stream_index)
                found_features = True
            except Exception:
                # case where ofs failed to find features - try to add more records and replay process up to initial_wind_size*4
                if self.window_size > start_window_size * 4:
                    raise Exception("OFS could not find features.")
                self.window_size += 50
                logging.info(f"Changed window size from {self.window_size - 50} to {self.window_size}")

    def fit_lazy(self, x_record, y_record):
        '''
        method to fit model for new record - in lazy online learning algorithms
        :param x_record: features data
        :param y_record: target label
        '''
        self.ol.created_model.partial_fit(x_record[:, self.current_selected_features],
                                          y_record) if self.ofs is not None else self.ol.created_model.partial_fit(
            x_record, y_record)

    def run(self):
        '''
        main method to simulate new experiment
        '''
        print(f"Starting Experiment:{self}")
        try:
            start_window_size = self.window_size
            num_of_correct_predictions, predictions_counter = 0, 0
            ddm = DDM()
            for record in range(self.X.shape[0]):
                x_record, y_record = np.array([self.X[record, :]]), np.ravel(np.array([self.y[record]]))
                if record < self.window_size:  # aggregate records till window size
                    continue
                elif record == self.window_size:  # first initialization
                    try:
                        self.init_ofs_ol(record)
                    except Exception as e:
                        # case where ofs failed to find features - try to add more records and replay process
                        if self.window_size > start_window_size * 4: raise Exception("OFS could not find features.")
                        self.window_size += 50
                        logging.info(f"Changed window size from {self.window_size - 50} to {self.window_size}")
                    continue

                # predict
                my_pred = self.ol.created_model.predict(
                    x_record) if self.ofs is None else self.ol.created_model.predict(
                    x_record[:, self.current_selected_features])
                predictions_counter += 1
                if y_record[0] == my_pred[0]: num_of_correct_predictions += 1

                ddm.add_element(num_of_correct_predictions / predictions_counter)  # add result to concept drift model
                self.prequential_accuracy.append(num_of_correct_predictions / predictions_counter)  # add accuracy
                self.memory_usage.append(psutil.Process(os.getpid()).memory_info().rss)  # add memory usage

                if self.ol.lazy:  # partial fit for lazy models
                    self.fit_lazy(x_record, y_record)
                if ddm.detected_change():  # check for concept drift
                    self.concept_drift_detection(start_window_size, record)
                elif record != self.X.shape[0] - 1 and self.ofs:
                    self.selected_features.append(self.selected_features[-1])
        except Experiment as e:
            logging.error(f"Error: {str(e)}")

    def save(self, path=None):
        '''
        method to save experiment results
        :param path: path to export results
        '''
        try:
            if path: self.export_path = path
            self.set_export_path()
            # create graphs
            Analysis.single_experiment_facade(self)
        except Experiment as e:
            logging.error(f"Error: {str(e)}")

    def set_export_path(self):
        '''
        method to create export path dir
        '''
        path = os.path.join(self.export_path, self.ds_name)  # ds
        path = os.path.join(path, str(self.window_size))  # window_size
        path = os.path.join(path, self.ofs.name) if self.ofs else os.path.join(path, '-') # ofs
        path = os.path.join(path, self.ol.name)  # ol
        if Experiment.create_dirs(path): self.export_path = path

    @classmethod
    def create_multiple_experiments(cls, ofs_list, ol_list, window_size_list, X, y, ds_name=None):
        '''
        helper method to create multiple experiments
        :param ofs_list: list of ofs algorithms to run
        :param ol_list: list of ol algorithms to run
        :param window_size_list: list of window size to run
        :param X: numpy array of features data
        :param y: numpy array of target data
        :param ds_name: name of dataset
        '''
        experiments = []
        for ofs in ofs_list:
            for ol in ol_list:
                for window_size in window_size_list:
                    experiments.append(Experiment(ofs=ofs, ol=ol, window_size=window_size, X=X, y=y, ds_name=ds_name))

        return experiments

    @classmethod
    def run_multiple_experiments(cls, experiments):
        '''
        helper method to run multiple experiments
        :param experiments:
        '''
        for experiment in experiments:
            experiment.run()

    @classmethod
    def save_multiple_experiments(cls, experiments, path=None):
        '''
        helper method to run multiple experiments
        :param experiments:
        '''

        for experiment in experiments:
            experiment.save(path=path)
        Analysis.multiple_experiment_facade(experiments)

    @classmethod
    def save_graphs(cls, experiments):
        Analysis.multiple_experiment_facade(experiments)

    @classmethod
    def transform_binary(cls, y):
        '''
        method to transfom target labels to binary
        :param y: numpy array of target
        :return: binary numpy array of target
        '''
        y = y.reshape((y.shape[0],1))
        data_df = pd.DataFrame.from_records(y)
        columns = data_df.columns

        max_target_value = data_df[columns[-1]].value_counts().sort_values(ascending=False).index[0]
        data_df[columns[-1]][data_df[columns[-1]] == max_target_value] = -2
        max_target_value = -2

        data_df[columns[-1]][data_df[columns[-1]] != max_target_value] = 0
        data_df[columns[-1]][data_df[columns[-1]] == max_target_value] = 1
        return np.ravel(data_df.to_numpy())

    @staticmethod
    def create_dirs(path):
        '''
        helper function to create dirs of path
        :param path: path to create dirs
        :return: boolean indicator if the creation succeeded
        '''
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            return False
        return True

    @staticmethod
    def init_loggings(name):
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        Experiment.create_dirs(DEFAULT_LOGS_EXPORT_PATH)
        log_name = f"{name}.log"
        logging.basicConfig(filename=os.path.join(DEFAULT_LOGS_EXPORT_PATH, log_name),
                            filemode='a',
                            format='%(asctime)s %(message)s',
                            datefmt='%d/%m/%Y %H:%M',
                            level=logging.INFO)

    def __str__(self):
        if self.ofs:
            return f'{self.ds_name}_{str(self.window_size)}_{self.ofs.name}_{self.ol.name}'
        else:
            return f'{self.ds_name}_{str(self.window_size)}_-_{self.ol.name}'

if __name__ == '__main__':
    pass