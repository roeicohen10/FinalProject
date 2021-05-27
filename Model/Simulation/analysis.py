import os, copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Analysis:
    '''
    class to analysis experiment results - mainly creates graphs
    '''

    @classmethod
    def single_experiment_facade(cls,experiment):
        '''
        helper method to unite all params of experiments to be save
        :param experiment: instance of experiment
        '''
        cls.single_experiment_graphs_facade(experiment)
        cls.single_experiment_csvs_facade(experiment)

    @classmethod
    def single_experiment_graphs_facade(cls,experiment):
        '''
        helper method to unite all graphs of experiments to be save
        :param experiment: instance of experiment
        '''
        methods_to_run = [cls.accuracy_timestamp_graph, cls.memory_timestamp_graph, cls.feature_timestamp_graph,cls.selected_features_true_false_graph]
        for index, method in enumerate(methods_to_run):
            if index > 1 and not experiment.ofs:
                continue
            try:
                method(experiment)
            except Exception as e:
                print(e)



    @classmethod
    def single_experiment_csvs_facade(cls,experiment):
        '''
        helper method to unite all csv data of experiments to be save
        :param experiment: instance of experiment
        '''
        methods_to_run = [cls.create_experiment_params_csv, cls.create_experiment_runtime_csv, cls.create_selected_features_csv]
        for index, method in enumerate(methods_to_run):
            if index > 1 and not experiment.ofs:
                continue
            try:
                method(experiment)
            except Exception as e:
                print(e)

    @classmethod
    def accuracy_timestamp_graph(cls,experiment):
        '''
        method to create accuracy timestamp graph
        :param experiment: instance of experiment
        '''
        image_path = os.path.join(experiment.export_path, f"acc_{experiment.base_file_name}.png")
        plt.clf()
        try:
            plt.plot(experiment.stream_records_indexes,experiment.prequential_accuracy)
        except ValueError as e:
            x_data = [num for num in range(len(experiment.prequential_accuracy))]
            plt.plot(x_data, experiment.prequential_accuracy)
        plt.title('Prequential Accuracy for each streamed record')
        plt.xlabel('TimeStamp')
        plt.ylabel('Accuracy')

        plt.savefig(image_path)

    @classmethod
    def feature_timestamp_graph(cls,experiment):
        '''
        method to create num of selected features, timestamp graph
        :param experiment: instance of experiment
        '''
        image_path = os.path.join(experiment.export_path, f"acc_features_{experiment.base_file_name}.png")
        plt.clf()
        plt.plot(experiment.stream_records_indexes, [len(sf) for sf in experiment.selected_features])
        plt.title('Number of selected features per timestamp')
        plt.xlabel('TimeStamp')
        plt.ylabel('Features')

        plt.savefig(image_path)

    @classmethod
    def memory_timestamp_graph(cls,experiment):
        '''
        method to create memory usage timestamp graph
        :param experiment: instance of experiment
        '''
        image_path = os.path.join(experiment.export_path, f"mem_{experiment.base_file_name}.png")
        plt.clf()
        try:
            plt.plot(experiment.stream_records_indexes, experiment.memory_usage)
        except ValueError as e:
            x_data = [num for num in range(len(experiment.memory_usage))]
            plt.plot(x_data, experiment.memory_usage)
        plt.title('Memory Usage')
        plt.xlabel('TimeStamp')
        plt.ylabel('Usage (Bytes)')

        plt.savefig(image_path)

    @classmethod
    def selected_features_true_false_graph(cls,experiment):
        '''
        method to create graph of selected feature in each run of ofs (indicates as +,-)
        :param experiment: instance of experiment
        '''
        image_path = os.path.join(experiment.export_path, f"true_false_features_{experiment.base_file_name}.png")
        plt.clf()
        y_val = [col_index for col_index in range(experiment.X.shape[1])]
        x_val = experiment.concept_drift_stream_indexes
        true_false_features = [[feature_index in sf for feature_index in y_val] for sf in experiment.concept_drift_selected_features]
        plt.yticks(y_val)
        plt.xticks(x_val)
        counter = 0
        for xe, ye in zip(x_val, true_false_features):
            true = [index for index in range(len(ye)) if ye[index]]
            false = list(set(y_val).difference(set(true)))

            plt.scatter([xe] * len(true), true, marker="+", c="black")

            plt.scatter([xe] * len(false), false, marker="_", c="black")
            counter += 1

        plt.title('Selected and Unselected features per timestamp')
        plt.xlabel('TimeStamp')
        plt.ylabel('Features')
        plt.savefig(image_path)

    @classmethod
    def create_experiment_params_csv(cls,experiment):
        '''
        method to save all general results of the experiment in csv file
        :param experiment: instance of experiment
        '''
        # create csv of results
        headers = ['Last accuracy', 'Mean OL algorithm runtime',
                   'Mean OFS algorithm runtime',
                   'Number of times OFS run', 'Selected Features']
        data = [[experiment.prequential_accuracy[-1], np.mean(experiment.ol_runtime),
                np.mean(experiment.ofs_runtime), len(experiment.concept_drift_selected_features),
                experiment.concept_drift_selected_features]]
        pd.DataFrame(data, columns=headers).to_csv(
            os.path.join(experiment.export_path, 'params.csv'), index=False)

    @classmethod
    def create_experiment_runtime_csv(cls,experiment):
        '''
        method to save runtime results (ol and ofs) of the experiment in csv file
        :param experiment: instance of experiment
        '''
        # create csv of results
        if not experiment.ofs:
            experiment.ofs_runtime = [0]*len(experiment.ol_runtime)
        pd.DataFrame({'OL Runtime':experiment.ol_runtime,
                      'OFS Runtime':experiment.ofs_runtime}).to_csv(
            os.path.join(experiment.export_path, 'runtimes.csv'), index=False)



    @classmethod
    def create_selected_features_csv(cls,experiment):
        '''
        method to save selected features of the experiment in csv file
        :param experiment: instance of experiment
        '''
        # create csv of results
        len_selected_features = [len(sf) for sf in experiment.concept_drift_selected_features]
        pd.DataFrame({'Selected Features':experiment.concept_drift_selected_features,
                      'Selected Features Len':len_selected_features}).to_csv(
            os.path.join(experiment.export_path, 'selected_features.csv'), index=False)


    @classmethod
    def multiple_experiment_facade(cls,experiments, export_path=None):
        '''
        method to run all multiple experiments results (insert acc to dict by ofs,ol,window size)
        :param experiments: list of experiments
        :param export_path: path to export data
        '''
        if not export_path:
            export_path = os.path.dirname(os.path.dirname(os.path.dirname(experiments[0].export_path))) # ds_path
        ol_names = set()
        accuracies = {}
        for experiment in experiments:
            ofs_name = experiment.ofs.name if experiment.ofs else '-'
            ol_names.add(experiment.ol.name)
            if not accuracies.get(ofs_name, None):
                accuracies[ofs_name] = {}
            if not accuracies[ofs_name].get(str(experiment.window_size),None):
                accuracies[ofs_name][str(experiment.window_size)] = {}
            accuracies[ofs_name][str(experiment.window_size)][experiment.ol.name] = experiment.prequential_accuracy[-1]
        cls.ofs_ol_accuracy_comparison_graph(accuracies, ol_names,export_path)


    @classmethod
    def ofs_ol_accuracy_comparison_graph(cls, data, ol_names,export_path):
        '''
        method which creates comparison acc between ol by windows size graphs for each ofs
        :param data: accurecies of each experiment
        :param ol_names: all online learning models names which used
        :param export_path: path to export data
        '''
        markers = ["x", "P", "*", "<", "."]
        colors = ["red", "green", "blue", "yellow", "pink", "black", "orange", "purple", "beige", "brown", "gray",
                  "cyan",
                  "magenta"]
        ol_scatter_markers = {ol_name: markers[index] for index, ol_name in enumerate(ol_names)}
        for ofs_name, ofs_data in data.items():
            plt.clf()
            f, ax = plt.subplots()
            color_index = 0
            plots = []
            for window_size, window_size_data in ofs_data.items():
                for ol_name,acc in window_size_data.items():
                    handle = ax.scatter(x=window_size, y=acc,
                                        marker=ol_scatter_markers[ol_name], color=colors[color_index],
                                        alpha=0.45)
                    plots.append(copy.copy(handle))
                color_index += 1

            ax.set_title("Accuracy per Window Size")
            ax.set_xlabel("Window Size")
            ax.set_ylabel("Accuracy")

            for h in plots:
                h.set_color("black")

            legends = ax.legend(plots,
                                list(ol_names),
                                loc='upper left',
                                bbox_to_anchor=(1, 0.5),
                                fontsize=8,
                                labelcolor="black")

            path = f"{os.path.join(export_path, ofs_name)}.png"
            plt.savefig(path, bbox_extra_artists=(legends,), bbox_inches='tight')


    @classmethod
    def create_accuracy_survey_report(cls, experiments, export_path=None):
        '''
        method for combine all accuracy result of multiple experiments on multiple datasets
        :param experiments: list of experiments
        :param export_path: path to export data
        '''
        if not export_path:
            export_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(experiments[0].export_path)))) # main dir path
        accuracies = {}
        for experiment in experiments:
            ofs_name = experiment.ofs.name if experiment.ofs else '-'
            if experiment.ol.name not in accuracies:
                accuracies[experiment.ol.name] = {}
            if not accuracies[experiment.ol.name].get(str(experiment.window_size), None):
                accuracies[experiment.ol.name][str(experiment.window_size)] = {}
            if not accuracies[experiment.ol.name][str(experiment.window_size)].get(ofs_name, None):
                accuracies[experiment.ol.name][str(experiment.window_size)][ofs_name] = {}

            accuracies[experiment.ol.name][str(experiment.window_size)][ofs_name][experiment.ds_name] = experiment.prequential_accuracy[-1]
        cls.create_acc_csvs(accuracies, export_path)

    @classmethod
    def create_acc_csvs(cls,accuracies, path):
        '''
        creates the accuracy csv files
        :param accuracies: dictionary of accuracies(ol->ofs->wind_size->ds)
        :param path: path to export data
        '''
        for ol_name, ol_data in accuracies.items():
            for wind_size, wind_data in ol_data.items():
                fname = f'{ol_name}_{wind_size}.csv'
                pd.DataFrame(wind_data).to_csv(os.path.join(path, fname))
