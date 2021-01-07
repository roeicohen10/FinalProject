import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from Streaming.Streaming import *

class Analyze:
    def __init__(self, stats):
        self.start_time = None
        self.end_time = None
        self.stats= stats


    def set_start_time(self,start_time):
        self.start_time=start_time

    def set_end_time(self,end_time):
        self.end_time=end_time

    def show_time_measures_plot(self,dataset_name,learning_algorithm_name):
        batch_len = len(self.stats['time_measures'])
        df = pd.DataFrame([batch_len,self.stats['time_measures']])
        df = df.transpose()
        sns.scatterplot(data=df, x=df[0], y=df[1])
        plt.title("Time taken for each batch\n" + "Dataset name: " + dataset_name + "\n" + "Learning Algorithm name: " + learning_algorithm_name)
        plt.ylabel("Time")
        plt.xlabel("Batch")
        plt.show()

    def show_accuracy_measures_plot(self,dataset_name,learning_algorithm_name):
        batch_len = len(self.stats['acc_measures'])
        df = pd.DataFrame([batch_len,self.stats['acc_measures']])
        df = df.transpose()
        sns.scatterplot(data=df, x=df[0], y=df[1])
        plt.title("Accuracy gained for each batch\n" + "Dataset name: " + dataset_name + "\n" + "Learning Algorithm name: " + learning_algorithm_name)
        plt.ylabel("Accuracy")
        plt.xlabel("Batch")
        plt.show()

    def show_memory_measures_plot_for_batch(self,dataset_name,learning_algorithm_name):
        batch_len = len(self.stats['memory_measures'])
        df = pd.DataFrame([batch_len,self.stats['memory_measures']])
        df = df.transpose()
        sns.scatterplot(data=df, x=df[0], y=df[1])
        plt.title("Memory used for each batch\n" + "Dataset name: " + dataset_name + "\n" + "Learning Algorithm name: " + learning_algorithm_name)
        plt.ylabel("Memory Used")
        plt.xlabel("Batch")
        plt.show()

    def get_run_time(self):
        return self.end_time-self.start_time

    def show_number_of_features_for_batch(self,dataset_name,learning_algorithm_name):
        num_of_features_list = [len(x) for x in self.stats["features"]]
        batch_len = len(num_of_features_list)
        df = pd.DataFrame([batch_len,num_of_features_list])
        df = df.transpose()
        sns.scatterplot(data=df,x=df[0],y=df[1])
        plt.title("Number of features for each batch\n"+"Dataset name: "+dataset_name+"\n"+"Learning Algorithm name: "+learning_algorithm_name)
        plt.ylabel("Number of Features")
        plt.xlabel("Batch")
        plt.show()

    def show_accuracy_for_number_of_features(self,dataset_name,learning_algorithm_name):
        num_of_features_list = [len(x) for x in self.stats["features"]]
        df = pd.DataFrame([num_of_features_list,self.stats['acc_measures']])
        df=df.transpose()
        sns.scatterplot(data=df,x=df[0],y=df[1])
        plt.title("Accuracy and Number of features\n"+"Dataset name: "+dataset_name+"\n"+"Learning Algorithm name: "+learning_algorithm_name)
        plt.xlabel("Number of Features")
        plt.ylabel("Accuracy")
        plt.show()





