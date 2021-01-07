import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from Backend.Streaming.Streaming import *
from mpl_toolkits.mplot3d import Axes3D

class Analyze:
    def __init__(self, stats,dataset_name,learning_algorithm_name):
        self.stats= stats
        self.dataset_name = dataset_name
        self.learning_algorithm_name = learning_algorithm_name
        self.stats_list = []
        self.stats_list.append(self.stats)


    def show_fs_time_measures_plot(self):
        batch_len = len(self.stats['fs_time_measures'])
        batch_len_list = list(range(0,batch_len))
        df = pd.DataFrame([batch_len_list,self.stats['fs_time_measures']])
        df = df.transpose()
        sns.scatterplot(data=df, x=df[0], y=df[1])
        plt.title("Time taken for Feature Selection for each batch\n" + "Dataset name: " + self.dataset_name + "\n" + "Learning Algorithm name: " + self.learning_algorithm_name)
        plt.ylabel("Feature Selection Time")
        plt.xlabel("Batch")
        plt.show()

    def show_accuracy_measures_plot(self):
        batch_len = len(self.stats['acc_measures'])
        batch_len_list = list(range(0, batch_len))
        df = pd.DataFrame([batch_len_list,self.stats['acc_measures']])
        df = df.transpose()
        sns.scatterplot(data=df, x=df[0], y=df[1])
        plt.title("Accuracy gained for each batch\n" + "Dataset name: " + self.dataset_name + "\n" + "Learning Algorithm name: " + self.learning_algorithm_name)
        plt.ylabel("Accuracy")
        plt.xlabel("Batch")
        plt.show()

    def show_memory_measures_plot_for_batch(self):
        batch_len = len(self.stats['memory_measures'])
        batch_len_list = list(range(0, batch_len))
        df = pd.DataFrame([batch_len_list,self.stats['memory_measures']])
        df = df.transpose()
        sns.scatterplot(data=df, x=df[0], y=df[1])
        plt.title("Memory used for each batch\n" + "Dataset name: " + self.dataset_name + "\n" + "Learning Algorithm name: " + self.learning_algorithm_name)
        plt.ylabel("Memory Used")
        plt.xlabel("Batch")
        plt.show()


    def show_number_of_features_for_batch(self):
        num_of_features_list = [len(x) for x in self.stats["features"]]
        batch_len = len(num_of_features_list)
        batch_len_list = list(range(0, batch_len))
        df = pd.DataFrame([batch_len_list,num_of_features_list])
        df = df.transpose()
        sns.scatterplot(data=df,x=df[0],y=df[1])
        plt.title("Number of features for each batch\n"+"Dataset name: "+self.dataset_name+"\n"+"Learning Algorithm name: "+self.learning_algorithm_name)
        plt.ylabel("Number of Features")
        plt.xlabel("Batch")
        plt.show()

    def show_accuracy_for_number_of_features(self):
        num_of_features_list = [len(x) for x in self.stats["features"]]
        df = pd.DataFrame([num_of_features_list,self.stats['acc_measures']])
        df=df.transpose()
        sns.scatterplot(data=df,x=df[0],y=df[1])
        plt.title("Accuracy and Number of features\n"+"Dataset name: "+self.dataset_name+"\n"+"Learning Algorithm name: "+self.learning_algorithm_name)
        plt.xlabel("Number of Features")
        plt.ylabel("Accuracy")
        plt.show()

    def show_fs_time_measures_plot(self):
        batch_len = len(self.stats['proc_time_measures'])
        batch_len_list = list(range(0, batch_len))
        df = pd.DataFrame([batch_len_list,self.stats['proc_time_measures']])
        df = df.transpose()
        sns.scatterplot(data=df, x=df[0], y=df[1])
        plt.title("Time taken for the process for each batch\n" + "Dataset name: " + self.dataset_name + "\n" + "Learning Algorithm name: " + self.learning_algorithm_name)
        plt.ylabel("Process Time")
        plt.xlabel("Batch")
        plt.show()

    def show_3d_proc_time_accuracy_batch(self):
        batch_len = len(self.stats['acc_measures'])
        batch_len_list = list(range(0, batch_len))
        df = pd.DataFrame([batch_len_list,self.stats['acc_measures'],self.stats['proc_time_measures']])
        df = df.transpose()
        sns.set(style="darkgrid")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x=df[0]
        y=df[1]
        z=df[2]
        plt.title("Accuracy gained given the process time and batch number\n" + "Dataset name: " + self.dataset_name + "\n" + "Learning Algorithm name: " + self.learning_algorithm_name)
        ax.set_xlabel("Batch")
        ax.set_ylabel("Accuracy")
        ax.set_zlabel("Process Time")
        ax.scatter(x,y,z)
        plt.show()

    def set_stats(self,stats):
        self.stats=stats
        self.stats_list.append(self.stats)

    def set_dataset_name(self,dataset_name):
        self.dataset_name=dataset_name

    def set_learning_algorithm_name(self,learning_algorithm_name):
        self.learning_algorithm_name=learning_algorithm_name







