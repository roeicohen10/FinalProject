import seaborn as sns
import matplotlib.pyplot as plt
from Streaming.Streaming import *

class Analyze:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.streaming = Stream_Data()


    def set_start_time(self,start_time):
        self.start_time=start_time

    def set_end_time(self,end_time):
        self.end_time=end_time

    def show_time_measures_plot(self):
        plt.plot(self.streaming.stats['time_measures'])
        plt.ylabel("Time")
        plt.xlabel("Batch")
        plt.show()

    def show_accuracy_measures_plot(self):
        plt.plot(self.streaming.stats['acc_measures'])
        plt.ylabel("Accuracy")
        plt.xlabel("Batch")
        plt.show()

    def show_memory_measures_plot(self):
        plt.plot(self.streaming.stats['memory_measures'])
        plt.ylabel("Memory Used")
        plt.xlabel("Batch")
        plt.show()

    def get_run_time(self):
        return self.end_time-self.start_time



