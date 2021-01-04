from pystreamfs import pystreamfs
import numpy as np
import pandas as pd
from OFS import *
from OL import OLModel


class Stream_Data():
    def __init__(self):
        self.data = None
        self.ol = None
        self.ol_name = None
        self.ofs = None
        self.feature_name = None
        self.X = None
        self.Y = None
        self.stats=None
        self.params = dict()

    def set_data(self, path):
        data = pd.read_csv(path)
        self.feature_name = np.array(data.drop('target', 1).columns)
        self.data = np.array(data)

    def prepare_data(self, target_index, shuffle):
        self.X, self.Y = pystreamfs.prepare_data(self.data, target_index, shuffle)

    def set_params(self, params):
        self.params = params

    def set_ofs(self, algorithm):
        self.ofs = algorithm

    def set_ol(self, model_name, regression=False, multi_class=False,**kwargs):
        self.ol,self.ol_name = OLModel.get_model(model_name, regression, multi_class,**kwargs)

    def simulate_stream(self):
        self.stats=pystreamfs.simulate_stream(self.X, self.Y, self.ofs, self.ol, self.params)

    def get_plot_stats(self):
        return pystreamfs.plot_stats(self.stats, self.feature_names, self.params, 'Online feature selection (OFS)',self.ol_name)