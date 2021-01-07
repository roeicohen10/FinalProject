from .pystreaming import pystreamfs
import numpy as np
import pandas as pd
from Backend.OL import OLModel
from Backend.OFS import OFSAlgo


class Stream_Data():
    def __init__(self):
        self.data = None
        self.ol = None
        self.ol_name = None
        self.ofs = None
        self.ofs_name = None
        self.feature_name = None
        self.X = None
        self.Y = None
        self.stats=None
        self.params = dict()
        # self.params['num_features']=0
        self.params['batch_size']=0


    def set_data(self, path,target_name):
        data = pd.read_csv(path)
        self.feature_name = np.array(data.drop(target_name, 1).columns)
        self.data = np.array(data)

    def set_X_Y(self,X,Y):
        self.X, self.Y = X,Y

    def prepare_data(self, target_index, shuffle=False):
        self.X, self.Y = pystreamfs.prepare_data(self.data, target_index, shuffle)


    # def set_params(self, params):
    #     self.params = params
    #
    def set_num_feature(self,num):
        self.params['num_features']=num

    def set_batch_size(self,num):
        self.params['batch_size'] = num

    def set_ofs(self):
        self.ofs,self.ofs_name = OFSAlgo.get_algo()


    # def set_ofs(self,algo):
    #     self.ofs=algo
    #     self.ofs_name = "invasing"

    def set_ol(self, model_name, regression=False, multi_class=False,**kwargs):
        self.ol,self.ol_name = OLModel.get_model(model_name, regression, multi_class)

    def simulate_stream(self,inc_num):
        self.stats= pystreamfs.simulate_stream(self.X, self.Y, self.ofs, self.ol, self.params, inc_num=inc_num)

    def get_plot_stats(self):
        pystreamfs.plot_stats(self.stats, self.feature_name, self.params, self.ofs_name, self.ol_name).show()