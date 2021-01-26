import tkinter as tk
from tkinter import *

class OL_Param(tk.Frame):
    def __init__(self,frame):
        super().__init__(frame)
        self.params= {}
        self.title=tk.Label(self,text="OL Param:")
        self.title.grid(row=0,column=0)

    def get_params(self):
        return self.params

    def validation(self):
        pass

class SVM_Param(OL_Param):
    def __init__(self,frame):
        super().__init__(frame)
        self.params.clear()
        self.params["regression"]=False
        self.params["multi_class"]=False
        self.regression_lable=tk.Label(self,text="regression:")
        self.regression_var = StringVar(self)
        self.regression_var.set("False")
        self.regression_menu = OptionMenu(self, self.regression_var, *["True","False"],command=self.set_regression)
        self.regression_lable.grid(row=1,column=0)
        self.regression_menu.grid(row=1,column=1)

        self.multi_class_lable=tk.Label(self,text="multi class:")
        self.multi_class_var = StringVar(self)
        self.multi_class_var.set("False")
        self.multi_class_menu = OptionMenu(self, self.multi_class_var, *["True","False"],command=self.set_multi_class)
        self.multi_class_lable.grid(row=1,column=2)
        self.multi_class_menu.grid(row=1,column=3)

    def set_regression(self,event):
        if event=="True":
            self.params["regression"]=True
        else:
            self.params["regression"]=False

    def set_multi_class(self,event):
        if event=="True":
            self.params["multi_class"]=True
        else:
            self.params["multi_class"]=False

    def validation(self):
        return True

class KNN_Param(OL_Param):
    def __init__(self,frame):
        super().__init__(frame)
        self.params.clear()
        self.params["regression"]=False
        self.params["k"]=1
        self.regression_lable=tk.Label(self,text="regression:")
        self.regression_var = StringVar(self)
        self.regression_var.set("False")
        self.regression_menu = OptionMenu(self, self.regression_var, *["True","False"],command=self.set_regression)
        self.regression_lable.grid(row=1,column=0)
        self.regression_menu.grid(row=1,column=1)

        self.K_lable=tk.Label(self,text="K:")
        self.K_var = StringVar(self)
        self.K_var.set(1)
        self.K_menu = OptionMenu(self, self.K_var, *range(1,10),command=self.set_k)
        self.K_lable.grid(row=1,column=2)
        self.K_menu.grid(row=1,column=3)

    def set_regression(self,event):
        if event=="True":
            self.params["regression"]=True
        else:
            self.params["regression"]=False

    def set_k(self,event):
        self.params["k"]=event

    def validation(self):
        return True

class NN_Param(OL_Param):
    def __init__(self,frame):
        super().__init__(frame)
        self.params.clear()
        self.params["regression"]=False
        self.regression_lable=tk.Label(self,text="regression:")
        self.regression_var = StringVar(self)
        self.regression_var.set("False")
        self.regression_menu = OptionMenu(self, self.regression_var, *["True","False"],command=self.set_regression)
        self.regression_lable.grid(row=1,column=0)
        self.regression_menu.grid(row=1,column=1)

    def set_regression(self,event):
        if event=="True":
            self.params["regression"]=True
        else:
            self.params["regression"]=False

    def validation(self):
        return True

class NB_Param(OL_Param):
    def __init__(self,frame):
        super().__init__(frame)
        self.params.clear()
        self.lable=tk.Label(self,text="No params")
        self.lable.grid(row=1,column=0)

    def validation(self):
        return True