import tkinter as tk
from tkinter import *

title_font='Helvetica 9 bold'

class Params_frame(tk.Frame):
    def __init__(self, frame):
        super().__init__(frame)
        self.params = {}
        self.set_title()
        self.set_default_params()
        self.set_fields()
        self.set_fields_position()

    def get_params(self):
        self.update_params()
        return self.params

    def validation(self):
        pass

    def set_title(self):
        pass

    def set_default_params(self):
        pass

    def set_fields(self):
        pass

    def set_fields_position(self):
        pass

    def update_params(self):
        pass


# This class represent the Random Forest params frame
class RF_Param(Params_frame):

    # This list represent all performance metrics options
    performance_metric_list=["acc","kappa"]

    # This list represent all split criterion options
    split_criterion_list=["info_gain","gini"]

    # This method is the contractor of the class
    def __init__(self, frame):
        super().__init__(frame)

    # This method tests validation for input params
    def validation(self):
        if self.n_estimators_input.get() == '' or not self.n_estimators_input.get().isnumeric():
            return False, "n_estimators"
        if self.lambda_value_input.get() == '' or not self.lambda_value_input.get().isnumeric():
            return False, "lambda_value"
        if self.performance_metric_sv.get() == '':
            return False, "performance_metric"
        if self.split_criterion_sv.get() == '':
            return False, "split_criterion"
        if self.split_confidence_input.get() == '' or not is_float(self.split_confidence_input.get()):
            return False, "split_confidence"
        if self.tie_threshold_input.get() == '' or not is_float(self.tie_threshold_input.get()):
            return False, "tie_threshold"
        return True, ""

    # This method set the Random Forest title
    def set_title(self):
        self.title = tk.Label(self, text="Random Forest Param:",font=title_font)
        self.title.grid(row=0, column=0,sticky = W)

    # This method set the default params
    def set_default_params(self):
        self.params["n_estimators"]=10
        self.params["lambda_value"] = 6
        self.params["performance_metric"] = 'acc'
        self.params["split_criterion"] = 'info_gain'
        self.params["split_confidence"] = 0.01
        self.params["tie_threshold"] = 0.05

    # This method set the fields for the params
    def set_fields(self):
        self.n_estimators_lable = tk.Label(self, text="n_estimators:")
        self.n_estimators_input = tk.Entry(self, textvariable=StringVar(self, self.params["n_estimators"]))

        self.lambda_value_lable = tk.Label(self, text="lambda_value:")
        self.lambda_value_input = tk.Entry(self, textvariable=StringVar(self, self.params["lambda_value"]))

        self.performance_metric_lable = tk.Label(self, text="performance_metric:")
        self.performance_metric_sv = StringVar(self, self.params["performance_metric"])
        self.performance_metric_menu = OptionMenu(self, self.performance_metric_sv, *self.performance_metric_list)

        self.split_criterion_lable = tk.Label(self, text="split_criterion:")
        self.split_criterion_sv = StringVar(self, self.params["split_criterion"])
        self.split_criterion_menu = OptionMenu(self, self.split_criterion_sv, *self.split_criterion_list)

        self.split_confidence_lable = tk.Label(self, text="split_confidence:")
        self.split_confidence_input = tk.Entry(self, textvariable=StringVar(self, self.params["split_confidence"]))

        self.tie_threshold_lable = tk.Label(self, text="tie_threshold:")
        self.tie_threshold_input = tk.Entry(self, textvariable=StringVar(self, self.params["tie_threshold"]))

    # This method set the fields position
    def set_fields_position(self):
        self.n_estimators_lable.grid(row=1, column=0, sticky=W)
        self.n_estimators_input.grid(row=1, column=1, sticky=W)

        self.lambda_value_lable.grid(row=2, column=0, sticky=W)
        self.lambda_value_input.grid(row=2, column=1, sticky=W)

        self.performance_metric_lable.grid(row=3, column=0, sticky=W)
        self.performance_metric_menu.grid(row=3, column=1, sticky=W)

        self.split_criterion_lable.grid(row=4, column=0, sticky=W)
        self.split_criterion_menu.grid(row=4, column=1, sticky=W)

        self.split_confidence_lable.grid(row=5, column=0, sticky=W)
        self.split_confidence_input.grid(row=5, column=1, sticky=W)

        self.tie_threshold_lable.grid(row=6, column=0, sticky=W)
        self.tie_threshold_input.grid(row=6, column=1, sticky=W)

    # This method updates the parameters according to the inputs
    def update_params(self):
        self.params.update({"n_estimators": int(self.n_estimators_input.get())})
        self.params.update({"lambda_value": int(self.lambda_value_input.get())})
        self.params.update({"performance_metric": self.performance_metric_sv.get()})
        self.params.update({"split_criterion": self.split_criterion_sv.get()})
        self.params.update({"split_confidence": float(self.split_confidence_input.get())})
        self.params.update({"tie_threshold": float(self.tie_threshold_input.get())})


# This class represent the KNN params frame
class KNN_Param(Params_frame):
    # This list represent all metrics options
    metric_list = ['euclidean', 'manhattan', 'chebyshev']

    # This method is the contractor of the class
    def __init__(self, frame):
        super().__init__(frame)

    # This method tests validation for input params
    def validation(self):
        if self.n_neighbors_input.get() == '' or not self.n_neighbors_input.get().isnumeric():
            return False, "n_neighbors"
        if self.leaf_size_input.get() == '' or not self.leaf_size_input.get().isnumeric():
            return False, "leaf_size"
        if self.max_window_size_input.get() == '' or not self.max_window_size_input.get().isnumeric():
            return False, "max_window_size"
        if self.metric_sv.get() == '':
            return False, "metric"
        return True, ""

    # This method set the KNN title
    def set_title(self):
        self.title = tk.Label(self, text="KNN Param:",font=title_font)
        self.title.grid(row=0, column=0,sticky = W)

    # This method set the default params
    def set_default_params(self):
        self.params["n_neighbors"] = 5
        self.params["max_window_size"] = 1000
        self.params["leaf_size"] = 30
        self.params["metric"] = self.metric_list[0]

    # This method set the fields for the params
    def set_fields(self):
        self.n_neighbors_lable = tk.Label(self, text="n_neighbors:")
        self.n_neighbors_input = tk.Entry(self, textvariable=StringVar(self, self.params["n_neighbors"]))

        self.max_window_size_lable = tk.Label(self, text="max_window_size:")
        self.max_window_size_input = tk.Entry(self, textvariable=StringVar(self, self.params["max_window_size"]))

        self.leaf_size_lable = tk.Label(self, text="leaf_size:")
        self.leaf_size_input = tk.Entry(self, textvariable=StringVar(self, self.params["leaf_size"]))

        self.metric_lable = tk.Label(self, text="metric:")
        self.metric_sv = StringVar(self, self.params["metric"])
        self.metric_menu = OptionMenu(self, self.metric_sv, *self.metric_list)

    # This method set the fields position
    def set_fields_position(self):
        self.n_neighbors_lable.grid(row=1, column=0, sticky=W)
        self.n_neighbors_input.grid(row=1, column=1, sticky=W)

        self.max_window_size_lable.grid(row=2, column=0, sticky=W)
        self.max_window_size_input.grid(row=2, column=1, sticky=W)

        self.leaf_size_lable.grid(row=3, column=0, sticky=W)
        self.leaf_size_input.grid(row=3, column=1, sticky=W)

        self.metric_lable.grid(row=4, column=0, sticky=W)
        self.metric_menu.grid(row=4, column=1, sticky=W)

    # This method updates the parameters according to the inputs
    def update_params(self):
        self.params.update({"n_neighbors": int(self.n_neighbors_input.get())})
        self.params.update({"max_window_size": int(self.max_window_size_input.get())})
        self.params.update({"leaf_size": int(self.leaf_size_input.get())})
        self.params.update({"metric": self.metric_sv.get()})


# This class represent the Perceptron Mask params frame
class NN_Param(Params_frame):

    # This method is the contractor of the class
    def __init__(self, frame):
        super().__init__(frame)

    # This method tests validation for input params
    def validation(self):
        if self.alpha_input.get() == '' or not is_float(self.alpha_input.get()):
            return False, "alpha"
        if self.max_iter_input.get() == '' or not self.max_iter_input.get().isnumeric() or int(self.max_iter_input.get())<1:
            return False, "max_iter"
        if self.random_state_input.get() == '' or not self.random_state_input.get().isnumeric():
            return False, "random_state"
        return True, ""

    # This method set the Perceptron Mask title
    def set_title(self):
        self.title = tk.Label(self, text="Perceptron Mask Param:",font=title_font)
        self.title.grid(row=0, column=0,sticky = W)

    # This method set the default params
    def set_default_params(self):
        self.params["alpha"]=0.0001
        self.params["max_iter"] = 1000
        self.params["random_state"] = 0

    # This method set the fields for the params
    def set_fields(self):
        self.alpha_lable = tk.Label(self, text="alpha:")
        self.alpha_input = tk.Entry(self, textvariable=StringVar(self, self.params["alpha"]))

        self.max_iter_lable = tk.Label(self, text="max_iter:")
        self.max_iter_input = tk.Entry(self, textvariable=StringVar(self, self.params["max_iter"]))

        self.random_state_lable = tk.Label(self, text="random_state:")
        self.random_state_input = tk.Entry(self, textvariable=StringVar(self, self.params["random_state"]))

    # This method set the fields position
    def set_fields_position(self):
        self.alpha_lable.grid(row=1, column=0, sticky=W)
        self.alpha_input.grid(row=1, column=1, sticky=W)

        self.max_iter_lable.grid(row=2, column=0, sticky=W)
        self.max_iter_input.grid(row=2, column=1, sticky=W)

        self.random_state_lable.grid(row=3, column=0, sticky=W)
        self.random_state_input.grid(row=3, column=1, sticky=W)

    # This method updates the parameters according to the inputs
    def update_params(self):
        self.params.update({"alpha": float(self.alpha_input.get())})
        self.params.update({"max_iter": int(self.max_iter_input.get())})
        self.params.update({"random_state": int(self.random_state_input.get())})


# This class represent the Naive-Bayes params frame
class NB_Param(Params_frame):
    # This method is the contractor of the class
    def __init__(self, frame):
        super().__init__(frame)

    # This method tests validation for input params
    def validation(self):
        return True,""

    # This method set the Naive-Bayes title
    def set_title(self):
        self.title = tk.Label(self, text="Naive-Bayes Param:",font=title_font)
        self.title.grid(row=0, column=0,sticky = W)

    # This method set the default params
    def set_default_params(self):
        pass

    # This method set the fields for the params
    def set_fields(self):
        self.no_params_label = tk.Label(self, text="No params")

    # This method set the fields position
    def set_fields_position(self):
        self.no_params_label.grid(row=1, column=0)

    # This method updates the parameters according to the inputs
    def update_params(self):
        return


# This class represent the Alpha Investing params frame
class AI_Param(Params_frame):
    # This method is the contractor of the class
    def __init__(self, frame):
        super().__init__(frame)

    # This method tests validation for input params
    def validation(self):
        if self.alpha_input.get() == '' or not is_float(self.alpha_input.get()):
            return False, "alpha"
        if self.dw_input.get() == '' or not is_float(self.dw_input.get()):
            return False, "dw"
        return True, ""

    # This method set the Alpha Investing title
    def set_title(self):
        self.title = tk.Label(self, text="Alpha Investing Param:",font=title_font)
        self.title.grid(row=0, column=0,sticky = W)

    # This method set the default params
    def set_default_params(self):
        self.params["alpha"] = 0.05
        self.params["dw"] = 0.05

    # This method set the fields for the params
    def set_fields(self):
        self.alpha_lable = tk.Label(self, text="alpha:")
        self.alpha_input = tk.Entry(self, textvariable=StringVar(self, self.params["alpha"]))

        self.dw_lable = tk.Label(self, text="dw:")
        self.dw_input = tk.Entry(self, textvariable=StringVar(self, self.params["dw"]))

    # This method set the fields position
    def set_fields_position(self):
        self.alpha_lable.grid(row=1, column=0, sticky=W)
        self.alpha_input.grid(row=1, column=1, sticky=W)

        self.dw_lable.grid(row=2, column=0, sticky=W)
        self.dw_input.grid(row=2, column=1, sticky=W)

    # This method updates the parameters according to the inputs
    def update_params(self):
        self.params.update({"alpha": float(self.alpha_input.get())})
        self.params.update({"dw": float(self.dw_input.get())})


# This class represent the SAOLA params frame
class SAOLA_Param(Params_frame):
    # This method is the contractor of the class
    def __init__(self, frame):
        super().__init__(frame)

    # This method tests validation for input params
    def validation(self):
        if self.alpha_input.get() == '' or not is_float(self.alpha_input.get()):
            return False, "alpha"
        return True, ""

    # This method set the SAOLA title
    def set_title(self):
        self.title = tk.Label(self, text="SAOLA Param:",font=title_font)
        self.title.grid(row=0, column=0,sticky = W)

    # This method set the default params
    def set_default_params(self):
        self.params["alpha"] = 0.05

    # This method set the fields for the params
    def set_fields(self):
        self.alpha_lable = tk.Label(self, text="alpha:")
        self.alpha_input = tk.Entry(self, textvariable=StringVar(self, self.params["alpha"]))

    # This method set the fields position
    def set_fields_position(self):
        self.alpha_lable.grid(row=1, column=0, sticky=W)
        self.alpha_input.grid(row=1, column=1, sticky=W)

    # This method updates the parameters according to the inputs
    def update_params(self):
        self.params.update({"alpha": float(self.alpha_input.get())})


# This class represent the OSFS params frame
class OSFS_Param(Params_frame):
    # This method is the contractor of the class
    def __init__(self, frame):
        super().__init__(frame)

    # This method tests validation for input params
    def validation(self):
        if self.alpha_input.get() == '' or not is_float(self.alpha_input.get()):
            return False, "alpha"
        return True, ""

    # This method set the OSFS title
    def set_title(self):
        self.title = tk.Label(self, text="OSFS Param:",font=title_font)
        self.title.grid(row=0, column=0,sticky = W)

    # This method set the default params
    def set_default_params(self):
        self.params["alpha"] = 0.05

    # This method set the fields for the params
    def set_fields(self):
        self.alpha_lable = tk.Label(self, text="alpha:")
        self.alpha_input = tk.Entry(self, textvariable=StringVar(self, self.params["alpha"]))

    # This method set the fields position
    def set_fields_position(self):
        self.alpha_lable.grid(row=1, column=0, sticky=W)
        self.alpha_input.grid(row=1, column=1, sticky=W)

    # This method updates the parameters according to the inputs
    def update_params(self):
        self.params.update({"alpha": float(self.alpha_input.get())})


# This class represent the F-OSFS params frame
class F_OSFS_Param(Params_frame):
    # This method is the contractor of the class
    def __init__(self, frame):
        super().__init__(frame)

    # This method tests validation for input params
    def validation(self):
        if self.alpha_input.get() == '' or not is_float(self.alpha_input.get()):
            return False, "alpha"
        return True, ""

    # This method set the F-OSFS title
    def set_title(self):
        self.title = tk.Label(self, text="F-OSFS Param:",font=title_font)
        self.title.grid(row=0, column=0,sticky = W)

    # This method set the default params
    def set_default_params(self):
        self.params["alpha"] = 0.05

    # This method set the fields for the params
    def set_fields(self):
        self.alpha_lable = tk.Label(self, text="alpha:")
        self.alpha_input = tk.Entry(self, textvariable=StringVar(self, self.params["alpha"]))

    # This method set the fields position
    def set_fields_position(self):
        self.alpha_lable.grid(row=1, column=0, sticky=W)
        self.alpha_input.grid(row=1, column=1, sticky=W)

    # This method updates the parameters according to the inputs
    def update_params(self):
        self.params.update({"alpha": float(self.alpha_input.get())})


# This class represent the Fires params frame
class Fires_Param(Params_frame):
    # This method is the contractor of the class
    def __init__(self, frame):
        super().__init__(frame)

    # This method tests validation for input params
    def validation(self):
        if self.mu_init_input.get() == '' or not is_float(self.mu_init_input.get()):
            return False, "mu_init"
        if self.sigma_init_input.get() == '' or not is_float(self.sigma_init_input.get()):
            return False, "sigma_init"
        if self.penalty_s_input.get() == '' or not is_float(self.penalty_s_input.get()):
            return False, "penalty_s"
        if self.penalty_r_input.get() == '' or not is_float(self.penalty_r_input.get()):
            return False, "penalty_r"
        if self.epochs_input.get() == '' or not self.epochs_input.get().isnumeric():
            return False, "epochs"
        if self.lr_mu_input.get() == '' or not is_float(self.lr_mu_input.get()):
            return False, "lr_mu"
        if self.lr_sigma_input.get() == '' or not is_float(self.lr_sigma_input.get()):
            return False, "lr_sigma"
        if self.num_selected_features_input.get() == '' or not self.num_selected_features_input.get().isnumeric():
            return False, "num_selected_features"
        return True, ""

    # This method set the Fires title
    def set_title(self):
        self.title = tk.Label(self, text="Fires Param:",font=title_font)
        self.title.grid(row=0, column=0,sticky = W)

    # This method set the default params
    def set_default_params(self):
        self.params["mu_init"] = 0
        self.params["sigma_init"] = 1
        self.params["penalty_s"] = 0.01
        self.params["penalty_r"] = 0.01
        self.params["epochs"] = 1
        self.params["lr_mu"] = 0.01
        self.params["lr_sigma"] = 0.01
        self.params["scale_weights"] = "True"
        self.params["num_selected_features"] = 5

    # This method set the fields for the params
    def set_fields(self):
        self.mu_init_lable = tk.Label(self, text="mu_init:")
        self.mu_init_input = tk.Entry(self, textvariable=StringVar(self, self.params["mu_init"]))

        self.sigma_init_lable = tk.Label(self, text="sigma_init:")
        self.sigma_init_input = tk.Entry(self, textvariable=StringVar(self, self.params["sigma_init"]))

        self.penalty_s_lable = tk.Label(self, text="penalty_s:")
        self.penalty_s_input = tk.Entry(self, textvariable=StringVar(self, self.params["penalty_s"]))

        self.penalty_r_lable = tk.Label(self, text="penalty_r:")
        self.penalty_r_input = tk.Entry(self, textvariable=StringVar(self, self.params["penalty_r"]))

        self.epochs_lable = tk.Label(self, text="epochs:")
        self.epochs_input = tk.Entry(self, textvariable=StringVar(self, self.params["epochs"]))

        self.lr_mu_lable = tk.Label(self, text="lr_mu:")
        self.lr_mu_input = tk.Entry(self, textvariable=StringVar(self, self.params["lr_mu"]))

        self.lr_sigma_lable = tk.Label(self, text="lr_sigma:")
        self.lr_sigma_input = tk.Entry(self, textvariable=StringVar(self, self.params["lr_sigma"]))

        self.scale_weights_lable = tk.Label(self, text="scale_weights:")
        self.scale_weights_sv = StringVar(self, self.params["scale_weights"])
        self.scale_weights_menu = OptionMenu(self, self.scale_weights_sv, *["True","False"])

        self.num_selected_features_lable = tk.Label(self, text="num_selected_features:")
        self.num_selected_features_input = tk.Entry(self, textvariable=StringVar(self, self.params["num_selected_features"]))

    # This method set the fields position
    def set_fields_position(self):
        self.mu_init_lable.grid(row=1, column=0, sticky=W)
        self.mu_init_input.grid(row=1, column=1, sticky=W)

        self.sigma_init_lable.grid(row=2, column=0, sticky=W)
        self.sigma_init_input.grid(row=2, column=1, sticky=W)

        self.penalty_s_lable.grid(row=3, column=0, sticky=W)
        self.penalty_s_input.grid(row=3, column=1, sticky=W)

        self.penalty_r_lable.grid(row=4, column=0, sticky=W)
        self.penalty_r_input.grid(row=4, column=1, sticky=W)

        self.epochs_lable.grid(row=5, column=0, sticky=W)
        self.epochs_input.grid(row=5, column=1, sticky=W)

        self.lr_mu_lable.grid(row=6, column=0, sticky=W)
        self.lr_mu_input.grid(row=6, column=1, sticky=W)

        self.lr_sigma_lable.grid(row=7, column=0, sticky=W)
        self.lr_sigma_input.grid(row=7, column=1, sticky=W)

        self.scale_weights_lable.grid(row=8, column=0, sticky=W)
        self.scale_weights_menu.grid(row=8, column=1, sticky=W)

        self.num_selected_features_lable.grid(row=9, column=0, sticky=W)
        self.num_selected_features_input.grid(row=9, column=1, sticky=W)

    # This method updates the parameters according to the inputs
    def update_params(self):
        self.params.update({"mu_init": float(self.mu_init_input.get())})
        self.params.update({"sigma_init": float(self.sigma_init_input.get())})
        self.params.update({"penalty_s": float(self.penalty_s_input.get())})
        self.params.update({"penalty_r": float(self.penalty_r_input.get())})
        self.params.update({"epochs": int(self.epochs_input.get())})
        self.params.update({"lr_mu": float(self.lr_mu_input.get())})
        self.params.update({"lr_sigma": float(self.lr_sigma_input.get())})
        self.params.update({"num_selected_features": int(self.num_selected_features_input.get())})
        if self.scale_weights_sv.get()=="True":
            self.params.update({"scale_weights": True})
        elif self.scale_weights_sv.get()=="False":
            self.params.update({"scale_weights": False})


# This method checks whether a string represents a float
def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
