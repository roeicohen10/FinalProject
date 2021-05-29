import tkinter.filedialog
import tkinter as tk
from tkinter import *
from View.Frames import *


# Online learning options
OL_OPTIONS = [
    'K-NN',
    'Perceptron Mask (ANN)',
    'Random Forest',
    'Naive-Bayes'
]

# Online feature selection options
OFS_OPTIONS = [
    'Alpha Investing',
    'SAOLA',
    'OSFS',
    'F-OSFS',
    'Fires',
    'Without OFS'
]

# This class is the main window in the UI
class Window:

    # This method is the contractor of the class.
    # The contractor defines the frames
    # The contractor calls the methods that fill the frames
    # The contractor calls the methods that set frames positions in the window
    def __init__(self,root):
        self.root=root
        self.exp_params_frame=tk.Frame(root)
        self.ol_params_frame=tk.Frame(root)
        self.ofs_params_frame = tk.Frame(root)

        self.ol_algorithms={}
        self.ofs_algorithms={}


        self.set_frame_position()

        self.fill_exp_params_frame()
        self.fill_ol_params_frame()
        self.fill_ofs_params_frame()

        self.set_item_position()

    # This method fill the experiment params frame
    def fill_exp_params_frame(self):
        self.create_exp_params()
        self.create_db_chooser()
        self.create_save_path_chooser()
        self.create_selected_algo()
        self.run_btn = tk.Button(self.exp_params_frame, text="Run", command=self.run, relief=GROOVE)

    # This method fill the OL params frame
    def fill_ol_params_frame(self):
        tk.Label(self.ol_params_frame, text="OL Params:").grid(row=0, column=0,sticky = W)
        self.create_ol_params()

    # This method fill the OFS params frame
    def fill_ofs_params_frame(self):
        tk.Label(self.ofs_params_frame, text="OFS Params:").grid(row=0, column=0,sticky = W)
        self.create_ofs_params()

    # This method creates fields for the experiment parameters
    def create_exp_params(self):
        self.bach_size_label=tk.Label(self.exp_params_frame, text="Enter bach size:")
        self.bach_size_input=tk.Entry(self.exp_params_frame)

        self.target_index_lable=tk.Label(self.exp_params_frame, text="Enter target index:")
        self.target_index_input = tk.Entry(self.exp_params_frame)

    # This method creates the fields for DB selection
    def create_db_chooser(self):
        self.fc_lable=tk.Label(self.exp_params_frame, text="Choose dataset:")
        self.btn_browse_to_db=tk.Button(self.exp_params_frame, text="Browse", command=self.open_browse_to_db_path, relief=GROOVE)
        self.path_to_db = StringVar()
        self.file_path=tk.Label(self.exp_params_frame, textvariable=self.path_to_db, relief=GROOVE, width=50, height=1)

    # This method creates the fields for selecting a path for saving the experiments
    def create_save_path_chooser(self):
        self.dc_lable=tk.Label(self.exp_params_frame, text="Path to save:")
        self.btn_browse_to_save_path=tk.Button(self.exp_params_frame, text="Browse", command=self.open_browse_path_to_save, relief=GROOVE)
        self.path_to_save = StringVar()
        self.save_path=tk.Label(self.exp_params_frame, textvariable=self.path_to_save, relief=GROOVE, width=50, height=1)

    # This method create fields for OL params
    def create_ol_params(self):
        self.knn_params = KNN_Param(self.ol_params_frame)
        self.ol_algorithms.update({OL_OPTIONS[0]:self.knn_params})

        self.nn_params = NN_Param(self.ol_params_frame)
        self.ol_algorithms.update({OL_OPTIONS[1]:self.nn_params})

        self.rf_params = RF_Param(self.ol_params_frame)
        self.ol_algorithms.update({OL_OPTIONS[2]:self.rf_params})

        self.nb_params = NB_Param(self.ol_params_frame)
        self.ol_algorithms.update({OL_OPTIONS[3]:self.nb_params})


    # This method create fields for OFS params
    def create_ofs_params(self):
        self.alpha_investing_params = AI_Param(self.ofs_params_frame)
        self.ofs_algorithms.update({OFS_OPTIONS[0]:self.alpha_investing_params})

        self.saola_params = SAOLA_Param(self.ofs_params_frame)
        self.ofs_algorithms.update({OFS_OPTIONS[1]:self.saola_params})

        self.osfs_params = OSFS_Param(self.ofs_params_frame)
        self.ofs_algorithms.update({OFS_OPTIONS[2]:self.osfs_params})

        self.f_osfs_params = F_OSFS_Param(self.ofs_params_frame)
        self.ofs_algorithms.update({OFS_OPTIONS[3]:self.f_osfs_params})

        self.fires_params = Fires_Param(self.ofs_params_frame)
        self.ofs_algorithms.update({OFS_OPTIONS[4]:self.fires_params})

        self.without=Params_frame(self.ofs_params_frame)
        self.ofs_algorithms.update({OFS_OPTIONS[5]:self.without})


    # This method creates selection menus for OL and OFS
    def create_selected_algo(self):
        self.ol_lable = tk.Label(self.exp_params_frame, text="Choose OL Algorithm:")
        self.ol_menu= {}
        for ol in OL_OPTIONS:
            var=tk.IntVar()
            c_box=tk.Checkbutton(self.exp_params_frame, text=ol, variable=var, onvalue=True, offvalue=False)
            self.ol_menu.update({ol:(c_box,var)})

        self.ofs_lable = tk.Label(self.exp_params_frame, text="Choose OFS Algorithm:")
        self.ofs_menu= {}
        for ofs in OFS_OPTIONS:
            var=tk.IntVar()
            c_box=tk.Checkbutton(self.exp_params_frame, text=ofs, variable=var, onvalue=True, offvalue=False)
            self.ofs_menu.update({ofs:(c_box,var)})

    # This method set all frame position in tha main window
    def set_frame_position(self):
        self.exp_params_frame.grid(row=0, column=0)
        self.ol_params_frame.grid(row=1, column=0, sticky = W)
        self.ofs_params_frame.grid(row=3, column=0, sticky = W)

    # This method set all item position in their frame
    def set_item_position(self):
        self.fc_lable.grid(row=0, column=0,sticky = W)
        self.btn_browse_to_db.grid(row=0, column=0, sticky = N)
        self.file_path.grid(row=1, column=0,sticky = W)

        self.dc_lable.grid(row=2, column=0,sticky = W)
        self.btn_browse_to_save_path.grid(row=2, column=0, sticky = N)
        self.save_path.grid(row=3, column=0,sticky = W)

        self.bach_size_label.grid(row=0, column=1, sticky = W)
        self.bach_size_input.grid(row=0, column=2, sticky = W)
        self.target_index_lable.grid(row=1, column=1, sticky = W)
        self.target_index_input.grid(row=1, column=2, sticky = W)

        self.ol_lable.grid(row=0, column=3,sticky = W)
        i=1
        for ol in OL_OPTIONS:
            self.ol_menu[ol][0].grid(row=i, column=3,sticky = W)
            i=i+1

        self.ofs_lable.grid(row=0, column=4,sticky = W)
        i = 1
        for ofs in OFS_OPTIONS:
            self.ofs_menu[ofs][0].grid(row=i, column=4,sticky = W)
            i = i + 1

        self.run_btn.grid(row=3, column=2,sticky = W)

        self.knn_params.grid(row=1, column=0,sticky = NW)
        self.nn_params.grid(row=1, column=1,sticky = NW)
        self.rf_params.grid(row=1, column=2, sticky = NW)
        self.nb_params.grid(row=1, column=3,sticky = NW)

        self.alpha_investing_params.grid(row=1, column=0,sticky = NW)
        self.saola_params.grid(row=1, column=1,sticky = NW)
        self.osfs_params.grid(row=1, column=2, sticky = NW)
        self.f_osfs_params.grid(row=1, column=3,sticky = NW)
        self.fires_params.grid(row=1, column=4, sticky=NW)

    # This method open a browse window and catch the chosen file path
    def open_browse_to_db_path(self):
        file = tk.filedialog.askopenfile(parent=self.exp_params_frame, title='Choose a file')
        if file:
            self.path_to_db.set(file.name)
            file.close()

    # This method open a browse window and catch the chosen folder path
    def open_browse_path_to_save(self):
        directory = tk.filedialog.askdirectory(parent=self.exp_params_frame, title='Choose a directory')
        self.path_to_save.set(directory)

    # This method tests validation of parameters for the experiment
    def exp_params_frame_validation(self):
        if self.path_to_db.get() == "":
            self.popup("Please choose a dataset","Error")
            return False
        if self.target_index_input.get() == "":
            self.popup("Please Enter target index","Error")
            return False
        if not self.target_index_input.get().isnumeric() or int(self.target_index_input.get())<0:
            self.popup("Please Enter a valid target index", "Error")
            return False
        if self.bach_size_input.get()== "":
            self.popup("Please Enter bach size","Error")
            return False
        if not self.bach_size_input.get().isnumeric() or int(self.bach_size_input.get())<=0:
            self.popup("Please Enter a valid bach size", "Error")
            return False
        ofs_flag=False
        for ofs in self.ofs_menu.values():
            if ofs[1].get()==1:
                ofs_flag=True
                break
        if not ofs_flag:
            self.popup("Please choose OFS algorithm", "Error")
            return False
        ol_flag=False
        for ol in self.ol_menu.values():
            if ol[1].get()==1:
                ol_flag=True
                break
        if not ol_flag:
            self.popup("Please choose OL algorithm", "Error")
            return False
        return True

    # This method checks the validation of parameters for OL algorithms
    def ol_params_frame_validation(self):
        if self.ol_menu[OL_OPTIONS[0]][1].get()==1:
            if self.knn_params.validation()[0]==False:
                self.popup("Please Enter a valid value to "+self.knn_params.validation()[1],"Error")
                return False
        if self.ol_menu[OL_OPTIONS[1]][1].get()==1:
            if self.nn_params.validation()[0]==False:
                self.popup("Please Enter a valid value to "+self.nn_params.validation()[1],"Error")
                return False
        if self.ol_menu[OL_OPTIONS[2]][1].get()==1:
            if self.rf_params.validation()[0]==False:
                self.popup("Please Enter a valid value to "+self.rf_params.validation()[1],"Error")
                return False
        if self.ol_menu[OL_OPTIONS[3]][1].get()==1:
            if self.nb_params.validation()[0]==False:
                self.popup("Please Enter a valid value to "+self.nb_params.validation()[1],"Error")
                return False
        return True

    # This method checks the validation of parameters for OFS algorithms
    def ofs_params_frame_validation(self):
        if self.ofs_menu[OFS_OPTIONS[0]][1].get()==1:
            if self.alpha_investing_params.validation()[0]==False:
                self.popup("Alpha Investing: Please Enter a valid value to "+self.alpha_investing_params.validation()[1],"Error")
                return False
        if self.ofs_menu[OFS_OPTIONS[1]][1].get()==1:
            if self.saola_params.validation()[0]==False:
                self.popup("SAOLA: Please Enter a valid value to "+self.saola_params.validation()[1],"Error")
                return False
        if self.ofs_menu[OFS_OPTIONS[2]][1].get()==1:
            if self.osfs_params.validation()[0]==False:
                self.popup("OSFS: Please Enter a valid value to "+self.osfs_params.validation()[1],"Error")
                return False
        if self.ofs_menu[OFS_OPTIONS[3]][1].get()==1:
            if self.f_osfs_params.validation()[0]==False:
                self.popup("F-OSFS: Please Enter a valid value to "+self.f_osfs_params.validation()[1],"Error")
                return False
        if self.ofs_menu[OFS_OPTIONS[4]][1].get()==1:
            if self.fires_params.validation()[0]==False:
                self.popup("Fires: Please Enter a valid value to "+self.fires_params.validation()[1],"Error")
                return False
        return True

    def get_selected_ol(self):
        selected_ol= {}
        for ol in OL_OPTIONS:
            if self.ol_menu[ol][1].get() == 1:
                selected_ol.update({ol:self.ol_algorithms.get(ol).get_params()})
        return selected_ol

    def get_selected_ofs(self):
        selected_ofs= {}
        for ofs in OFS_OPTIONS:
            if self.ofs_menu[ofs][1].get() == 1:
                selected_ofs.update({ofs:self.ofs_algorithms.get(ofs).get_params()})
        return selected_ofs

    # This method pops up a message
    def popup(self,message,title):
        popup = tk.Toplevel()
        popup.geometry('300x100')
        popup.wm_title(title)
        tk.Label(popup,text=str(message)).pack()

    def run(self):
        if not self.exp_params_frame_validation():
            return
        if not self.ol_params_frame_validation():
            return
        if not self.ofs_params_frame_validation():
            return
        file_path=self.path_to_db.get()
        path_to_save=self.path_to_save.get()
        file_target_index=int(self.target_index_input.get())
        window_sizes=[int(self.bach_size_input.get())]
        selected_ol=self.get_selected_ol()
        selected_ofs=self.get_selected_ofs()








