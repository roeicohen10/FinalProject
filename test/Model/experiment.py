from Model.Simulation.experiment import Experiment
from Model.OFS.ofs_ac import OnlineFeatureSelectionAC
from Model.OL.ol_ac import OnlineLearningAC
from Model.Simulation.parse import Parse


ofs_algos = OnlineFeatureSelectionAC.get_all_ofs_algo()
ol_models = OnlineLearningAC.get_all_ol_algo()

def single_experiment_test(file_path,file_name,file_target_index=-1,window_size=300,ol_index=0,ofs_index=0):
    print(ofs_algos[ofs_index].get_algorithm_default_parameters())
    print(ol_models[ofs_index].get_model_default_parameters())

    X, y, classes = Parse.read_ds(file_path, target_index=file_target_index)
    ofs_instance, ol_instance = ofs_algos[ofs_index](), ol_models[ol_index]()
    experiment = Experiment(ofs=ofs_instance, ol=ol_instance, window_size=window_size, X=X, y=y, ds_name=file_name,
                            transform_binary=False, special_name='single_test')
    experiment.ol.set_algorithm_fit_parameters(classes=classes)
    try:
        experiment.run()
        experiment.save()
    except Exception as e:
        print(experiment)
        print(e)


def multi_experiments_test(file_path,file_name,file_target_index=-1,window_sizes=[300,500]):
    export_path = r"C:\Users\Roi\Desktop\check"

    X, y, classes = Parse.read_ds(file_path, target_index=file_target_index)
    ds_exps = []
    for window_size in window_sizes:
        for ofs in ofs_algos:
            ofs_instance = ofs()
            for ol in ol_models:
                ol_instance = ol()
                ol_instance.set_algorithm_fit_parameters(classes=classes)
                experiment = Experiment(ofs=ofs_instance,ol=ol_instance , window_size=window_size, X=X, y=y, ds_name=file_name, transform_binary=False ,special_name='multi')
                ds_exps.append(experiment)


    Experiment.run_multiple_experiments(ds_exps)
    Experiment.save_multiple_experiments(ds_exps, path=export_path)



if __name__ == '__main__':
    file_path = r'C:\Users\Roi\Documents\Degree\Semester 8\פרוייקט גמר\datasets\new\Ozone Level Detection Data Set\ozone.csv'
    file_name = 'Ozone'

    single_experiment_test(file_path,file_name, file_target_index=-1, window_size=300, ol_index=0, ofs_index=0)
    # multi_experiments_test(file_path, file_name, file_target_index=-1, window_sizes=[300, 500])
