from Backend.Streaming.Streaming import Stream_Data
from Backend.OL.OLModel import MODELS
from Backend.OFS.OFSAlgo import Algorithms
from Backend.Analyze import Analyze
import scipy.io
import os

def run_simulation(path,target_name,target_index,fs_model_index,fs_model_parms,ol_model_index,batch_size=50,mat=False):
    stream = Stream_Data()
    stream.set_batch_size(batch_size)
    stream.set_num_feature(50)
    stream.set_ol(MODELS[ol_model_index])
    stream.set_ofs(Algorithms[fs_model_index])
    if mat:
        mat = scipy.io.loadmat(path)
        X = mat['X']  # data
        X = X.astype(float)
        y = mat['Y']  # label
        y = y[:, 0]
        y = y.astype(float)
        stream.set_X_Y(X, y)
    else:
        stream.set_data(path, target_name)
        stream.prepare_data(target_index,shuffle=True)
    stream.params = fs_model_parms

    print("start simulate")
    stream.simulate_stream(inc_num=False)
    print("end simulate")

    analyze(stream.stats)

def analyze(stats):
    data = Analyze(stats)
    data.show_accuracy_measures_plot()
    data.show_time_measures_plot()
    data.show_memory_measures_plot()
    data.show_accuracy_for_number_of_features()
    data.show_number_of_features()
    print(f"Num of features per epoch: {[len(x) for x in stats['features']]}")


if __name__ == "__main__":
    fs_model_parms = {
        "w0":0.05,
        "dw":0.05,
        "batch_size": 250
    }

    path = "../data/COIL20.mat"
    suffix = os.path.basename(path).split(".")[1]
    mat = True if suffix == "mat" else False
    run_simulation(path, target_name="GPS Spoofing", target_index=21, fs_model_index=0, fs_model_parms=fs_model_parms, ol_model_index=0, batch_size=250, mat=mat)


