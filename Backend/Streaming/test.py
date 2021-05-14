import time, os, logging, psutil
from Backend.OL.OLModel import OL_MODELS
from Backend.OFS.OFSAlgo import OFS_ALGO
from Backend.utils.report import Report
from skmultiflow.drift_detection import DDM
import matplotlib.pyplot as plt
from scipy.io import arff
import arff, numpy as np, pandas as pd
import copy, seaborn as sns

# get final project path
DIR_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_EXPORT_PATH = os.path.join(DIR_PATH, 'data/Experiments')

def init_model(model, model_params, X_train, y_train, ofs_algo=None, ol_fit_params=None, ofs_params=None,
               window_size=30):
    ol_model = model(**model_params)
    selected_features = None
    ofs_running_time, ol_running_time = [], []

    if ofs_algo is None:
        start_t = time.perf_counter()
        ol_model = ol_model.fit(X_train, y_train, **ol_fit_params)
        ol_running_time.append(time.perf_counter() - start_t)
        print("here")
    else:
        start_t = time.perf_counter()
        selected_features, params = ofs_algo(X_train, y_train,
                                             param=ofs_params)
        ofs_running_time.append(time.perf_counter() - start_t)

        if len(selected_features) == 0:
            logging.error(f"Could not find features for window size {window_size}")
            raise Exception("Could not find features")

        print(f"selected features: {len(selected_features)}")

        start_t = time.perf_counter()
        ol_model = ol_model.fit(X_train[:, selected_features], y_train, **ol_fit_params)
        ol_running_time.append(time.perf_counter() - start_t)

    return ol_model, selected_features, ol_running_time, ofs_running_time


def run_simulation(X, y, window_size=30, ol_algo=None, ol_params=None, ol_fit_params=None, ofs_algo=None,
                   ofs_params=None, lazy=True):
    start_window_size = window_size
    corrects, n_samples, samples_counter = 0, 0, 0
    ol_running_time, ofs_running_time, memory_usage = [], [], []
    first_size_window, prequential_accuracy, stream, selected_features_tf_records = [], [], [], []
    all_selected_features_len, selected_features, selected_features_concept, selected_features_true_false = [], [], [], []
    ddm = DDM()
    for record in range(X.shape[0]):

        record_x, record_y = np.array([X[record, :]]), np.array([y[record]])
        if record < window_size:
            first_size_window.append(record)
            continue
        if record == window_size:
            try:
                ol_model, selected_features, ol_run_time, ofs_run_time = init_model(ol_algo, ol_params,
                                                                                    X[first_size_window, :],
                                                                                    y[first_size_window],
                                                                                    ofs_algo=ofs_algo,
                                                                                    ol_fit_params=ol_fit_params,
                                                                                    ofs_params=ofs_params,
                                                                                    window_size=window_size)
                if ofs_algo:
                    selected_features_concept.append(list(selected_features))
                    selected_features_tf_records.append(record)
            except Exception as e:

                if window_size > start_window_size * 5:
                    raise Exception("OFS could not find features.")

                first_size_window.append(record)
                window_size += 50
                logging.info(f"Changed window size from {window_size - 50} to {window_size}")
                print(f"Changed window size from {window_size - 50} to {window_size}")
                continue

            ol_running_time.extend(ol_run_time)
            if ofs_algo is not None:
                ofs_running_time.extend(ofs_run_time)
            continue

        my_pred = ol_model.predict(record_x) if ofs_algo is None else ol_model.predict(
            record_x[:, selected_features])
        samples_counter += 1
        if record_y[0] == my_pred[0]:
            corrects += 1

        ddm.add_element(corrects / samples_counter)
        prequential_accuracy.append(corrects / samples_counter)
        memory_usage.append(psutil.Process(os.getpid()).memory_info().rss)
        stream.append(record)
        if ofs_algo:
            all_selected_features_len.append(len(selected_features))
            selected_features_true_false.append([index in selected_features for index in range(X.shape[1])])


        if lazy:
            ol_model.partial_fit(record_x[:, selected_features], record_y) if ofs_algo is not None else ol_model.partial_fit(record_x, record_y)
        if ddm.detected_change():
            print("concept")
            while True:
                try:
                    ol_model, selected_features, ol_run_time, ofs_run_time = init_model(ol_algo, ol_params,
                                                                                        X[record - window_size:record,
                                                                                        :],
                                                                                        y[record - window_size:record],
                                                                                        ofs_algo=ofs_algo,
                                                                                        ol_fit_params=ol_fit_params,
                                                                                        ofs_params=ofs_params,
                                                                                        window_size=window_size)
                    if ofs_algo:
                        selected_features_concept.append(list(selected_features))
                        selected_features_true_false.append([index in selected_features for index in range(X.shape[1])])
                        selected_features_tf_records.append(record)
                    break
                except Exception:
                    if window_size > start_window_size * 5:
                        raise Exception("OFS could not find features.")
                    window_size += 50
                    logging.info(f"Changed window size from {window_size - 50} to {window_size}")
                    print(f"Changed window size from {window_size - 50} to {window_size}")

            ol_running_time.extend(ol_run_time)

            if ofs_algo is not None:
                ofs_running_time.extend(ofs_run_time)

    return prequential_accuracy, ol_running_time, ofs_running_time, \
           selected_features_concept, all_selected_features_len, selected_features_true_false, \
           stream, selected_features_tf_records, memory_usage


def run_experiment(ds_name, ds_path, classes=None):
    window_sizes = [300, 500]
    dataset = arff.load(open(ds_path))
    if not classes:
        try:
            for tup in dataset['attributes']:
                if tup[0] == 'target':
                    classes = [int(target) for target in tup[1]]
        except Exception:
            classes = [1, 2]

    export_ds_path = DEFAULT_EXPORT_PATH if not create_dir(DEFAULT_EXPORT_PATH, ds_name) else os.path.join(
        DEFAULT_EXPORT_PATH, ds_name)

    data = np.array(dataset['data'])
    x_train, y_train = data[:, :-1].astype(np.float), data[:, -1].astype(np.int)

    ofs_experiments_results = {}
    ol_experiments_results = {}
    for wind_size in window_sizes:
        wind_path = ds_path if not create_dir(export_ds_path, str(wind_size)) else os.path.join(export_ds_path,
                                                                                                str(wind_size))

        for ofs_algo in OFS_ALGO:
            ofs_path = ds_path if not create_dir(wind_path, ofs_algo['name']) else os.path.join(wind_path,
                                                                                                ofs_algo['name'])

            if ofs_algo.get('name') == "Fires":
                ofs_algo["params"]["target_values"] = classes
                ofs_algo["params"]["n_total_ftr"] = x_train.shape[1]

                func, model = ofs_algo["func"](ofs_algo["params"])
                ofs_algo["func"] = func
                ofs_algo["params"] = {"num_selected_features":5,
                                      "model": model}

            for ol_model in OL_MODELS:
                ol_path = ds_path if not create_dir(ofs_path, ol_model['name']) else os.path.join(ofs_path,
                                                                                                  ol_model['name'])
                base_name = f"{wind_size}_{ofs_algo['name']}_{ol_model['name']}"

                init_loggings(path=ol_path, base_name=base_name)

                if "max_window_size" in ol_model.get("params"):
                    ol_model["params"]["max_window_size"] = wind_size
                if "classes" in ol_model.get("fit_params"):
                    ol_model["fit_params"]["classes"] = classes


                print("Starting new experiment")
                print(f"window: {wind_size}, ofs: {ofs_algo['name']}, ol: {ol_model['name']}")
                try:
                    prequential_accuracy, ol_running_time, ofs_running_time, \
                    selected_features_concept, selected_features_len, selected_features_true_false, stream, selected_features_tf_records, memory_usage = run_simulation(
                        x_train, y_train,
                        window_size=wind_size,
                        ol_algo=ol_model['func'],
                        ol_params=ol_model['params'],
                        ol_fit_params=ol_model[
                            'fit_params'],
                        ofs_algo=ofs_algo['func'],
                        ofs_params=ofs_algo['params'],
                        lazy=ol_model['lazy'])
                except Exception as e:
                    logging.error(str(e))
                    continue

                print(f"Last accuracy: {prequential_accuracy[-1]}")

                acc_image_path, mem_image_path,acc_features_image_path, true_false_features_image_name = \
                    create_plots(prequential_accuracy, selected_features_len, selected_features_true_false, base_name,
                             ol_path, stream, selected_features_tf_records, memory_usage,ofs_algo['func'])

                logs = [f"Last accuracy: {prequential_accuracy[-1]}",
                        f"Mean OL algorithm runtime: {np.mean(np.array(ol_running_time)) * 1000} ms",
                        f"Mean OFS algorithm runtime: {np.mean(np.array(ofs_running_time)) * 1000} ms",
                        f"Acc: {str(prequential_accuracy)}", f"OFS Runtime: {str(ofs_running_time)}",
                        f"OL Runtime: {str(ol_running_time)}",
                        f"Selected Features: {selected_features_concept}"]
                set_logs(logs)

                # ofs data
                if ofs_algo['name'] not in ofs_experiments_results:
                    ofs_experiments_results[ofs_algo['name']] = {"times": []}
                if f"{str(wind_size)}_acc" not in ofs_experiments_results[ofs_algo['name']]:
                    ofs_experiments_results[ofs_algo['name']][f"{str(wind_size)}_acc"] = []

                ofs_experiments_results[ofs_algo['name']]["times"].append(np.mean(np.array(ofs_running_time)) * 1000)
                ofs_experiments_results[ofs_algo['name']][f"{str(wind_size)}_acc"].append(
                    (ol_model['name'], prequential_accuracy[-1]))

                # ol data
                if ol_model['name'] not in ol_experiments_results:
                    ol_experiments_results[ol_model['name']] = {"times": []}
                if f"{str(wind_size)}_acc" not in ol_experiments_results[ol_model['name']]:
                    ol_experiments_results[ol_model['name']][f"{str(wind_size)}_acc"] = []

                ol_experiments_results[ol_model['name']]["times"].append(np.mean(np.array(ol_running_time)) * 1000)
                ol_experiments_results[ol_model['name']][f"{str(wind_size)}_acc"].append(prequential_accuracy[-1])
                report_params = {
                    'ol_algo': f"{ol_model['name']}({ol_model['params']})",
                    'ofs_algo': f"{ofs_algo['name']}({ofs_algo['params']})",
                    'window_size': str(wind_size),
                    'ol_runtime': f"{ol_experiments_results[ol_model['name']]['times'][-1]} ms",
                    'ofs_runtime': f"{ofs_experiments_results[ofs_algo['name']]['times'][-1]} ms",
                    'accuracy': f'{prequential_accuracy[-1]}',
                    'selected_features': f'{selected_features_concept}',
                    'first_image': acc_image_path,
                    'second_image': mem_image_path,
                    'export_path': ol_path
                }
                if ofs_algo['func']:
                    report_params['third_image'] = acc_features_image_path
                    report_params['forth_image'] = true_false_features_image_name
                # create report
                Report.create_single_experiment_report(**report_params)

    ofs_images_paths = create_scatter(ofs_experiments_results, export_ds_path)

    ofs_time = calc_mean_runtime(ofs_experiments_results)
    ol_times = calc_mean_runtime(ol_experiments_results)
    init_loggings(export_ds_path, "times")

    set_logs([ofs_time, ol_times])
    create_ds_report(ds_name,",".join(str(ws) for ws in window_sizes), ofs_images_paths,ofs_time, ol_times, export_ds_path)
    Report.combine_ds_experiments_reports(ds_name)

def create_ds_report(ds_name,window_sizes, ofs_images, ofs_runtimes, ol_runtimes, export_path):
    ds_params = {
        'ds_name':ds_name,
        'window_sizes': window_sizes,
        'wo_image': ofs_images.get('-', ''),
        'ai_image': ofs_images.get('Alpha Investing',''),
        'osfs_image': ofs_images.get('OSFS',''),
        'fosfs_image': ofs_images.get('Fast OSFS',''),
        'saola_image': ofs_images.get('SAOLA',''),
        'saola_runtime': str(ofs_runtimes.get('SAOLA','')),
        'ai_runtime': str(ofs_runtimes.get('Alpha Investing','')),
        'osfs_runtime':str(ofs_runtimes.get('OSFS','')),
        'fosfs_runtime':str(ofs_runtimes.get('Fast OSFS','')),
        'nn_runtime': str(ol_runtimes.get('Neural Network','')),
        'knn_3_runtime': str(ol_runtimes.get('K-Nearest Neighbors 3','')),
        'knn_5_runtime': str(ol_runtimes.get('K-Nearest Neighbors 5','')),
        'nb_runtime': str(ol_runtimes.get('Naive Bayes','')),
        'rf_runtime': str(ol_runtimes.get('Random Forest', '')),
        'export_path': export_path
    }
    Report.create_ds_report(**ds_params)

def create_plots(prequential_accuracy, selected_features_len, selected_features_true_false,
                 base_name, ol_path, stream, selected_features_tf_records, memory_usage,ofs_algo):

    acc_image_path = os.path.join(ol_path, f"acc_{base_name}.png")
    acc_features_image_path = os.path.join(ol_path, f"acc_features_{base_name}.png")
    mem_image_path = os.path.join(ol_path, f"mem_{base_name}.png")
    true_false_features_image_path = os.path.join(ol_path, f"true_false_features_{base_name}.png")

    create_plot(x_val=list(range(len(prequential_accuracy))), y_val=prequential_accuracy,
                file_path=acc_image_path,
                title="Prequential Accuracy for each streamed record", x_label="TimeStamp", y_label="Accuracy")

    create_plot(x_val=stream, y_val=memory_usage,
                file_path=mem_image_path,
                title="Memory Usage", x_label="TimeStamp",
                y_label="Usage (Bytes)", scatter=False)

    if ofs_algo:
        create_plot(x_val=stream, y_val=selected_features_len,
                    file_path=acc_features_image_path,
                    title="Number of selected features per timestamp", x_label="TimeStamp",
                    y_label="Number of features", scatter=False)

        create_true_false_scatter(x_val=selected_features_tf_records, y_val=list(range(len(selected_features_true_false[0]))),
                                  true_false=selected_features_true_false,
                                  file_path=true_false_features_image_path,
                                  title="Selected and Unselected features per timestamp", x_label="TimeStamp",
                                  y_label="Features")
    return acc_image_path, mem_image_path,acc_features_image_path, true_false_features_image_path

def calc_mean_runtime(experiments_results):
    times = {}
    for key, val in experiments_results.items():
        times[key] = np.mean(np.array(val["times"]))
    return times


def init_loggings(path, base_name):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_name = f"log_{base_name}.log"
    logging.basicConfig(filename=os.path.join(path, log_name),
                        filemode='w',
                        format='%(message)s',
                        level=logging.INFO)


def set_logs(logs):
    for log in logs:
        logging.info(log)


def create_plot(x_val, y_val, file_path, title="", x_label="", y_label="", scatter=False):
    plt.clf()
    if not scatter:
        plt.plot(x_val, y_val)
    else:
        plt.scatter(x_val, y_val)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig(file_path)


def create_true_false_scatter(x_val, y_val, true_false, file_path, title, x_label, y_label):
    plt.clf()
    plt.yticks(y_val)
    plt.xticks(x_val)
    counter = 0
    for xe, ye in zip(x_val, true_false):

        true = [index for index in range(len(ye)) if ye[index]]
        false = list(set(y_val).difference(set(true)))

        plt.scatter([xe] * len(true), true, marker="+", c="black")

        plt.scatter([xe] * len(false), false, marker="_", c="black")
        counter+=1

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(file_path)



def create_scatter(ofs_res, file_path):
    markers = ["x", "P", "*", "<","."]
    colors = ["red", "green", "blue", "yellow", "pink", "black", "orange", "purple", "beige", "brown", "gray", "cyan",
              "magenta"]
    ol_scatter_markers = {ol['name']: markers[index] for index, ol in enumerate(OL_MODELS)}
    paths = {}
    for name, vals in ofs_res.items():
        plt.clf()
        f, ax = plt.subplots()
        color_index = 0
        plots, names = [], set()
        for key, val in vals.items():
            if "acc" in key:
                window_size = key.split("_")[0]
                for ind in range(len(val)):
                    handle = ax.scatter(x=window_size, y=val[ind][1],
                                        marker=ol_scatter_markers[val[ind][0]], color=colors[color_index], alpha=0.45)
                    names.add(val[ind][0])
                    plots.append(copy.copy(handle))
                color_index += 1

        ax.set_title("Accuracy per Window Size")
        ax.set_xlabel("Window Size")
        ax.set_ylabel("Accuracy")

        for h in plots:
            h.set_color("black")

        legends = ax.legend(plots,
                            list(names),
                            loc='upper left',
                            bbox_to_anchor=(1, 0.5),
                            fontsize=8,
                            labelcolor="black")

        path = f"{os.path.join(file_path, name)}.png"
        plt.savefig(path, bbox_extra_artists=(legends,), bbox_inches='tight')
        paths[name] = path
    return paths
def create_dir(path, name):
    full_path = os.path.join(path, name)
    try:
        os.mkdir(full_path)
    except FileExistsError:
        return True
    except OSError:
        return False
    return True


if __name__ == '__main__':
    files_paths = [
        r'C:\Users\Roi\Documents\Degree\Semester 8\פרוייקט גמר\datasets\new\Wafer\Wafer_TRAIN.arff',
        r'C:\Users\Roi\Documents\Degree\Semester 8\פרוייקט גמר\datasets\new\NonInvasiveFetalECGThorax1\NonInvasiveFetalECGThorax1_TRAIN.arff',
        r'C:\Users\Roi\Documents\Degree\Semester 8\פרוייקט גמר\datasets\new\TwoPatterns\TwoPatterns_TRAIN.arff',
        r'C:\Users\Roi\Documents\Degree\Semester 8\פרוייקט גמר\datasets\new\SemgHandSubjectCh2\SemgHandSubjectCh2_TRAIN.arff',
    ]
    file_names = ['Wafer','NonInvasiveFetalECGThorax1','TwoPatterns','SemgHandSubjectCh2']

    for file_path, file_name in zip(files_paths, file_names):
        try:
            run_experiment(ds_name=file_name,
                       ds_path=file_path)
        except Exception as e:
            print(e)
            pass
    # x_val =[10,20,30,40,50]
    # y_val = list(range(150))
    # true_false = [[np.random.random()> 0.5 for z in range(150)] for x in range(5)]
    # file_path=""
    # title = "Selected and Unselected features per timestamp"
    # x_label = "TimeStamp"
    # y_label = "Features"
    # create_true_false_scatter(x_val, y_val, true_false, file_path, title, x_label, y_label)
