# from multiprocessing import Process
import subprocess
import warnings
import os
# import yaml
import numpy as np
import pandas as pd
import sys
import ast
from sklearn.model_selection import KFold
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from .. import config as cfg
from .. import config_layout as cly
from datetime import datetime


def currentFuncName(n=0):
    return sys._getframe(n + 1).f_code.co_name


def print_log(log_line, verbose=True, time_stamp=True, log_file=cly.LOG_FILE_PATH):
    tm = ""
    if time_stamp:
        tm = datetime.now().strftime("%Y-%m-%d %H:%M:%S: ")
    if verbose:
        if log_file is None:
            print(tm + log_line)
        else:
            with open(log_file, 'a') as file:
                file.write(tm + log_line + "\n")


def mount_nextcloud(nextCloudUser="DEEP_IAM-41539aaf-8a57-4a53-9707-26c77f635c69",
                    mountHere="/home/petersi/davfs",
                    sudo=True):
    command = [
        "mount",
        "-t",
        "davfs",
        "-o",
        "noexec",
        f"https://data-deep.a.incd.pt/remote.php/dav/files/{nextCloudUser}",
        f"{mountHere}",
    ]
    if sudo:
        command = ["sudo"] + command

    result = subprocess.Popen(
        command
    )  # , stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = result.communicate()
    if error:
        warnings.warn(f"Error while mounting NextCloud: {error}")
    return output, error


def umount_nextcloud(umountFrom="/home/petersi/davfs", sudo=True):
    try:
        print_log(f"running {currentFuncName()}")
        command = ["umount", f"{umountFrom}"]
        if sudo:
            command = ["sudo"] + command

        result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = result.communicate()
        if error:
            warnings.warn(f"Error while mounting NextCloud: {error}")
        return output, error
    except Exception as err:
        print_log(f"{currentFuncName()}: Unexpected {err=}, {type(err)=}")


def rclone_directory(fromPath, toPath):
    try:
        print_log(f"running {currentFuncName()}")
        command = ["rclone", "copy", f"rshare:{fromPath}", f"{toPath}"]
        result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = result.communicate()
        if error:
            warnings.warn(f"Error while mounting NextCloud: {error}")
        return output, error
    except Exception as err:
        print_log(f"{currentFuncName()}: Unexpected {err=}, {type(err)=}")


def check_path_existence_list(listOfPaths):
    try:
        print_log(f"running {currentFuncName()}")
        if isinstance(listOfPaths, str):
            listOfPaths = (listOfPaths,)
        notFound = []
        for path in listOfPaths:
            if os.path.exists(path) is False:
                notFound.append(path)
        if len(notFound) > 0:
            print_log(f"{currentFuncName()}: Number of not found paths: {len(notFound)}")
        return notFound
    except Exception as err:
        print_log(f"{currentFuncName()}: Unexpected {err=}, {type(err)=}")


def delete_old_files(dir_path, file_extension=".csv"):
    try:
        print_log(f"running {currentFuncName()}")
        if os.path.isdir(dir_path):
            for fl in os.listdir(dir_path):
                file_path = os.path.join(dir_path, fl)
                if os.path.isfile(file_path) and os.path.splitext(fl)[1] == file_extension:
                    print_log(f"{currentFuncName()}: os.remove({file_path})")
                    os.remove(file_path)
        else:
            print_log(f"{currentFuncName()}: Error: path == {dir_path} is not directory")
    except Exception as err:
        print_log(f"{currentFuncName()}: Unexpected {err=}, {type(err)=}")


def make_dataset(csv_input_path, config_yaml):
    try:
        print_log(f"running {currentFuncName()}")
        csv_file = pd.read_csv(csv_input_path, na_filter=False, header=[0, 1])
        col_ind_X = [i for i in range(len(csv_file.columns))
                     if csv_file.columns[i][0].split("__")[0] in ast.literal_eval(config_yaml['input_data_d1']) +
                     ast.literal_eval(config_yaml['input_data_d2']) and csv_file.columns[i][1] in
                     ast.literal_eval(config_yaml['dataset'][0]['ORP_list'])]
        col_ind_Y = [i for i in range(len(csv_file.columns))
                     if csv_file.columns[i][0].split("__")[0] == config_yaml['measurements']
                     and csv_file.columns[i][1] in ast.literal_eval(config_yaml['dataset'][0]['ORP_list'])]
        data_X = csv_file.iloc[:, col_ind_X].values.tolist()
        data_Y = csv_file.iloc[:, col_ind_Y].values.tolist()
        return (data_X, data_Y)
    except Exception as err:
        print_log(f"{currentFuncName()}: Unexpected {err=}, {type(err)=}")


def append_new_column_to_csv(input_path, output_path, columns, column_names):
    df = pd.read_csv(input_path, na_filter=False, header=[0, 1])
    for i in range(len(column_names)):
        df[column_names[i]] = pd.DataFrame(columns[i])

    df.to_csv(output_path, index=False)


def define_model(parameters):  # dotiahnut hodnoty z configu
    try:
        print_log(f"running {currentFuncName()}")
        model = Sequential()
        parameters = parameters["model_parameters"]
        # neural network architecture
        nn_layer = parameters["nn_layer"]
        for i in range(len(nn_layer)):
            keys = nn_layer[i].keys()
            layerName, otherSettings = [], []
            s = list()
            for key in keys:
                if key == "name":
                    layerName = nn_layer[i]["name"]
                elif key == "other_settings":
                    otherSettings = [nn_layer[i]["other_settings"]]
                else:
                    s.append(key + "=" + nn_layer[i][key])
            s = otherSettings + s
            s = ", ".join(s)
            s = layerName + "(" + s + ")"
            print_log(f"eval(model.add({s}))")
            eval("model.add(" + s + ")")

        # optimizer settings
        opt = parameters["optimizer"]
        keys = opt.keys()
        optName, otherSettings = [], []
        s = list()
        for key in keys:
            if key == "name":
                optName = opt["name"]
            elif key == "other_settings":
                otherSettings = [opt["other_settings"]]
            else:
                s.append(key + "=" + opt[key])
        s = otherSettings + s
        s = ", ".join(s)
        s = optName + "(" + s + ")"
        optimizer = s

        # model compile
        mcompile = parameters["model_compile"]
        keys = mcompile.keys()
        otherSettings = []
        s = list()
        print_log(f"{currentFuncName()}: optimizer0 == {optimizer}")
        for key in keys:
            if key == "optimizer":
                if mcompile["optimizer"] != "":
                    optimizer = mcompile["optimizer"]
                optimizer = ["optimizer = " + optimizer, ]
            elif key == "other_settings":
                otherSettings = [mcompile["other_settings"]]
            else:
                s.append(key + "=" + mcompile[key])
        s = otherSettings + optimizer + s
        s = ", ".join(s)
        print_log(f"{currentFuncName()}: model.compile({s})")
        eval("model.compile(" + s + ")")
        return model
    except Exception as err:
        print_log(f"{currentFuncName()}: Unexpected {err=}, {type(err)=}")


def train_model(dataX, dataY, parameters):
    try:
        print_log(f"running {currentFuncName()}")
        n_folds = parameters["train_model_settings"]["n_folds"]
        epochs = parameters["train_model_settings"]["epochs"]
        batch_size = parameters["train_model_settings"]["batch_size"]
        scores, histories, models = list(), list(), list()

        kfold = KFold(n_folds, shuffle=True, random_state=1)  # prepare cross validation
        for train_ix, test_ix in kfold.split(dataX):  # enumerate splits
            model = define_model(parameters)  # define model
            trainX = [dataX[train_ix[i]] for i in range(len(train_ix))]
            trainY = [dataY[train_ix[i]] for i in range(len(train_ix))]
            testX = [dataX[test_ix[i]] for i in range(len(test_ix))]
            testY = [dataY[test_ix[i]] for i in range(len(test_ix))]

            history = model.fit(
                trainX,
                trainY,
                epochs=epochs,  # fit model
                batch_size=batch_size,
                validation_data=(testX, testY),
                verbose=0,
            )
            _, acc = model.evaluate(testX, testY, verbose=0)  # evaluate model
            print_log(f"{currentFuncName()}: > %.3f" % (acc * 100.0))
            scores.append(acc)  # stores scores
            histories.append(history)
            models.append(model)
        return scores, histories, models
    except Exception as err:
        print_log(f"{currentFuncName()}: Unexpected {err=}, {type(err)=}")


def prediction_vector(model, dataX):
    try:
        print_log(f"running {currentFuncName()}")
        prediction = model.predict(dataX)
        print_log(f"{currentFuncName()}: prediction = np.argmax(prediction, axis=1)")
        prediction = np.argmax(prediction, axis=1)
        print_log(f"{currentFuncName()}: return prediction")
        return prediction
    except Exception as err:
        print_log(f"{currentFuncName()}: Unexpected {err=}, {type(err)=}")


def unlist(x):
    if isinstance(x, list):
        return x[0]
    else:
        return x


def test_model(model, dataX, dataY=[]):
    try:
        print_log(f"running {currentFuncName()}")
        prediction = prediction_vector(model, dataX)
        if dataY == []:
            return prediction
        else:
            acc = len([1 for i in range(len(dataY)) if unlist(dataY[i]) == prediction[i]]) / len(dataY)
            return prediction, acc
    except Exception as err:
        print_log(f"{currentFuncName()}: Unexpected {err=}, {type(err)=}")


def load_model(modelPath, parameters):
    try:
        print_log(f"running {currentFuncName()}")
        model = define_model(parameters)
        print_log(f"{currentFuncName()}: modelPath == {modelPath}")
        model.load_weights(modelPath, skip_mismatch=False, by_name=False, options=None)
        return model
    except Exception as err:
        print_log(f"{currentFuncName()}: Unexpected {err=}, {type(err)=}")


def save_model(model, modelPath):
    try:
        print_log(f"running {currentFuncName()}")
        model.save_weights(modelPath)
        return 1
    except Exception as err:
        print_log(f"{currentFuncName()}: Unexpected {err=}, {type(err)=}")
