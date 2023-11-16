from multiprocessing import Process
import subprocess
import warnings
import os
import yaml
import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
import uc-microstep-mis-ai4eosc_thunder_nowcast_ml.config as cfg
from datetime import datetime

currentFuncName = lambda n=0: sys._getframe(n + 1).f_code.co_name


def print_log(log_line, verbose=True, time_stamp=True, log_file=cfg.LOG_FILE_PATH):
    tm = ""
    if time_stamp:
        tm = datetime.now().strftime("%Y-%m-%d %H:%M:%S: ")
    if verbose:
        if log_file is None:
            print(tm + log_line)
        else:
            with open(log_file, 'a') as file:
                file.write(tm + log_line + "\n")


def mount_nextcloud(
    nextCloudUser="DEEP_IAM-41539aaf-8a57-4a53-9707-26c77f635c69",
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
    command = ["umount", f"{umountFrom}"]
    if sudo:
        command = ["sudo"] + command

    result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = result.communicate()
    if error:
        warnings.warn(f"Error while mounting NextCloud: {error}")
    return output, error


def rclone_directory(fromPath, toPath):
    command = ["rclone", "copy", f"rshare:{fromPath}", f"{toPath}"]
    result = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = result.communicate()
    if error:
        warnings.warn(f"Error while mounting NextCloud: {error}")
    return output, error


def check_path_existence_list(listOfPaths):
    if isinstance(listOfPaths, str):
        listOfPaths = (listOfPaths,)
    notFound = []
    for path in listOfPaths:
        if os.path.exists(path) is False:
            notFound.append(path)
    if len(notFound) > 0:
        print_log(f"{currentFuncName()}: Number of not found paths: {len(notFound)}")
    return notFound


def delete_old_files(dir_path, file_extension=".csv"):
    if os.path.isdir(dir_path):
        for fl in os.listdir(dir_path):
            file_path = os.path.join(dir_path, fl)
            if os.path.isfile(file_path) and os.path.splitext(fl)[1] == file_extension:
                os.remove(file_path)
    else:
        print_log(f"{currentFuncName()}: Error: path == {dir_path} is not directory")


def make_dataset(csv_input_path, config_yaml):
    csv_file = pd.read_csv(csv_input_path, na_filter=False, header=[0, 1])
    col_ind_X = [i for i in range(len(csv_file.columns))
                 if csv_file.columns[i][0].split("__")[0] in eval(config_yaml['all_data_models'])
                    and csv_file.columns[i][1] in eval(config_yaml['ORP_list'])]
    col_ind_Y = [i for i in range(len(csv_file.columns))
                 if csv_file.columns[i][0].split("__")[0] == config_yaml['measurements']
                    and csv_file.columns[i][1] in eval(config_yaml['ORP_list'])]
    data_X = csv_file.iloc[:, col_ind_X].values.tolist()
    data_Y = csv_file.iloc[:, col_ind_Y].values.tolist()
    return (data_X, data_Y)


def define_model(parameters):  # dotiahnut hodnoty z configu
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
    print_log(f"optimizer0 == {optimizer}")
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
    print_log("model.compile(" + s + ")")
    eval("model.compile(" + s + ")")
    return model


def train_model(dataX, dataY, parameters):
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
        print_log("> %.3f" % (acc * 100.0))
        scores.append(acc)  # stores scores
        histories.append(history)
        models.append(model)
    return scores, histories, models


def prediction_vector(model, dataX):
    prediction = model.predict(dataX)
    prediction = np.argmax(prediction, axis=1)
    return prediction


def test_model(model, dataX, dataY=[]):
    prediction = prediction_vector(model, dataX)
    if dataY == []:
        return prediction
    else:
        acc = np.sum(prediction == dataY) / len(dataY)
        return prediction, acc


def load_model(modelPath, parameters):
    model = define_model(parameters)
    print_log(f"modelPath == {modelPath}")
    model.load_weights(modelPath, skip_mismatch=False, by_name=False, options=None)
    return model


def save_model(model, modelPath):
    model.save_weights(modelPath)
    return 1
