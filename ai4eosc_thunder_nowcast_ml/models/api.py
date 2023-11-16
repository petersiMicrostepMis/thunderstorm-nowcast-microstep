# -*- coding: utf-8 -*-
"""
Integrate a model with the DEEP API
"""

import json
import argparse
import pkg_resources
import tempfile
import mimetypes
import subprocess
import tarfile
import datetime

import os
# from io import BytesIO
import sys
import shutil
base_directory = os.path.dirname(os.path.abspath(__file__))
base_directory = os.path.dirname(os.path.dirname(base_directory))
sys.path.append(base_directory)
# import project's config.py
import ai4eosc_thunder_nowcast_ml.config as cfg
import ai4eosc_thunder_nowcast_ml.features.build_features as bf
import ai4eosc_thunder_nowcast_ml.models.model_utils as mutils
from tensorflow.keras.utils import to_categorical

from aiohttp.web import HTTPBadRequest
from functools import wraps

# Authorization
# from flaat import Flaat
# flaat = Flaat()

currentFuncName = lambda n=0: sys._getframe(n + 1).f_code.co_name


def print_log(log_line, verbose=True, time_stamp=True, log_file=cfg.LOG_FILE_PATH):
    tm = ""
    if time_stamp:
        tm = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S: ")
    if verbose:
        if log_file is None:
            print(tm + log_line)
        else:
            with open(log_file, 'a') as file:
                file.write(tm + log_line + "\n")


def _catch_error(f):
    """Decorate function to return an error as HTTPBadRequest, in case
    """
    @wraps(f)
    def wrap(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            raise HTTPBadRequest(reason=e)
    return wrap


def _fields_to_dict(fields_in):
    """
    Example function to convert mashmallow fields to dict()
    """
    dict_out = {}

    for key, val in fields_in.items():
        param = {}
        param['default'] = val.load_default
        param['type'] = type(val.load_default)
        if key == 'files' or key == 'urls':
            param['type'] = str

        val_help = val.metadata['description']
        if 'enum' in val.metadata.keys():
            val_help = "{}. Choices: {}".format(val_help,
                                                val.metadata['enum'])
        param['help'] = val_help

        try:
            val_req = val.required
        except:
            val_req = False
        param['required'] = val_req

        dict_out[key] = param
    return dict_out


def load_config_yaml_file(config_yaml_path):
    if not os.path.isfile(config_yaml_path):
        print_log(f"{currentFuncName()}: Error: yaml config is missing, config_yaml_path == {config_yaml_path}")
    else:
        config_yaml = bf.load_config_yaml(config_yaml_path)
    return config_yaml


def set_string_argument(arg_name, arg_default_value, preffix="", **kwargs):
    try:
        if not kwargs[arg_name] is None and kwargs[arg_name] != "":
            string_argument = kwargs[arg_name]
        else:
            print_log(f"{currentFuncName()}: Info: Set default value for {arg_name}: {arg_default_value}")
            string_argument = arg_default_value
    except Exception:
        string_argument = arg_default_value
    return preffix + string_argument


def set_bool_argument(arg_name, arg_default_value, **kwargs):
    bool_argument = arg_default_value
    try:
        bool_argument = kwargs[arg_name]
    except Exception:
        bool_argument = arg_default_value
    return bool_argument


def set_file_argument(arg_name, arg_default_value, **kwargs):
    try:
        if kwargs[arg_name]:
            kwargs[arg_name] = [kwargs[arg_name]]
            for fl in kwargs[arg_name]:
                filename = fl.filename
            print_log(f"{currentFuncName()}: filesize {os.path.getsize(filename)}")
            if os.path.getsize(filename) <= 2:
                filename = arg_default_value
                print_log(f"{currentFuncName()}: Info: Set default value for {arg_name}: {arg_default_value}, due to loaded file has zero size")
        else:
            print_log(f"{currentFuncName()}: Info: Set default value for {arg_name}: {arg_default_value}")
            filename = arg_default_value
    except Exception:
        filename = arg_default_value
    print_log(f"{currentFuncName()}: filename == {filename}")
    return filename


def set_kwargs(argument, arg2=None, **kwargs):
    if argument == "conf_nn":
        return set_file_argument("conf_nn", cfg.CONFIG_YAML_NN_PATH, **kwargs)
    elif argument == "conf_data":
        return set_file_argument("conf_data", cfg.CONFIG_YAML_DATA_PATH(arg2), **kwargs)
    elif argument == "model_hdf5":
        return set_file_argument("model_hdf5", cfg.MODEL_FILE_PATH, **kwargs)
    elif argument == "data_pred":
        return set_file_argument("data_pred", cfg.DEFAULT_DATA_TARGZ, **kwargs)
    elif argument == "data_train":
        return set_file_argument("data_train", cfg.DEFAULT_DATA_TARGZ, **kwargs)
    elif argument == "output_name":
        return set_string_argument("output_name", cfg.OUTPUT_NAME(arg2), **kwargs)
    elif argument == "urls_inp":
        return set_string_argument("urls_inp", "", **kwargs)
    elif argument == "urls_out":
        return set_string_argument("urls_out", "", **kwargs)
    elif argument == "path_inp":
        return set_string_argument("path_inp", "", preffix=cfg.NEXTCLOUD_DATA_DIR, **kwargs)
    elif argument == "path_out":
        return set_string_argument("path_out", "", preffix=cfg.NEXTCLOUD_DATA_DIR, **kwargs)
    elif argument == "get_default_configs":
        return set_bool_argument("get_default_configs", False, **kwargs)
    elif argument == "use_last_data":
        return set_bool_argument("use_last_data", False, **kwargs)
    else:
        return f"{currentFuncName()}: Bad 'variable' argument: {argument}"


def get_metadata():
    """
    Function to read metadata
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.get_metadata
    :return:
    """

    module = __name__.split('.', 1)
    module = ["py"]

    try:
        pkg = pkg_resources.get_distribution(module[0])
    except pkg_resources.RequirementParseError:
        # if called from CLI, try to get pkg from the path
        distros = list(pkg_resources.find_distributions(cfg.BASE_DIR, only=True))
        if len(distros) == 1:
            pkg = distros[0]
        else:
            pkg = pkg_resources.find_distributions(cfg.BASE_DIR, only=True)
    except Exception as e:
        raise HTTPBadRequest(reason=e)

    # One can include arguments for prepare_datasets() in the metadata
    prepare_datasets_args = _fields_to_dict(get_prepare_datasets_args())
    # make 'type' JSON serializable
    for key, val in prepare_datasets_args.items():
        prepare_datasets_args[key]['type'] = str(val['type'])

    # One can include arguments for train() in the metadata
    train_args = _fields_to_dict(get_train_args())
    # make 'type' JSON serializable
    for key, val in train_args.items():
        train_args[key]['type'] = str(val['type'])

    # One can include arguments for predict() in the metadata
    predict_args = _fields_to_dict(get_predict_args())
    # make 'type' JSON serializable
    for key, val in predict_args.items():
        predict_args[key]['type'] = str(val['type'])

    meta = {
        'name': None,
        'version': None,
        'summary': None,
        'home-page': None,
        'author': None,
        'author-email': None,
        'license': None,
        'help-prepare-datasets' : prepare_datasets_args,
        'help-train' : train_args,
        'help-predict' : predict_args
    }

    for line in pkg.get_metadata_lines("PKG-INFO"):
        line_low = line.lower()   # to avoid inconsistency due to letter cases
        for par in meta:
            if line_low.startswith(par.lower() + ":"):
                _, value = line.split(": ", 1)
                meta[par] = value

    return meta


def warm():
    """
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.warm
    :return:
    """
    # e.g. prepare the data


def get_prepare_datasets_args():
    return cfg.PrepareDatasetsArgsSchema().fields


@_catch_error
def prepare_datasets(**kwargs):
    message = ""
    # use the schema
    schema = cfg.PrepareDatasetsArgsSchema()
    # deserialize key-word arguments
    prepare_datasets_args = schema.load(kwargs)
    print_log(f"prepare_datasets_args['data_input_source'] == {prepare_datasets_args['data_input_source']}")
    # data_input_source
    data_input_source = set_kwargs("data_input_source", **kwargs)
    # what to do
    type_data_process = set_kwargs("type_data_process", **kwargs)

    # delete old files in working_dir (only with .csv extension)
    mutils.delete_old_files(cfg.RAW_DATA_DIR, ".csv")
    mutils.delete_old_files(cfg.TRAIN_DIR, ".csv")
    mutils.delete_old_files(cfg.TEST_DIR, ".csv")
    mutils.delete_old_files(cfg.VALIDATION_DIR, ".csv")

    # load new data - nexcloud, own source, local (default)
    source_path = data_input_source
    dest_path = cfg.RAW_DATA_DIR
    if type_data_process == "train":
        dest_path_train_file = cfg.TRAIN_FILE
        dest_path_test_file = cfg.TEST_FILE
        dest_path_validate_file = cfg.VALIDATION_FILE
        config_yaml = bf.load_config_yaml(cfg.CONFIG_YAML_DATA_PATH("train"))
        print_log(f"prepare_data_train({source_path}, {dest_path}, {dest_path_train_file}, {dest_path_test_file}, {dest_path_validate_file}, {config_yaml})")
        bf.prepare_data_train(source_path, dest_path, dest_path_train_file, dest_path_test_file, dest_path_validate_file, config_yaml)
    elif type_data_process == "test":
        dest_path_test_file = cfg.TEST_FILE
        config_yaml = bf.load_config_yaml(cfg.CONFIG_YAML_DATA_PATH("test"))
        print_log(f"prepare_data_test({source_path}, {dest_path}, {dest_path_test_file}, {config_yaml})")
        bf.prepare_data_test(source_path, dest_path, dest_path_test_file, config_yaml)
    elif type_data_process == "predict":
        dest_path_predict_file = cfg.PREDICT_FILE
        config_yaml = bf.load_config_yaml(cfg.CONFIG_YAML_DATA_PATH("predict"))
        print_log(f"prepare_data_predict({source_path}, {dest_path}, {dest_path_predict_file}, {config_yaml})")
        bf.prepare_data_predict(source_path, dest_path, dest_path_test_file, config_yaml)
    else:
        message = f"{currentFuncName()}: Bad value for type_data_process == {type_data_process}. Use one of 'train', 'test', 'predict'"

    return message


def try_copy(src, dest):
    try:
        print_log(f"{currentFuncName()}: shutil.copy({src}, {dest})")
        shutil.copy(src, dest)
    except Exception as e:
        print_log(f"{currentFuncName()}: Error in copy file from {src} to {dest}. Exception: {e}")
        return e


def copy_directories(src, dest):
    if not os.path.isdir(src):
        print_log(f"{currentFuncName()}: Source {src} is not a directory")
        return 0
    if not os.path.isdir(dest):
        print_log(f"{currentFuncName()}: Dest ${dest} is not a directory")
        return 0
    for f in os.listdir(src):
        if os.path.isdir(os.path.join(src, f)):
            print_log(f"{currentFuncName()}: shutil.copytree(os.path.join({src}, {f}), os.path.join({dest}, {f}), dirs_exist_ok=True)")
            shutil.copytree(os.path.join(src, f), os.path.join(dest, f), dirs_exist_ok=True)
        else:
            print_log(f"{currentFuncName()}: Item {f} is not a directory")


def copy_recursively(src, dest):
    if not os.path.isdir(src):
        return 0
    else:
        if not os.path.exists(dest) or not os.path.isdir(dest):
            print_log(f"{currentFuncName()}: os.mkdir({dest})")
            os.mkdir(dest)

    for f in os.listdir(src):
        if os.path.isfile(os.path.join(src, f)):
            print_log(f"{currentFuncName()}: shutil.copy(os.path.join({src}, {f}), os.path.join({dest}, {f}))")
            shutil.copy(os.path.join(src, f), os.path.join(dest, f))
        elif os.path.isdir(os.path.join(src, f)):
            print_log(f"{currentFuncName()}: copy_recursively(os.path.join({src}, {f}), os.path.join({dest}, {f}))")
            copy_recursively(os.path.join(src, f), os.path.join(dest, f))
        else:
            print_log(f"{currentFuncName()}: src == {src}, dest == {dest}, f == {f}")


def get_predict_args():
    """
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.get_predict_args
    :return:
    """
    return cfg.PredictArgsSchema().fields


@_catch_error
def predict(**kwargs):
    """
    Function to execute prediction
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.predict
    :param kwargs:
    :return:
    """

    def _before_return():
        # delete temp file
        if conf_nn_path != cfg.CONFIG_YAML_NN_PATH and os.path.isfile(conf_nn_path):
            print_log(f"os.remove({conf_nn_path})")
            os.remove(conf_nn_path)
        if conf_data_path != cfg.CONFIG_YAML_DATA_PATH("predict") and os.path.isfile(conf_data_path):
            print_log(f"os.remove({conf_data_path})")
            os.remove(conf_data_path)
        if model_hdf5_path != cfg.MODEL_FILE_PATH and os.path.isfile(model_hdf5_path):
            print_log(f"os.remove({model_hdf5_path})")
            os.remove(model_hdf5_path)
        if data_input_targz != cfg.DEFAULT_DATA_TARGZ and os.path.isfile(data_input_targz):
            print_log(f"os.remove({data_input_targz})")
            os.remove(data_input_targz)
        # move log file
        shutil.move(cfg.LOG_FILE_PATH, output_dir_name + "/log_file.txt")

    def _make_zipfile(source_dir, output_filename):
        shutil.make_archive(output_filename, 'zip', source_dir)

    def _on_return(**kwargs):
        # make tar.gz file
        print_log(f"_make_tarfile({output_dir_name}, {output_dir_name}.zip)", log_file=None)
        _make_zipfile(output_dir_name, output_dir_name)
        print_log("OK", log_file=None)
        # send to nextcloud or on gui
        if path_out != "":
            if kwargs["accept"] == "application/json":
                return open(output_dir_name + ".zip", 'rb', buffering=0)
            elif kwargs["accept"] == "application/zip":
                return open(output_dir_name + ".zip", 'rb', buffering=0)
        else:
            if kwargs["accept"] == "application/json":
                return open(output_dir_name + ".zip", 'rb', buffering=0)
            elif kwargs["accept"] == "application/zip":
                return open(output_dir_name + ".zip", 'rb', buffering=0)

    # if (not any([kwargs['urls'], kwargs['files']]) or
    #         all([kwargs['urls'], kwargs['files']])):
    #     raise Exception("You must provide either 'url' or 'data' in the payload")

    # prepare log file
    f = open(cfg.LOG_FILE_PATH, "w")
    f.close()

    # get input values - default values should be preset
    conf_nn_path = set_kwargs("conf_nn", **kwargs)
    conf_data_path = set_kwargs("conf_data", "predict", **kwargs)  # cfg.CONFIG_YAML_DATA_PATH("predict")
    model_hdf5_path = set_kwargs("model_hdf5", "predict", **kwargs)  # cfg.MODEL_FILE_PATH
    data_input_targz = set_kwargs("data_pred", **kwargs)  # cfg.DEFAULT_DATA_TARGZ
    output_name = set_kwargs("output_name", "predict", **kwargs)  # cfg.OUTPUT_NAME("predict")
    # urls_inp = set_kwargs("urls_inp", **kwargs)  # ""
    # urls_out = set_kwargs("urls_out", **kwargs)  # ""
    path_inp = set_kwargs("path_out", **kwargs)
    path_out = set_kwargs("path_out", **kwargs)
    get_default_configs = set_kwargs("get_default_configs", **kwargs)
    use_last_data = set_kwargs("use_last_data", **kwargs)

    # print input
    print_log(f"conf_nn_path == {conf_nn_path}")
    print_log(f"conf_data_path == {conf_data_path}")
    print_log(f"model_hdf5_path == {model_hdf5_path}")
    print_log(f"data_input_targz == {data_input_targz}")
    print_log(f"output_name == {output_name}")
    print_log(f"path_inp == {path_inp}")
    print_log(f"path_out == {path_out}")
    print_log(f"get_default_configs == {get_default_configs}")
    print_log(f"use_last_data == {use_last_data}")

    # clean directories
    if os.path.isdir(cfg.NEXTCLOUD_INPUTS):
        print_log(f"shutil.rmtree({cfg.NEXTCLOUD_INPUTS})")
        shutil.rmtree(cfg.NEXTCLOUD_INPUTS)
    else:
        print_log(f"{cfg.NEXTCLOUD_INPUTS} doesn't exist or it isn't a directory")

    if os.path.isdir(cfg.GUI_INPUTS):
        print_log(f"shutil.rmtree({cfg.GUI_INPUTS})")
        shutil.rmtree(cfg.GUI_INPUTS)
    else:
        print_log(f"{cfg.GUI_INPUTS} doesn't exist or it isn't a directory")

    if use_last_data == False:
        print_log(f"mutils.delete_old_files({cfg.WORKING_DATA_DIR}, .csv)")
        mutils.delete_old_files(cfg.WORKING_DATA_DIR, ".csv")
        print_log(f"mutils.delete_old_files({cfg.RAW_DATA_DIR}, .csv)")
        mutils.delete_old_files(cfg.RAW_DATA_DIR, ".csv")
        print_log(f"mutils.delete_old_files({cfg.TRAIN_DIR}, .csv)")
        mutils.delete_old_files(cfg.TRAIN_DIR, ".csv")
        print_log(f"mutils.delete_old_files({cfg.TEST_DIR}, .csv)")
        mutils.delete_old_files(cfg.TEST_DIR, ".csv")
        print_log(f"mutils.delete_old_files({cfg.VALIDATION_DIR}, .csv)")
        mutils.delete_old_files(cfg.VALIDATION_DIR, ".csv")

    # makedir output_name and deleted directories
    output_dir_name = os.path.join(cfg.WORK_SAVE_DIR, output_name)
    print_log(f"os.mkdir({output_dir_name})")
    os.mkdir(output_dir_name)
    print_log(f"os.mkdir({cfg.NEXTCLOUD_INPUTS})")
    os.mkdir(cfg.NEXTCLOUD_INPUTS)
    print_log(f"os.mkdir({cfg.GUI_INPUTS})")
    os.mkdir(cfg.GUI_INPUTS)

    # return default config files
    if get_default_configs == True:
        print_log(f"shutil.copy({cfg.CONFIG_YAML_NN_PATH}, {output_dir_name})")
        shutil.copy(cfg.CONFIG_YAML_NN_PATH, output_dir_name)
        print_log(f"shutil.copy({cfg.CONFIG_YAML_DATA_PATH('train')}, {output_dir_name})")
        shutil.copy(cfg.CONFIG_YAML_DATA_PATH("train"), output_dir_name)
        print_log(f"shutil.copy({cfg.CONFIG_YAML_DATA_PATH('predict')}, {output_dir_name})")
        shutil.copy(cfg.CONFIG_YAML_DATA_PATH("predict"), output_dir_name)
        print_log(f"shutil.copy({cfg.MODEL_FILE_PATH}, {output_dir_name})")
        shutil.copy(cfg.MODEL_FILE_PATH, output_dir_name)
        _before_return()
        return _on_return(**kwargs)

    # copy input from GUI
    print_log(f"shutil.copy({conf_nn_path}, os.path.join({cfg.GUI_INPUTS}, {cfg.CONFIG_YAML_NN_FILE_NAME}))")
    shutil.copy(conf_nn_path, os.path.join(cfg.GUI_INPUTS, cfg.CONFIG_YAML_NN_FILE_NAME))
    print_log(f"shutil.copy({conf_data_path}, os.path.join({cfg.GUI_INPUTS}, {cfg.CONFIG_YAML_DATA_FILE_NAME('predict')}))")
    shutil.copy(conf_data_path, os.path.join(cfg.GUI_INPUTS, cfg.CONFIG_YAML_DATA_FILE_NAME("predict")))
    print_log(f"shutil.copy({model_hdf5_path}, os.path.join({cfg.GUI_INPUTS}, {cfg.MODEL_FILE_NAME}))")
    shutil.copy(model_hdf5_path, os.path.join(cfg.GUI_INPUTS, cfg.MODEL_FILE_NAME))
    print_log(f"shutil.copy({data_input_targz}, os.path.join({cfg.GUI_INPUTS}, {cfg.DEFAULT_DATA_TARGZ_FILENAME}))")
    shutil.copy(data_input_targz, os.path.join(cfg.GUI_INPUTS, cfg.DEFAULT_DATA_TARGZ_FILENAME))
    filename = os.path.basename(data_input_targz)
    tar = tarfile.open(os.path.join(cfg.GUI_INPUTS, filename))
    tar.extractall(cfg.GUI_INPUTS)
    tar.close()
    print_log(f"os.path.join({cfg.GUI_INPUTS}, {filename})")
    os.remove(os.path.join(cfg.GUI_INPUTS, filename))

    # copy file from path_inp to cfg.NEXTCLOUD_INPUTS, untar/unzip
    if path_inp != "":
        filename = os.path.basename(path_inp)
        # mutils.rclone_directory(path_inp, cfg.NEXTCLOUD_INPUTS)
        shutil.copyfile(path_inp, cfg.NEXTCLOUD_INPUTS)
        tar = tarfile.open(os.path.join(cfg.NEXTCLOUD_INPUTS, filename))
        tar.extractall(cfg.NEXTCLOUD_INPUTS)
        tar.close()
        print_log(f"os.remove(os.path.join({cfg.NEXTCLOUD_INPUTS}, {filename}))")
        os.remove(os.path.join(cfg.NEXTCLOUD_INPUTS, filename))

    # copy data to output directory, default -> gui -> nextcloud
    # filename must have same name as default!!!
    # default
    print_log(f"Copy default configs to output directory")
    try_copy(cfg.CONFIG_YAML_NN_PATH, output_dir_name)
    try_copy(cfg.CONFIG_YAML_DATA_PATH("predict"), output_dir_name)
    try_copy(cfg.MODEL_FILE_PATH, output_dir_name)
    print_log(f"copy_recursively(cfg.SERVER_DATA_DIR, cfg.RAW_DATA_DIR)")  # , dirs_exist_ok=True")
    copy_recursively(cfg.SERVER_DATA_DIR, cfg.RAW_DATA_DIR)  # , dirs_exist_ok=True)
    # gui
    print_log(f"Copy GUI configs to output directory")
    try_copy(conf_nn_path, output_dir_name)
    try_copy(conf_data_path, output_dir_name)
    try_copy(model_hdf5_path, output_dir_name)
    print_log(f"copy_recursively({cfg.GUI_INPUTS}, {cfg.RAW_DATA_DIR})")  # , dirs_exist_ok=True)")
    copy_recursively(cfg.GUI_INPUTS, cfg.RAW_DATA_DIR)  # , dirs_exist_ok=True)
    # nextcloud
    print_log(f"Copy nextcloud configs to output directory")
    try_copy(os.path.join(cfg.NEXTCLOUD_INPUTS, cfg.CONFIG_YAML_NN_FILE_NAME), output_dir_name)
    try_copy(os.path.join(cfg.NEXTCLOUD_INPUTS, cfg.CONFIG_YAML_DATA_FILE_NAME("predict")), output_dir_name)
    try_copy(os.path.join(cfg.NEXTCLOUD_INPUTS, cfg.MODEL_FILE_NAME), output_dir_name)
    print_log(f"copy_recursively({cfg.NEXTCLOUD_INPUTS}, {cfg.RAW_DATA_DIR})")  # , dirs_exist_ok=True)")
    copy_recursively(cfg.NEXTCLOUD_INPUTS, cfg.RAW_DATA_DIR)  # , dirs_exist_ok=True)

    # load yaml
    print_log(f"config_yaml = bf.load_config_yaml(os.path.join({output_dir_name}, {cfg.CONFIG_YAML_DATA_FILE_NAME('predict')}))")
    config_yaml = bf.load_config_yaml(os.path.join(output_dir_name, cfg.CONFIG_YAML_DATA_FILE_NAME("predict")))
    print_log(f"config_yaml = bf.load_config_yaml(os.path.join({output_dir_name}, {cfg.CONFIG_YAML_NN_FILE_NAME('predict')}))")
    config_nn_yaml = bf.load_config_yaml(os.path.join(output_dir_name, cfg.CONFIG_YAML_NN_FILE_NAME("predict")))

    # make dataset
    if use_last_data == False:
        print_log(f"prepare_data_predict({cfg.RAW_DATA_DIR}, {cfg.WORKING_DATA_DIR}, {cfg.PREDICT_FILE}, {config_yaml})")
        bf.prepare_data_predict(cfg.RAW_DATA_DIR, cfg.WORKING_DATA_DIR, cfg.PREDICT_FILE, config_yaml)

    print_log(f"dataPredictX, dataPredictY = mutils.make_dataset({cfg.PREDICT_FILE}, config_yaml)")
    dataPredictX, dataPredictY = mutils.make_dataset(cfg.PREDICT_FILE, config_yaml)

    # load model
    print_log(f"Load model")
    modelLoad = mutils.load_model(os.path.join(output_dir_name, cfg.MODEL_FILE_NAME), config_nn_yaml)

    # copy data to output
    print_log(f"Copy data to output")
    if os.path.exists(cfg.PREDICT_FILE):
        print_log(f"shutil.copy({cfg.PREDICT_FILE}, {output_dir_name})")
        shutil.copy(cfg.PREDICT_FILE, output_dir_name)

    # make prediction
    print_log(f"Make prediction")
    prediction = mutils.test_model(modelLoad, dataPredictX)
   
    message = { "status": "ok",
                "training": []}

    # save prediction to output text file

    # return output
    _before_return()
    _on_return(kwargs)
    return message


def _predict_data(*args):
    """
    (Optional) Helper function to make prediction on an uploaded file
    """
    # message = 'Not implemented (predict_data())'
    # message = {"Error": message}
    files = []
    file_inputs = ['conf_nn', 'conf_data', 'model_hdf5', 'data_pred']
    for arg in args:
        for f_i in file_inputs:
            file_objs = arg[f_i]
            for f in file_objs:
                files.append({f_i: f.filename})

    return files # message


def _predict_url(*args):
    """
    (Optional) Helper function to make prediction on an URL
    """
    message = 'Not implemented (predict_url())'
    message = {"Error": message}
    return message


def get_train_args():
    """
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.get_train_args
    :param kwargs:
    :return:
    """
    return cfg.TrainArgsSchema().fields


# @flaat.login_required() line is to limit access for only authorized people
# Comment this line, if you open training for everybody
# More info: see https://github.com/indigo-dc/flaat
# @flaat.login_required()  # Allows only authorized people to train
def train(**kwargs):
    """
    Train network
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.train
    :param kwargs:
    :return:
    """

    def _before_return():
        # delete temp file
        if conf_nn_path != cfg.CONFIG_YAML_NN_PATH and os.path.isfile(conf_nn_path):
            print_log(f"os.remove({conf_nn_path})")
            os.remove(conf_nn_path)
        if conf_data_path != cfg.CONFIG_YAML_DATA_PATH("train") and os.path.isfile(conf_data_path):
            print_log(f"os.remove({conf_data_path})")
            os.remove(conf_data_path)
        if data_input_targz != cfg.DEFAULT_DATA_TARGZ and os.path.isfile(data_input_targz):
            print_log(f"os.remove({data_input_targz})")
            os.remove(data_input_targz)
        # move log file
        shutil.move(cfg.LOG_FILE_PATH, output_dir_name + "/log_file.txt")

    def _make_zipfile(source_dir, output_filename):
        shutil.make_archive(output_filename, 'zip', source_dir)

    def _on_return(**kwargs):
        # make tar.gz file
        print_log(f"_make_tarfile({output_dir_name}, {output_dir_name}.zip)", log_file=None)
        _make_zipfile(output_dir_name, output_dir_name)
        print_log("OK", log_file=None)
        # send to nextcloud or on gui
        if path_out != "":
            if kwargs["accept"] == "application/json":
                return open(output_dir_name + ".zip", 'rb', buffering=0)
            elif kwargs["accept"] == "application/zip":
                return open(output_dir_name + ".zip", 'rb', buffering=0)
        else:
            if kwargs["accept"] == "application/json":
                return open(output_dir_name + ".zip", 'rb', buffering=0)
            elif kwargs["accept"] == "application/zip":
                return open(output_dir_name + ".zip", 'rb', buffering=0)

    # if (not any([kwargs['urls'], kwargs['files']]) or
    #         all([kwargs['urls'], kwargs['files']])):
    #     raise Exception("You must provide either 'url' or 'data' in the payload")

    # prepare log file
    f = open(cfg.LOG_FILE_PATH, "w")
    f.close()

    # get input values - default values should be preset
    conf_nn_path = set_kwargs("conf_nn", **kwargs)  # cfg.CONFIG_YAML_NN_PATH
    conf_data_path = set_kwargs("conf_data", "train", **kwargs)  # cfg.CONFIG_YAML_DATA_PATH("train")
    data_input_targz = set_kwargs("data_pred", **kwargs)  # cfg.DEFAULT_DATA_TARGZ
    output_name = set_kwargs("output_name", "train", **kwargs)  # cfg.OUTPUT_NAME("train")
    # urls_inp = set_kwargs("urls_inp", **kwargs)  # ""
    # urls_out = set_kwargs("urls_out", **kwargs)  # ""
    path_inp = set_kwargs("path_inp", **kwargs)  # ""
    path_out = set_kwargs("path_out", **kwargs)  # ""
    use_last_data = set_kwargs("use_last_data", **kwargs)

    # print input
    print_log(f"conf_nn_path == {conf_nn_path}")
    print_log(f"conf_data_path == {conf_data_path}")
    print_log(f"data_input_targz == {data_input_targz}")
    print_log(f"output_name == {output_name}")
    print_log(f"path_inp == {path_inp}")
    print_log(f"path_out == {path_out}")
    print_log(f"use_last_data == {use_last_data}")

    # clean directories
    if os.path.isdir(cfg.NEXTCLOUD_INPUTS):
        print_log(f"shutil.rmtree({cfg.NEXTCLOUD_INPUTS})")
        shutil.rmtree(cfg.NEXTCLOUD_INPUTS)
    else:
        print_log(f"{cfg.NEXTCLOUD_INPUTS} doesn't exist or it isn't a directory")
    
    if os.path.isdir(cfg.GUI_INPUTS):
        print_log(f"shutil.rmtree({cfg.GUI_INPUTS})")
        shutil.rmtree(cfg.GUI_INPUTS)
    else:
        print_log(f"{cfg.GUI_INPUTS} doesn't exist or it isn't a directory")

    if use_last_data == False:
        print_log(f"mutils.delete_old_files({cfg.WORKING_DATA_DIR}, .csv)")
        mutils.delete_old_files(cfg.WORKING_DATA_DIR, ".csv")
        print_log(f"mutils.delete_old_files({cfg.RAW_DATA_DIR}, .csv)")
        mutils.delete_old_files(cfg.RAW_DATA_DIR, ".csv")
        print_log(f"mutils.delete_old_files({cfg.TRAIN_DIR}, .csv)")
        mutils.delete_old_files(cfg.TRAIN_DIR, ".csv")
        print_log(f"mutils.delete_old_files({cfg.TEST_DIR}, .csv)")
        mutils.delete_old_files(cfg.TEST_DIR, ".csv")
        print_log(f"mutils.delete_old_files({cfg.VALIDATION_DIR}, .csv)")
        mutils.delete_old_files(cfg.VALIDATION_DIR, ".csv")

    # makedir output_name and deleted directories
    output_dir_name = os.path.join(cfg.WORK_SAVE_DIR, output_name)
    print_log(f"os.mkdir({output_dir_name})")
    os.mkdir(output_dir_name)
    print_log(f"os.mkdir({cfg.NEXTCLOUD_INPUTS})")
    os.mkdir(cfg.NEXTCLOUD_INPUTS)
    print_log(f"os.mkdir({cfg.GUI_INPUTS})")
    os.mkdir(cfg.GUI_INPUTS)

    # copy input from GUI
    print_log(f"shutil.copy({conf_nn_path}, os.path.join({cfg.GUI_INPUTS}, {cfg.CONFIG_YAML_NN_FILE_NAME}))")
    shutil.copy(conf_nn_path, os.path.join(cfg.GUI_INPUTS, cfg.CONFIG_YAML_NN_FILE_NAME))
    print_log(f"shutil.copy({conf_data_path}, os.path.join({cfg.GUI_INPUTS}, {cfg.CONFIG_YAML_DATA_FILE_NAME('train')}))")
    shutil.copy(conf_data_path, os.path.join(cfg.GUI_INPUTS, cfg.CONFIG_YAML_DATA_FILE_NAME("train")))
    print_log(f"shutil.copy({data_input_targz}, os.path.join({cfg.GUI_INPUTS}, {cfg.DEFAULT_DATA_TARGZ_FILENAME}))")
    shutil.copy(data_input_targz, os.path.join(cfg.GUI_INPUTS, cfg.DEFAULT_DATA_TARGZ_FILENAME))
    filename = os.path.basename(data_input_targz)
    tar = tarfile.open(os.path.join(cfg.GUI_INPUTS, filename))
    tar.extractall(cfg.GUI_INPUTS)
    tar.close()
    print_log(f"os.path.join({cfg.GUI_INPUTS}, {filename})")
    os.remove(os.path.join(cfg.GUI_INPUTS, filename))

    # copy file from path_inp to cfg.NEXTCLOUD_INPUTS, untar/unzip
    if path_inp != "":
        filename = os.path.basename(path_inp)
        # mutils.rclone_directory(path_inp, cfg.NEXTCLOUD_INPUTS)
        shutil.copyfile(path_inp, cfg.NEXTCLOUD_INPUTS)
        tar = tarfile.open(os.path.join(cfg.NEXTCLOUD_INPUTS, filename))
        tar.extractall(cfg.NEXTCLOUD_INPUTS)
        tar.close()
        print_log(f"os.remove(os.path.join({cfg.NEXTCLOUD_INPUTS}, {filename}))")
        os.remove(os.path.join(cfg.NEXTCLOUD_INPUTS, filename))

    # copy data to output directory, default -> gui -> nextcloud
    # filename must have same name as default!!!
    # default
    print_log(f"Copy default configs to output directory")
    try_copy(cfg.CONFIG_YAML_NN_PATH, output_dir_name)
    try_copy(cfg.CONFIG_YAML_DATA_PATH("train"), output_dir_name)
    print_log(f"copy_recursively({cfg.SERVER_DATA_DIR}, {cfg.RAW_DATA_DIR})")  # , dirs_exist_ok=True)")
    copy_recursively(cfg.SERVER_DATA_DIR, cfg.RAW_DATA_DIR)  # , dirs_exist_ok=True)
    # gui
    print_log(f"Copy GUI configs to output directory")
    try_copy(conf_nn_path, output_dir_name)
    try_copy(conf_data_path, output_dir_name)
    print_log(f"copy_recursively({cfg.GUI_INPUTS}, {cfg.RAW_DATA_DIR})")  # , dirs_exist_ok=True)")
    copy_recursively(cfg.GUI_INPUTS, cfg.RAW_DATA_DIR)  # , dirs_exist_ok=True)
    # nextcloud
    print_log(f"Copy nextcloud configs to output directory")
    try_copy(os.path.join(cfg.NEXTCLOUD_INPUTS, cfg.CONFIG_YAML_NN_FILE_NAME), output_dir_name)
    try_copy(os.path.join(cfg.NEXTCLOUD_INPUTS, cfg.CONFIG_YAML_DATA_FILE_NAME("train")), output_dir_name)
    print_log(f"copy_recursively({cfg.NEXTCLOUD_INPUTS}, {cfg.RAW_DATA_DIR})")  # , dirs_exist_ok=True)")
    copy_recursively(cfg.NEXTCLOUD_INPUTS, cfg.RAW_DATA_DIR)  # , dirs_exist_ok=True)

    print_log(f"config_yaml = bf.load_config_yaml(os.path.join({output_dir_name}, {cfg.CONFIG_YAML_DATA_FILE_NAME('train')}))")
    config_yaml = bf.load_config_yaml(os.path.join(output_dir_name, cfg.CONFIG_YAML_DATA_FILE_NAME("train")))

    # make dataset
    if use_last_data == False:
        print_log(f"prepare_data_train({cfg.RAW_DATA_DIR}, {cfg.WORKING_DATA_DIR}, {cfg.TRAIN_FILE}, {cfg.TEST_FILE}, {cfg.VALIDATION_FILE}, {config_yaml})")
        bf.prepare_data_train(cfg.RAW_DATA_DIR, cfg.WORKING_DATA_DIR, cfg.TRAIN_FILE, cfg.TEST_FILE, cfg.VALIDATION_FILE, config_yaml)

    print_log(f"dataTrainX, dataTrainY = mutils.make_dataset({cfg.TRAIN_FILE}, config_yaml)")
    dataTrainX, dataTrainY = mutils.make_dataset(cfg.TRAIN_FILE, config_yaml)

    # copy data to output
    print_log(f"Copy data to output")
    if os.path.exists(cfg.TRAIN_FILE):
        print_log(f"shutil.copy({cfg.TRAIN_FILE}, {output_dir_name})")
        shutil.copy(cfg.TRAIN_FILE, output_dir_name)
    if os.path.exists(cfg.TEST_FILE):
        print_log(f"shutil.copy({cfg.TEST_FILE}, {output_dir_name})")
        shutil.copy(cfg.TEST_FILE, output_dir_name)
    if os.path.exists(cfg.VALIDATION_FILE):
        print_log(f"shutil.copy({cfg.VALIDATION_FILE}, {output_dir_name})")
        shutil.copy(cfg.VALIDATION_FILE, output_dir_name)

    # model training
    print_log(f"config_nn_yaml = bf.load_config_yaml(os.path.join({output_dir_name}, {cfg.CONFIG_YAML_NN_FILE_NAME}))")
    config_nn_yaml = bf.load_config_yaml(os.path.join(output_dir_name, cfg.CONFIG_YAML_NN_FILE_NAME))
    print_log(f"trainScores, trainHistories, trainModels = mutils.train_model(dataTrainX, dataTrainY, config_nn_yaml)")
    trainScores, trainHistories, trainModels = mutils.train_model(
        dataTrainX, dataTrainY, config_nn_yaml
    )

    # model saving
    print_log(f"mutils.save_model(trainModels[1], os.path.join({output_dir_name}, {cfg.MODEL_FILE_NAME}))")
    mutils.save_model(trainModels[1], os.path.join(output_dir_name, cfg.MODEL_FILE_NAME))
   
    message = { "status": "ok",
                "training": []}

    # return output
    _before_return()
    _on_return()
    return message


# during development it might be practical
# to check your code from CLI (command line interface)
def main():
    """
    Runs above-described methods from CLI
    (see below an example)
    """

    if args.method == 'get_metadata':
        meta = get_metadata()
        print(json.dumps(meta))
        return meta
    elif args.method == 'prepare_datasets':
        results = prepare_datasets(**vars(args))
        print(json.dumps(results))
        return results
    elif args.method == 'validate':
        results = predict(**vars(args))
        print(json.dumps(results))
        return results
    elif args.method == 'predict':
        # [!] you may need to take special care in the case of args.files [!]
        results = predict(**vars(args))
        print(json.dumps(results))
        return results
    elif args.method == 'train':
        results = train(**vars(args))
        print(json.dumps(results))
        return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model parameters',
                                     add_help=False)

    cmd_parser = argparse.ArgumentParser()
    subparsers = cmd_parser.add_subparsers(
                            help='methods. Use \"api.py method --help\" to get more info', 
                            dest='method')
    
    # -------------------------------------------------------------------------------------
    # configure parser to call get_metadata()
    get_metadata_parser = subparsers.add_parser('get_metadata',
                                         help='get_metadata method',
                                         parents=[parser])
    # normally there are no arguments to configure for get_metadata()
    
    # -------------------------------------------------------------------------------------
    # configure arguments for prepare_datasets()
    prepare_datasets_parser = subparsers.add_parser('prepare_datasets', 
                                         help='prepare_datasets method',
                                         parents=[parser])
    # one should convert get_prepare_datasets_args() to add them in prepare_datasets_parser
    # For example:
    prepare_datasets_args = _fields_to_dict(get_prepare_datasets_args())
    for key, val in prepare_datasets_args.items():
        prepare_datasets_parser.add_argument('--%s' % key,
                                        default=val['default'],
                                        type=val['type'],
                                        help=val['help'],
                                        required=val['required'])

    # -------------------------------------------------------------------------------------
    # configure arguments for predict()
    predict_parser = subparsers.add_parser('predict',
                                           help='commands for prediction',
                                           parents=[parser])
    # one should convert get_predict_args() to add them in predict_parser
    # For example:
    predict_args = _fields_to_dict(get_predict_args())
    for key, val in predict_args.items():
        predict_parser.add_argument('--%s' % key,
                               default=val['default'],
                               type=val['type'],
                               help=val['help'],
                               required=val['required'])
    
    # -------------------------------------------------------------------------------------
    ## configure arguments for train()
    train_parser = subparsers.add_parser('train',
                                         help='commands for training',
                                         parents=[parser])
    # one should convert get_train_args() to add them in train_parser
    # For example:
    train_args = _fields_to_dict(get_train_args())
    for key, val in train_args.items():
        train_parser.add_argument('--%s' % key,
                                  default=val['default'],
                                  type=val['type'],
                                  help=val['help'],
                                  required=val['required'])

    args = cmd_parser.parse_args()
    
    main()
