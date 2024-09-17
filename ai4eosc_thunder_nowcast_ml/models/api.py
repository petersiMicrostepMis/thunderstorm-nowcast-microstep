# -*- coding: utf-8 -*-
"""
Integrate a model with the DEEP API
"""

import json
import argparse
import pkg_resources
import tarfile
import datetime
import mlflow
import numpy as np

import os
import sys
import shutil
import pandas as pd
# import project's config.py
from .. import config as cfg
from .. import config_layout as cly
from ..features import build_features as bf
from ..models import model_utils as mutils
from ..models import statistics as stat
from keras.utils import to_categorical

from aiohttp.web import HTTPBadRequest
from functools import wraps

# Authorization
# from flaat import Flaat
# flaat = Flaat()


def currentFuncName(n=0):
    return sys._getframe(n + 1).f_code.co_name


def print_log(log_line, verbose=True, time_stamp=True, log_file=cly.LOG_FILE_PATH):
    tm = ""
    if time_stamp:
        tm = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S: ")
    if verbose:
        if log_file is None:
            print(tm + log_line)
        else:
            print(tm + log_line)
            print(f"os.makedirs(os.path.dirname({log_file}), exist_ok=True)")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            try:
                with open(log_file, 'a') as file:
                    file.write(tm + log_line + "\n")
            except Exception as err:
                print(f"Missing log file: {err}")
                f = open(log_file, "w")
                f.close()


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
        except Exception:
            val_req = False
        param['required'] = val_req

        dict_out[key] = param
    return dict_out


def load_config_yaml_file(config_yaml_path):
    print_log(f"{currentFuncName()}:")
    if not os.path.isfile(config_yaml_path):
        print_log(f"{currentFuncName()}: Error: yaml config is missing, config_yaml_path == {config_yaml_path}")
    else:
        config_yaml = bf.load_config_yaml(config_yaml_path)
    return config_yaml


def set_string_argument(arg_name, arg_default_value, preffix="", **kwargs):
    print_log(f"{currentFuncName()}:")
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
    print_log(f"{currentFuncName()}:")
    bool_argument = arg_default_value
    try:
        bool_argument = kwargs[arg_name]
    except Exception:
        bool_argument = arg_default_value
    return bool_argument


def set_file_argument(arg_name, arg_default_value, **kwargs):
    print_log(f"{currentFuncName()}:")
    try:
        if kwargs[arg_name]:
            kwargs[arg_name] = [kwargs[arg_name]]
            for fl in kwargs[arg_name]:
                filename = fl.filename
            print_log(f"{currentFuncName()}: filesize {os.path.getsize(filename)}")
            if os.path.getsize(filename) <= 2:
                filename = arg_default_value
                print_log(f"{currentFuncName()}: Info: Set default value for {arg_name}: {arg_default_value}, \
                          due to loaded file has zero size")
        else:
            print_log(f"{currentFuncName()}: Info: Set default value for {arg_name}: {arg_default_value}")
            filename = arg_default_value
    except Exception:
        filename = arg_default_value
    print_log(f"{currentFuncName()}: filename == {filename}")
    return filename


def set_kwargs(argument, arg2=None, **kwargs):
    print_log(f"{currentFuncName()}:")
    # predict
    if argument == "select_option_pr":
        return set_string_argument("select_option_pr", "", **kwargs)
    elif argument == "select_dtm_pr":
        return set_string_argument("select_dtm_pr", "", **kwargs)
    elif argument == "select_mlo_pr":
        return set_string_argument("select_mlo_pr", "", **kwargs)
    elif argument == "select_nnw_pr":
        return set_string_argument("select_nnw_pr", "", **kwargs)
    elif argument == "select_ino_pr":
        return set_string_argument("select_ino_pr", "", **kwargs)
    elif argument == "select_usr_pr":
        return set_string_argument("select_usr_pr", "", **kwargs)
    elif argument == "cfg_file_pr":
        return set_file_argument("cfg_file_pr", "", **kwargs)
    # train
    elif argument == "select_dtm_tr":
        return set_string_argument("select_dtm_tr", "", **kwargs)
    elif argument == "select_mlo_tr":
        return set_string_argument("select_mlo_tr", "", **kwargs)
    elif argument == "select_nnw_tr":
        return set_string_argument("select_nnw_tr", "", **kwargs)
    elif argument == "select_ino_tr":
        return set_string_argument("select_ino_tr", "", **kwargs)
    elif argument == "select_usr_tr":
        return set_string_argument("select_usr_tr", "", **kwargs)
    else:
        return f"{currentFuncName()}: Bad 'variable' argument: {argument}"


def get_metadata():
    """
    Function to read metadata
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.get_metadata
    :return:
    """

    module = __name__.split('.', 1)
    # module = ["py"]

    # prepare log file
    print(f"os.makedirs(os.path.dirname({cly.LOG_FILE_PATH}), exist_ok=True)")
    os.makedirs(os.path.dirname(cly.LOG_FILE_PATH), exist_ok=True)
    f = open(cly.LOG_FILE_PATH, "w")
    f.close()
    print_log(f"{currentFuncName()}:")

    try:
        pkg = pkg_resources.get_distribution(module[0])
    except pkg_resources.RequirementParseError:
        # if called from CLI, try to get pkg from the path
        distros = list(pkg_resources.find_distributions(cly.BASE_DIR, only=True))
        if len(distros) == 1:
            pkg = distros[0]
        else:
            pkg = pkg_resources.find_distributions(cly.BASE_DIR, only=True)
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
        'help-prepare-datasets': prepare_datasets_args,
        'help-train': train_args,
        'help-predict': predict_args
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
    print_log(f"{currentFuncName()}:")


def get_prepare_datasets_args():
    print_log(f"{currentFuncName()}:")
    return cfg.PrepareDatasetsArgsSchema().fields


@_catch_error
def prepare_datasets(**kwargs):
    print_log(f"{currentFuncName()}:")
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
    mutils.delete_old_files(cly.RAW_DATA_DIR, ".csv")
    mutils.delete_old_files(cly.TRAIN_DIR, ".csv")
    mutils.delete_old_files(cly.TEST_DIR, ".csv")
    mutils.delete_old_files(cly.VALIDATION_DIR, ".csv")

    # load new data - nextcloud, own source, local (default)
    source_path = data_input_source
    dest_path = cly.RAW_DATA_DIR
    if type_data_process == "train":
        dest_path_train_file = cly.TRAIN_FILE
        dest_path_test_file = cly.TEST_FILE
        dest_path_validate_file = cly.VALIDATION_FILE
        config_yaml = bf.load_config_yaml(cfg.CONFIG_YAML_DATA_PATH("train"))
        print_log(f"bf.prepare_data_train({source_path}, {dest_path}, {dest_path_train_file}, {dest_path_test_file}, \
                  {dest_path_validate_file}, {config_yaml})")
        bf.prepare_data_train(source_path, dest_path, dest_path_train_file, dest_path_test_file,
                              dest_path_validate_file, config_yaml)
    elif type_data_process == "test":
        dest_path_test_file = cly.TEST_FILE
        config_yaml = bf.load_config_yaml(cfg.CONFIG_YAML_DATA_PATH("test"))
        print_log(f"prepare_data_test({source_path}, {dest_path}, {dest_path_test_file}, {config_yaml})")
        bf.prepare_data_test(source_path, dest_path, dest_path_test_file, config_yaml)
    elif type_data_process == "predict":
        dest_path_predict_file = cly.PREDICT_FILE
        config_yaml = bf.load_config_yaml(cfg.CONFIG_YAML_DATA_PATH("predict"))
        print_log(f"prepare_data_predict({source_path}, {dest_path}, {dest_path_predict_file}, {config_yaml})")
        bf.prepare_data_predict(source_path, dest_path, dest_path_test_file, config_yaml)
    else:
        message = f"{currentFuncName()}: Bad value for type_data_process == {type_data_process}. \
                  Use one of 'train', 'test', 'predict'"

    return message


def try_copy(src, dest):
    print_log(f"{currentFuncName()}:")
    try:
        print_log(f"{currentFuncName()}: shutil.copy({src}, {dest})")
        shutil.copy(src, dest)
    except Exception as e:
        print_log(f"{currentFuncName()}: Error in copy file from {src} to {dest}. Exception: {e}")
        return e


def copy_directories(src, dest):
    print_log(f"{currentFuncName()}:")
    if not os.path.isdir(src):
        print_log(f"{currentFuncName()}: Source {src} is not a directory")
        return 0
    if not os.path.isdir(dest):
        print_log(f"{currentFuncName()}: Dest ${dest} is not a directory")
        return 0
    for f in os.listdir(src):
        if os.path.isdir(os.path.join(src, f)):
            print_log(f"{currentFuncName()}: shutil.copytree(os.path.join({src}, {f}), os.path.join({dest}, {f}), \
                      dirs_exist_ok=True)")
            shutil.copytree(os.path.join(src, f), os.path.join(dest, f), dirs_exist_ok=True)
        else:
            print_log(f"{currentFuncName()}: Item {f} is not a directory")


def copy_recursively(src, dest):
    print_log(f"{currentFuncName()}:")
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
    print_log(f"{currentFuncName()}:")
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
        print_log("predict: _before_return")
        # move log file
        print_log(f"shutil.move({cly.LOG_FILE_PATH}, {output_dir_name}/log_file.txt)")
        shutil.move(cly.LOG_FILE_PATH, output_dir_name + "/log_file.txt")

    def _make_zipfile(source_dir, output_filename):
        shutil.make_archive(output_filename, 'zip', source_dir)

    def _on_return(**kwargs):
        print_log("predict: _on_return")
        date_suffix = ""
        if os.path.isdir(output_dir_name):
            date_suffix = "_" + datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
        # make tar.gz file
        print_log(f"_make_zipfile({output_dir_name}, {output_dir_name}{date_suffix}.zip)", log_file=None)
        _make_zipfile(output_dir_name, output_dir_name + date_suffix)
        print_log("OK", log_file=None)
        # send to nextcloud or on gui
        if ino_pr["send_outputs_to"] == "nextcloud":
            print_log(f"shutil.move({output_dir_name}.zip, {cly.NEXTCLOUD}/{ino_pr['path_out']})",
                      log_file=None)
            shutil.move(output_dir_name + ".zip", cly.NEXTCLOUD + "/" + ino_pr["path_out"])
        if ino_pr["send_outputs_to"] == "swagger" or kwargs["accept"] == "application/zip":
            print_log(f"open({output_dir_name}.zip, 'rb', buffering=0)", log_file=None)
            return open(output_dir_name + ".zip", 'rb', buffering=0)
            # if kwargs["accept"] == "application/json":
            #     return open(output_dir_name + ".zip", 'rb', buffering=0)
            # elif kwargs["accept"] == "application/zip":
            #     return open(output_dir_name + ".zip", 'rb', buffering=0)

    message = {"status": "ok",
               "training": []}

    try:
        # prepare log file
        f = open(cly.LOG_FILE_PATH, "w")
        f.close()
        print_log(f"{currentFuncName()}:")

        # get input values - default values should be preset
        option_pr = set_kwargs("select_option_pr", **kwargs)
        name_dtm_pr = set_kwargs("select_dtm_pr", **kwargs)
        name_mlo_pr = set_kwargs("select_mlo_pr", **kwargs)
        name_nnw_pr = set_kwargs("select_nnw_pr", **kwargs)
        name_ino_pr = set_kwargs("select_ino_pr", **kwargs)
        name_usr_pr = set_kwargs("select_usr_pr", **kwargs)
        new_config_file_name = set_kwargs("cfg_file_pr", **kwargs)

        config_dtm_pr_path = cfg.file_list_dtm[cfg.config_names_dtm.index(name_dtm_pr)]
        config_mlo_pr_path = cfg.file_list_mlo[cfg.config_names_mlo.index(name_mlo_pr)]
        config_nnw_pr_path = cfg.file_list_nnw[cfg.config_names_nnw.index(name_nnw_pr)]
        config_ino_pr_path = cfg.file_list_ino[cfg.config_names_ino.index(name_ino_pr)]
        config_usr_pr_path = cfg.file_list_usr[cfg.config_names_usr.index(name_usr_pr)]

        # print inputs
        print_log(f"option_pr == {option_pr}")
        print_log(f"name_dtm_pr == {name_dtm_pr}")
        print_log(f"name_mlo_pr == {name_mlo_pr}")
        print_log(f"name_nnw_pr == {name_nnw_pr}")
        print_log(f"name_ino_pr == {name_ino_pr}")
        print_log(f"name_usr_pr == {name_usr_pr}")
        print_log(f"new_config_file_name == {new_config_file_name}")
        print_log(f"config_dtm_pr_path == {config_dtm_pr_path}")
        print_log(f"config_mlo_pr_path == {config_mlo_pr_path}")
        print_log(f"config_nnw_pr_path == {config_nnw_pr_path}")
        print_log(f"config_ino_pr_path == {config_ino_pr_path}")
        print_log(f"config_usr_pr_path == {config_usr_pr_path}")

        if option_pr not in ["Prediction", "Get all config files"]:
            cfg_file_pr = set_kwargs("cfg_file_pr", **kwargs)
            if os.path.isdir(cly.NEXTCLOUD_USER_CONFIG_DIR):
                save_dir = cly.NEXTCLOUD_USER_CONFIG_DIR
            else:
                save_dir = cly.SERVER_USER_CONFIGS_DIR

            if option_pr == "Add new config to Data management configs":
                save_as = save_dir + "/" + cly.CONFIG_DATA_MANAGEMENT + "/" + cly.CONFIG_DATA_MANAGEMENT_PRFX \
                    + new_config_file_name + ".yaml"
            elif option_pr == "Add new config to MLflow outputs configs":
                save_as = save_dir + "/" + cly.CONFIG_MLFLOW_OUTPUTS + "/" + cly.CONFIG_MLFLOW_OUTPUTS_PRFX \
                    + new_config_file_name + ".yaml"
            elif option_pr == "Add new config to Neural networks configs":
                save_as = save_dir + "/" + cly.CONFIG_NEURAL_NETWORKS + "/" + cly.CONFIG_NEURAL_NETWORKS_PRFX \
                    + new_config_file_name + ".yaml"
            elif option_pr == "Add new config to input/output settings":
                save_as = save_dir + "/" + cly.CONFIG_INOUTS + "/" + cly.CONFIG_INOUTS_PRFX + new_config_file_name \
                    + ".yaml"
            elif option_pr == "Add new config to MLflow user configs":
                save_as = save_dir + "/" + cly.CONFIG_USERS + "/" + cly.CONFIG_USERS_PRFX + new_config_file_name \
                    + ".yaml"
            if os.path.isfile(save_as):
                print_log(f"Config file {save_as} exists, updating is now forbidden")
            else:
                print_log(f"shutil.move({cfg_file_pr}, {save_as})")
                shutil.move(cfg_file_pr, save_as)
            message = {}
            return message

        # load yaml config files
        dtm_pr = load_config_yaml_file(config_dtm_pr_path)
        # mlo_pr = load_config_yaml_file(config_mlo_pr_path)
        nnw_pr = load_config_yaml_file(config_nnw_pr_path)
        ino_pr = load_config_yaml_file(config_ino_pr_path)
        # usr_pr = load_config_yaml_file(config_usr_pr_path)

        # prepare output_name and deleted directories
        send_to = ""
        if ino_pr["send_outputs_to"] == "nextcloud":
            send_to = cly.NEXTCLOUD

        if ino_pr["path_out"] == "":
            save_dir = cly.WORK_SAVE_DIR
        else:
            save_dir = os.path.join(send_to, ino_pr["path_out"])

        output_dir_name = os.path.join(save_dir, ino_pr["output_name"])
        if os.path.isdir(output_dir_name) is True:
            output_dir_name = output_dir_name + datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
        print_log(f"os.makedirs({output_dir_name}, exist_ok=True)")
        os.makedirs(output_dir_name, exist_ok=True)

        # return default config files
        if option_pr == "Get all config files":
            for tp in [cfg.file_list_dtm, cfg.file_list_mlo, cfg.file_list_nnw,
                       cfg.file_list_ino, cfg.file_list_usr]:
                for fl in tp:
                    print_log(f"shutil.copy({fl}, {output_dir_name})")
                    shutil.copy(fl, output_dir_name)
            _before_return()
            return _on_return(**kwargs)

        # clean directories
        if ino_pr["use_last_data"] is False:
            print_log(f"mutils.delete_old_files({cly.WORKING_DATA_DIR}, .csv)")
            mutils.delete_old_files(cly.WORKING_DATA_DIR, ".csv")
            print_log(f"mutils.delete_old_files({cly.RAW_DATA_DIR}, .csv)")
            mutils.delete_old_files(cly.RAW_DATA_DIR, ".csv")
            print_log(f"mutils.delete_old_files({cly.TRAIN_DIR}, .csv)")
            mutils.delete_old_files(cly.TRAIN_DIR, ".csv")
            print_log(f"mutils.delete_old_files({cly.TEST_DIR}, .csv)")
            mutils.delete_old_files(cly.TEST_DIR, ".csv")
            print_log(f"mutils.delete_old_files({cly.VALIDATION_DIR}, .csv)")
            mutils.delete_old_files(cly.VALIDATION_DIR, ".csv")

        # data source
        if ino_pr["data_source"] == "server":
            data_source = ""
        elif ino_pr["data_source"] == "nextcloud":
            data_source = cly.NEXTCLOUD_DATA_DIR

        # set hdf5 file path
        if ino_pr["model_hdf5"] == "":
            model_hdf5_path = cly.DEFAULT_MODEL_HDF5_PATH
            model_hdf5_name = cly.DEFAULT_MODEL_HDF5_FILENAME
        else:
            model_hdf5_path = os.path.join(data_source, ino_pr["model_hdf5"])
            model_hdf5_name = os.path.basename(model_hdf5_path)

        # set targz path
        if ino_pr["targz_data_path"] == "":
            targz_data_path = cly.DEFAULT_DATA_TARGZ_PATH
            targz_data_name = cly.DEFAULT_DATA_TARGZ_FILENAME
        else:
            targz_data_path = os.path.join(data_source, ino_pr["targz_data_path"])
            targz_data_name = os.path.basename(targz_data_path)
        print_log(f"shutil.copy({targz_data_path}, os.path.join({cly.RAW_DATA_DIR}, {targz_data_name}))")
        shutil.copy(targz_data_path, os.path.join(cly.RAW_DATA_DIR, targz_data_name))

        if ino_pr["prediction_outfilename"] == "":
            prediction_outfilename = cly.PREDICTION_OUTFILENAME
        else:
            prediction_outfilename = ino_pr["prediction_outfilename"]

        print_log(f"tar = tarfile.open(os.path.join({cly.RAW_DATA_DIR}, {targz_data_name}), mode='r:gz')")
        tar = tarfile.open(os.path.join(cly.RAW_DATA_DIR, targz_data_name), mode='r:gz')
        for member in tar.getmembers():
            print_log(f"tar.extract(member, {cly.RAW_DATA_DIR})")
            tar.extract(member, cly.RAW_DATA_DIR)
        print_log("tar.close()")
        tar.close()

        print_log(f"os.path.join({cly.RAW_DATA_DIR}, {targz_data_name})")
        os.remove(os.path.join(cly.RAW_DATA_DIR, targz_data_name))

        # copy data to output directory
        # filename must have same name as default!!!
        print_log("Copy default configs to output directory")
        try_copy(config_nnw_pr_path, output_dir_name)
        try_copy(config_dtm_pr_path, output_dir_name)
        try_copy(model_hdf5_path, output_dir_name)

        # make dataset
        if ino_pr["use_last_data"] is False:
            print_log(f"prepare_data_predict({cly.RAW_DATA_DIR},{cly.WORKING_DATA_DIR},{cly.PREDICT_FILE},dtm_pr)")
            bf.prepare_data_predict(cly.RAW_DATA_DIR, cly.WORKING_DATA_DIR, cly.PREDICT_FILE, dtm_pr)

        print_log(f"dataPredictX, dataPredictY, dprXcols, dprYcols = mutils.make_dataset({cly.PREDICT_FILE}, dtm_pr)")
        dataPredictX, dataPredictY, dprXcols, dprYcols = mutils.make_dataset(cly.PREDICT_FILE, dtm_pr)

        # load model
        print_log("Load model")
        modelLoad, model_response_header = mutils.load_model(os.path.join(output_dir_name, model_hdf5_name),
                                                             nnw_pr, True)
        print_log(f"model_response_header == {model_response_header}")

        # make prediction
        print_log("Make prediction on test data")
        prediction_pr = mutils.test_model(modelLoad, nnw_pr, dataPredictX, [])
        print_log(f"mutils.append_new_column_to_csv({cly.PREDICT_FILE}, \
                 {os.path.join(output_dir_name, prediction_outfilename)}, \
                 [prediction_pr, ], [{model_response_header}, ])")
        mutils.append_new_column_to_csv(cly.PREDICT_FILE, os.path.join(output_dir_name, prediction_outfilename),
                                        [prediction_pr, ], [model_response_header, ])

        # return output
        _before_return()
        return _on_return(**kwargs)

    except Exception as err:
        print_log(f"{currentFuncName()}: Unexpected {err=}, {type(err)=}")
        _before_return()
        return 1


def _predict_data(*args):
    """
    (Optional) Helper function to make prediction on an uploaded file
    """
    print_log(f"{currentFuncName()}:")
    message = 'Not implemented (predict_data())'
    message = {"Error": message}
    return message


def _predict_url(*args):
    """
    (Optional) Helper function to make prediction on an URL
    """
    print_log(f"{currentFuncName()}:")
    message = 'Not implemented (predict_url())'
    message = {"Error": message}
    return message


def get_train_args():
    """
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.get_train_args
    :param kwargs:
    :return:
    """
    print_log(f"{currentFuncName()}:")
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

    def _before_return(output_dir_name):
        # delete temp file
        # move log file
        print_log(f"shutil.move({cly.LOG_FILE_PATH}, {output_dir_name}/log_file.txt)")
        shutil.move(cly.LOG_FILE_PATH, output_dir_name + "/log_file.txt")

    def _make_zipfile(source_dir, output_filename):
        shutil.make_archive(output_filename, 'zip', source_dir)

    def _on_return(**kwargs):
        date_suffix = ""
        if os.path.isdir(output_dir_name):
            date_suffix = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
        # make tar.gz file
        print_log(f"_make_zipfile({output_dir_name}, {output_dir_name}{date_suffix}.zip)", log_file=None)
        _make_zipfile(output_dir_name, output_dir_name + date_suffix)
        print_log("OK", log_file=None)
        # send to nextcloud or on gui
        if ino_tr["send_outputs_to"] == "nextcloud":
            print_log(f"shutil.move({output_dir_name}.zip, {cly.NEXTCLOUD}/{ino_tr['path_out']})",
                      log_file=None)
            shutil.move(output_dir_name + ".zip", cly.NEXTCLOUD + "/" + ino_tr["path_out"])
        if ino_tr["send_outputs_to"] == "swagger" or kwargs["accept"] == "application/zip":
            print_log(f"open({output_dir_name}.zip, 'rb', buffering=0)", log_file=None)
            return open(output_dir_name + ".zip", 'rb', buffering=0)

    def _write_mlflow_metrics(stats_key, s):
        tmp = stat.unlist_all(stats_key)
        if isinstance(tmp, (int, float, str)):
            print_log(f"mlflow.log_metric({s} {key}, {tmp})")
            mlflow.log_metric(s + " " + key, tmp)
        elif isinstance(tmp, (list, tuple)):
            for i, m in enumerate(tmp):
                print_log(f"mlflow.log_metric({s} {key}, {m}, step={i})")
                mlflow.log_metric(s + " " + key, m, step=i)

    try:
        # prepare log file
        f = open(cly.LOG_FILE_PATH, "w")
        f.close()
        print_log(f"{currentFuncName()}:")

        # get input values - default values should be preset
        name_dtm_tr = set_kwargs("select_dtm_tr", **kwargs)
        name_mlo_tr = set_kwargs("select_mlo_tr", **kwargs)
        name_nnw_tr = set_kwargs("select_nnw_tr", **kwargs)
        name_ino_tr = set_kwargs("select_ino_tr", **kwargs)
        name_usr_tr = set_kwargs("select_usr_tr", **kwargs)

        config_dtm_tr_path = cfg.file_list_dtm[cfg.config_names_dtm.index(name_dtm_tr)]
        config_mlo_tr_path = cfg.file_list_mlo[cfg.config_names_mlo.index(name_mlo_tr)]
        config_nnw_tr_path = cfg.file_list_nnw[cfg.config_names_nnw.index(name_nnw_tr)]
        config_ino_tr_path = cfg.file_list_ino[cfg.config_names_ino.index(name_ino_tr)]
        config_usr_tr_path = cfg.file_list_usr[cfg.config_names_usr.index(name_usr_tr)]

        # print inputs
        print_log(f"name_dtm_tr == {name_dtm_tr}")
        print_log(f"name_mlo_tr == {name_mlo_tr}")
        print_log(f"name_nnw_tr == {name_nnw_tr}")
        print_log(f"name_ino_tr == {name_ino_tr}")
        print_log(f"name_usr_tr == {name_usr_tr}")
        print_log(f"config_dtm_tr_path == {config_dtm_tr_path}")
        print_log(f"config_mlo_tr_path == {config_mlo_tr_path}")
        print_log(f"config_nnw_tr_path == {config_nnw_tr_path}")
        print_log(f"config_ino_tr_path == {config_ino_tr_path}")
        print_log(f"config_usr_tr_path == {config_usr_tr_path}")

        # load yaml config files
        dtm_tr = load_config_yaml_file(config_dtm_tr_path)
        mlo_tr = load_config_yaml_file(config_mlo_tr_path)
        nnw_tr = load_config_yaml_file(config_nnw_tr_path)
        ino_tr = load_config_yaml_file(config_ino_tr_path)
        usr_tr = load_config_yaml_file(config_usr_tr_path)

        print_log(f"ino_tr['use_last_data'] == {ino_tr['use_last_data']}")
        if ino_tr["use_last_data"] is False:
            print_log(f"mutils.delete_old_files({cly.WORKING_DATA_DIR}, .csv)")
            mutils.delete_old_files(cly.WORKING_DATA_DIR, ".csv")
            print_log(f"mutils.delete_old_files({cly.RAW_DATA_DIR}, .csv)")
            mutils.delete_old_files(cly.RAW_DATA_DIR, ".csv")
            print_log(f"mutils.delete_old_files({cly.TRAIN_DIR}, .csv)")
            mutils.delete_old_files(cly.TRAIN_DIR, ".csv")
            print_log(f"mutils.delete_old_files({cly.TEST_DIR}, .csv)")
            mutils.delete_old_files(cly.TEST_DIR, ".csv")
            print_log(f"mutils.delete_old_files({cly.VALIDATION_DIR}, .csv)")
            mutils.delete_old_files(cly.VALIDATION_DIR, ".csv")

        # data source
        if ino_tr["data_source"] == "server":
            data_source = ""
        elif ino_tr["data_source"] == "nextcloud":
            data_source = cly.NEXTCLOUD_DATA_DIR

        # prepare output_name and deleted directories
        send_to = ""
        if ino_tr["send_outputs_to"] == "nextcloud":
            send_to = cly.NEXTCLOUD

        if ino_tr["path_out"] == "":
            save_dir = cly.WORK_SAVE_DIR
        else:
            save_dir = os.path.join(send_to, ino_tr["path_out"])

        output_dir_name = os.path.join(save_dir, ino_tr["output_name"])
        if os.path.isdir(output_dir_name) is True:
            output_dir_name = output_dir_name + datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
        print_log(f"os.makedirs({output_dir_name}, exist_ok=True)")
        os.makedirs(output_dir_name, exist_ok=True)

        # model HDF5
        if ino_tr["model_hdf5"] == "":
            model_hdf5_path = cly.DEFAULT_MODEL_HDF5_PATH
            model_hdf5_name = cly.DEFAULT_MODEL_HDF5_FILENAME
        else:
            model_hdf5_path = os.path.join(data_source, ino_tr["model_hdf5"])
            model_hdf5_name = ino_tr["model_hdf5"]

        # set targz path
        if ino_tr["targz_data_path"] == "":
            targz_data_path = cly.DEFAULT_DATA_TARGZ_PATH
            targz_data_name = cly.DEFAULT_DATA_TARGZ_FILENAME
        else:
            targz_data_path = os.path.join(data_source, ino_tr["targz_data_path"])
            targz_data_name = os.path.basename(targz_data_path)
        print_log(f"shutil.copy({targz_data_path}, os.path.join({cly.RAW_DATA_DIR}, {targz_data_name}))")
        shutil.copy(targz_data_path, os.path.join(cly.RAW_DATA_DIR, targz_data_name))

        # set output file names
        if ino_tr["train_outfilename"] == "":
            train_outfilename = cly.TRAIN_OUTFILENAME
        else:
            train_outfilename = ino_tr["train_outfilename"]

        if ino_tr["test_outfilename"] == "":
            test_outfilename = cly.TEST_OUTFILENAME
        else:
            test_outfilename = ino_tr["test_outfilename"]

        if ino_tr["validation_outfilename"] == "":
            validation_outfilename = cly.VALIDATION_OUTFILENAME
        else:
            validation_outfilename = ino_tr["validation_outfilename"]

        print_log(f"tar = tarfile.open(os.path.join({cly.RAW_DATA_DIR}, {targz_data_name}), mode='r:gz')")
        tar = tarfile.open(os.path.join(cly.RAW_DATA_DIR, targz_data_name), mode='r:gz')
        for member in tar.getmembers():
            print_log(f"tar.extract(member, {cly.RAW_DATA_DIR})")
            tar.extract(member, cly.RAW_DATA_DIR)
        print_log("tar.close()")
        tar.close()

        print_log(f"os.path.join({cly.RAW_DATA_DIR}, {targz_data_name})")
        os.remove(os.path.join(cly.RAW_DATA_DIR, targz_data_name))

        # copy data to output directory
        # filename must have same name as default!!!
        print_log("Copy default configs to output directory")
        try_copy(config_nnw_tr_path, output_dir_name)
        try_copy(config_dtm_tr_path, output_dir_name)

        # make dataset
        if ino_tr["use_last_data"] is False:
            print_log(f"bf.prepare_data_train({cly.RAW_DATA_DIR}, {cly.WORKING_DATA_DIR}, {cly.TRAIN_FILE}, \
                    {cly.TEST_FILE}, {cly.VALIDATION_FILE}, dtm_tr)")
            bf.prepare_data_train(cly.RAW_DATA_DIR, cly.WORKING_DATA_DIR, cly.TRAIN_FILE, cly.TEST_FILE,
                                  cly.VALIDATION_FILE, dtm_tr)

        print_log(f"dataTrainX, dataTrainY, dtrXcols, dtrYcols = mutils.make_dataset({cly.TRAIN_FILE}, dtm_tr)")
        dataTrainX, dataTrainY, dtrXcols, dtrYcols = mutils.make_dataset(cly.TRAIN_FILE, dtm_tr)

        print_log(f"dataTestX, dataTestY, dteXcols, dteYcols = mutils.make_dataset({cly.TEST_FILE}, dtm_tr)")
        dataTestX, dataTestY, dteXcols, dteYcols = mutils.make_dataset(cly.TEST_FILE, dtm_tr)

        print_log("dataValidationX, dataValidationY, dvalXcols, dvalYcols = ")
        print_log(f"mutils.make_dataset({cly.VALIDATION_FILE}, dtm_tr)")
        dataValidationX, dataValidationY, dvalXcols, dvalYcols = mutils.make_dataset(cly.VALIDATION_FILE, dtm_tr)

        # copy data to output
        print_log("Copy data to output")
        if os.path.exists(cly.TRAIN_FILE):
            print_log(f"shutil.copy({cly.TRAIN_FILE}, {output_dir_name})")
            shutil.copy(cly.TRAIN_FILE, output_dir_name)
        if os.path.exists(cly.TEST_FILE):
            print_log(f"shutil.copy({cly.TEST_FILE}, {output_dir_name})")
            shutil.copy(cly.TEST_FILE, output_dir_name)
        if os.path.exists(cly.VALIDATION_FILE):
            print_log(f"shutil.copy({cly.VALIDATION_FILE}, {output_dir_name})")
            shutil.copy(cly.VALIDATION_FILE, output_dir_name)

        if ino_tr["use_last_model"] is False:
            # model training
            print_log("trainScores, trainHistories, trainModels = mutils.train_model(dataTrainX, dataTrainY, nnw_tr)")
            trainScores, trainHistories, trainModels = mutils.train_model(dataTrainX, dataTrainY, nnw_tr)
            # best model
            best_model_index = np.argmax(trainScores)
        else:
            # load model
            print_log("Load model")
            modelLoad = mutils.load_model(model_hdf5_path, nnw_tr)
            trainModels = list()
            trainModels.append(modelLoad)
            best_model_index = 0

        # write outputs train, test, validation
        dataY_header = list(dtrYcols)
        for i in range(len(dataY_header)):
            dataY_header[i] = (dataY_header[i][0].replace("measurements", "dataTrainY"), dataY_header[i][1])
        dataY_header = pd.MultiIndex.from_tuples(list(zip(*dataY_header)))
        print_log(f"dataY_header == {dataY_header}")

        model_response_header = list(dtrYcols)
        for i in range(len(model_response_header)):
            model_response_header[i] = (model_response_header[i][0].replace("measurements", "modelResponse"),
                                        model_response_header[i][1])

        # model saving
        print_log(f"mutils.save_model(trainModels[{best_model_index}], os.path.join({output_dir_name}, \
                  {model_hdf5_name}), {model_response_header})")
        mutils.save_model(trainModels[best_model_index], os.path.join(output_dir_name, model_hdf5_name),
                          model_response_header)

        model_response_header = pd.MultiIndex.from_tuples(list(zip(*model_response_header)))
        print_log(f"model_response_header == {model_response_header}")

        print_log("Make prediction on train data")
        print_log(f"prediction_trn = mutils.test_model(trainModels[{best_model_index}],nnw_tr,dataTrainX)")
        prediction_trn = mutils.test_model(trainModels[best_model_index], nnw_tr, dataTrainX)
        print_log(f"mutils.append_new_column_to_csv({cly.TRAIN_FILE},{os.path.join(output_dir_name,train_outfilename)},\
                  [dataTrainY, prediction_trn], [{dataY_header}, {model_response_header}])")
        mutils.append_new_column_to_csv(cly.TRAIN_FILE, (os.path.join(output_dir_name, train_outfilename)),
                                        [dataTrainY, prediction_trn], [dataY_header, model_response_header])

        print_log("Make prediction on test data")
        print_log(f"prediction_tst, acc_tst = mutils.test_model(trainModels[{best_model_index}],nnw_tr, \
                  dataTestX,dataTestY)")
        prediction_tst, acc_tst = mutils.test_model(trainModels[best_model_index], nnw_tr, dataTestX, dataTestY)
        print_log(f"mutils.append_new_column_to_csv({cly.TEST_FILE},{os.path.join(output_dir_name,test_outfilename)},\
                  [dataTestY, prediction_tst], [{dataY_header}, {model_response_header}])")
        mutils.append_new_column_to_csv(cly.TEST_FILE, os.path.join(output_dir_name, test_outfilename),
                                        [dataTestY, prediction_tst], [dataY_header, model_response_header])

        print_log("Make prediction on validation data")
        print_log(f"prediction_val, acc_val = mutils.test_model(trainModels[{best_model_index}], nnw_tr, \
                  dataValidationX, dataValidationY)")
        prediction_val, acc_val = mutils.test_model(trainModels[best_model_index], nnw_tr,
                                                    dataValidationX, dataValidationY)
        print_log(f"mutils.append_new_column_to_csv({cly.VALIDATION_FILE}, \
                  {os.path.join(output_dir_name, validation_outfilename)}, [dataValidationY, prediction_val], \
                  [{dataY_header}, {model_response_header}])")
        mutils.append_new_column_to_csv(cly.VALIDATION_FILE, os.path.join(output_dir_name, validation_outfilename),
                                        [dataValidationY, prediction_val], [dataY_header, model_response_header])

        # compute statistics
        stats_trn, stats_tst, stats_val = {}, {}, {}
        print_log(f"np.shape(dataTrainY) == {np.shape(dataTrainY)}")
        print_log(f"np.shape(prediction_trn) == {np.shape(prediction_trn)}")
        print_log("contingency_table_trn = stat.contingency_table(dataTrainY, prediction_trn)")
        contingency_table_trn = stat.contingency_table(dataTrainY, prediction_trn)
        print_log(f"contingency_table_trn == {contingency_table_trn}")
        print_log("contingency_table_tst = stat.contingency_table(dataTestY, prediction_tst)")
        contingency_table_tst = stat.contingency_table(dataTestY, prediction_tst)
        print_log("contingency_table_val = stat.contingency_table(dataValidationY, prediction_val)")
        print_log(f"contingency_table_tst == {contingency_table_tst}")
        contingency_table_val = stat.contingency_table(dataValidationY, prediction_val)
        print_log(f"np.shape(contingency_table_trn) == {np.shape(contingency_table_trn)}")
        print_log(f"contingency_table_val == {contingency_table_val}")

        print_log(f"ino_tr['statistics'] == {ino_tr['statistics']}")
        for st in ino_tr['statistics']:
            print_log(f"st == {st}")
            if st['stat'] == "contingency_table":
                stats_trn.update({"cont_table": contingency_table_trn})
                stats_tst.update({"cont_table": contingency_table_tst})
                stats_val.update({"cont_table": contingency_table_val})
            elif st['stat'] == "F1":
                stats_trn.update({"F1": stat.metrics_F1(contingency_table_trn)})
                stats_tst.update({"F1": stat.metrics_F1(contingency_table_tst)})
                stats_val.update({"F1": stat.metrics_F1(contingency_table_val)})
            elif st['stat'] == "POD":
                stats_trn.update({"POD": stat.metrics_POD(contingency_table_trn)})
                stats_tst.update({"POD": stat.metrics_POD(contingency_table_tst)})
                stats_val.update({"POD": stat.metrics_POD(contingency_table_val)})
            elif st['stat'] == "FAR":
                stats_trn.update({"FAR": stat.metrics_FAR(contingency_table_trn)})
                stats_tst.update({"FAR": stat.metrics_FAR(contingency_table_tst)})
                stats_val.update({"FAR": stat.metrics_FAR(contingency_table_val)})
            elif st['stat'] == "ACC":
                stats_trn.update({"ACC": stat.metrics_ACC(contingency_table_trn)})
                stats_tst.update({"ACC": stat.metrics_ACC(contingency_table_tst)})
                stats_val.update({"ACC": stat.metrics_ACC(contingency_table_val)})
            elif st['stat'] == "CSI":
                stats_trn.update({"CSI": stat.metrics_CSI(contingency_table_trn)})
                stats_tst.update({"CSI": stat.metrics_CSI(contingency_table_tst)})
                stats_val.update({"CSI": stat.metrics_CSI(contingency_table_val)})
            elif st['stat'] == "HSS":
                stats_trn.update({"HSS": stat.metrics_HSS(contingency_table_trn)})
                stats_tst.update({"HSS": stat.metrics_HSS(contingency_table_tst)})
                stats_val.update({"HSS": stat.metrics_HSS(contingency_table_val)})
            elif st['stat'] == "MSI":
                stats_trn.update({"MSI": stat.metrics_MSI(contingency_table_trn)})
                stats_tst.update({"MSI": stat.metrics_MSI(contingency_table_tst)})
                stats_val.update({"MSI": stat.metrics_MSI(contingency_table_val)})

        # log statistics
        print_log(f"stats_trn == {stats_trn}")
        print_log(f"stats_tst == {stats_tst}")
        print_log(f"stats_val == {stats_val}")

        # save statistics to mlflow
        MLFLOW_TRACKING_USERNAME = usr_tr["user_egi_checkin"]
        print_log(f"MLFLOW_TRACKING_USERNAME == {MLFLOW_TRACKING_USERNAME}")
        MLFLOW_TRACKING_PASSWORD = usr_tr["passwd"]
        print_log(f"MLFLOW_TRACKING_PASSWORD == {MLFLOW_TRACKING_PASSWORD}")
        MLFLOW_REMOTE_SERVER = usr_tr["mlflow_remote_server"]
        print_log(f"MLFLOW_REMOTE_SERVER == {MLFLOW_REMOTE_SERVER}")

        os.environ["MLFLOW_TRACKING_USERNAME"] = MLFLOW_TRACKING_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = MLFLOW_TRACKING_PASSWORD
        os.environ["MLFLOW_REMOTE_SERVER"] = MLFLOW_REMOTE_SERVER

        MLFLOW_EXPERIMENT_NAME = mlo_tr["experiment_name"]
        print_log(f"MLFLOW_EXPERIMENT_NAME == {MLFLOW_EXPERIMENT_NAME}")
        print_log(f"mlflow.set_tracking_uri({MLFLOW_REMOTE_SERVER})")
        mlflow.set_tracking_uri(MLFLOW_REMOTE_SERVER)
        print_log(f"mlflow.set_experiment({MLFLOW_EXPERIMENT_NAME})")
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        with mlflow.start_run():
            print_log(f"mlflow.log_param('Input data models', {dtm_tr['input_data_d1'] + dtm_tr['input_data_d2']})")
            mlflow.log_param("Input data models", dtm_tr["input_data_d1"] + dtm_tr['input_data_d2'])
            print_log(f"mlflow.log_param('Input data sources', {dtm_tr['input_data_sources']})")
            mlflow.log_param("Input data sources", dtm_tr["input_data_sources"])
            print_log(f"mlflow.log_param('Measurements', {dtm_tr['measurements']})")
            mlflow.log_param("Measurements", dtm_tr["measurements"])
            print_log(f"mlflow.log_param('Time tolerance', {dtm_tr['time_tolerance']})")
            mlflow.log_param("Time tolerance", dtm_tr["time_tolerance"])
            print_log(f"mlflow.log_param('Forecast time', {dtm_tr['forecast_time']})")
            mlflow.log_param("Forecast time", dtm_tr["forecast_time"])
            print_log(f"mlflow.log_param('Threshold value', {dtm_tr['threshold_value']})")
            mlflow.log_param("Threshold value", dtm_tr["threshold_value"])
            print_log(f"mlflow.log_param('AREA list', {dtm_tr['dataset'][0]['AREA_list']})")
            mlflow.log_param("AREA list", dtm_tr["dataset"][0]["AREA_list"])
            print_log(f"mlflow.log_param('Train dataset', {dtm_tr['train']['seasons']})")
            mlflow.log_param("Train dataset", dtm_tr["train"]["seasons"])
            print_log(f"mlflow.log_param('Test dataset', {dtm_tr['test']['seasons']})")
            mlflow.log_param("Test dataset", dtm_tr["test"]["seasons"])
            print_log(f"mlflow.log_param('Validation dataset', {dtm_tr['validate']['seasons']})")
            mlflow.log_param("Validation dataset", dtm_tr["validate"]["seasons"])
            for val1 in nnw_tr["model_parameters"]:
                if isinstance(nnw_tr["model_parameters"][val1], list):
                    i = 0
                    for val2 in nnw_tr["model_parameters"][val1]:
                        prfx = "_" + str(i + 1)
                        print_log(f"mlflow.log_param({val1}{prfx}, {val2})")
                        mlflow.log_param(val1 + prfx, val2)
                        i = i + 1
                else:
                    print_log(f"mlflow.log_params({val1}'_' + str(key): val for key, val in \
                            {nnw_tr['model_parameters'][val1].items()})")
                    mlflow.log_params({val1 + "_" + str(key): val for key, val in
                                       nnw_tr["model_parameters"][val1].items()})
            print_log(f"mlflow.log_params({nnw_tr['train_model_settings']})")
            mlflow.log_params(nnw_tr['train_model_settings'])
            for key in stats_trn:
                _write_mlflow_metrics(stats_trn[key], "train")
            for key in stats_tst:
                _write_mlflow_metrics(stats_tst[key], "test")
            for key in stats_val:
                _write_mlflow_metrics(stats_val[key], "validation")

        message = {"status": "ok",
                   "training": []}

        # return output
        _before_return(output_dir_name)
        _on_return()
        return message

    except Exception as err:
        message = {"status": "error",
                   "message": err}
        print_log(f"{currentFuncName()}: Unexpected {err=}, {type(err)=}")
        _before_return(cly.NEXTCLOUD_DATA_DIR)
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
        print("PREDICT:")
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
    subparsers = cmd_parser.add_subparsers(help='methods. Use \"api.py method --help\" to get more info',
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
    # configure arguments for train()
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
