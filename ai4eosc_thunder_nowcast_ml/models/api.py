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

import os
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

## Authorization
#from flaat import Flaat
#flaat = Flaat()

currentFuncName = lambda n=0: sys._getframe(n + 1).f_code.co_name

def print_log(log_line, verbose=True, time_stamp=True):
    tm = ""
    if time_stamp:
        tm = datetime.now().strftime("%Y-%m-%d %H:%M:%S: ")
    if verbose:
        print(tm + log_line)


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


def set_string_argument(arg_name, arg_default_value, **kwargs):
    try:
        if not kwargs[arg_name] is None and kwargs[arg_name] != "":
            string_argument = kwargs[arg_name]
        else:
            print_log(f"{currentFuncName()}: Info: Set default value for {arg_name}: {arg_default_value}")
            string_argument = arg_default_value
    except Exception:
        string_argument = arg_default_value
    return string_argument


def set_bool_argument(arg_name, arg_default_value, **kwargs):
    bool_argument = arg_default_value
    try:
        bool_argument = eval(kwargs[arg_name])
    except Exception:
        bool_argument = arg_default_value
    return bool_argument


def set_kwargs(argument, arg2=None, **kwargs):
    if argument == "model_path":
        return set_string_argument("model_path", cfg.MODEL_FILE_PATH(arg2), **kwargs)
    elif argument == "rcloneNextcloud":
        set_bool_argument("rclone_nextcloud", False, **kwargs)
    elif argument == "data_input_source":
        return set_string_argument("data_input_source", cfg.SERVER_DATA_DIR, **kwargs)
    elif argument == "type_data_process":
        return set_string_argument("type_data_process", "train", **kwargs)
    elif argument == "config_yaml_path":
        return set_string_argument("config_yaml_path", cfg.CONFIG_YAML_PATH, **kwargs)
    elif argument == "config_model_yaml_path":
        return set_string_argument("config_model_yaml_path", cfg.CONFIG_MODEL_YAML_PATH(arg2), **kwargs)
        
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
        distros = list(pkg_resources.find_distributions(cfg.BASE_DIR, 
                                                        only=True))
        if len(distros) == 1:
            pkg = distros[0]
        else:
            pkg = pkg_resources.find_distributions(cfg.BASE_DIR, only=True)
    except Exception as e:
        raise HTTPBadRequest(reason=e)

    ### One can include arguments for prepare_datasets() in the metadata
    prepare_datasets_args = _fields_to_dict(get_prepare_datasets_args())
    # make 'type' JSON serializable
    for key, val in prepare_datasets_args.items():
        prepare_datasets_args[key]['type'] = str(val['type'])

    ### One can include arguments for train() in the metadata
    train_args = _fields_to_dict(get_train_args())
    # make 'type' JSON serializable
    for key, val in train_args.items():
        train_args[key]['type'] = str(val['type'])

    ### One can include arguments for predict() in the metadata
    predict_args = _fields_to_dict(get_predict_args())
    # make 'type' JSON serializable
    for key, val in predict_args.items():
        predict_args[key]['type'] = str(val['type'])

    ### One can include arguments for test() in the metadata
    test_args = _fields_to_dict(get_test_args())
    # make 'type' JSON serializable
    for key, val in test_args.items():
        test_args[key]['type'] = str(val['type'])

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
        'help-predict' : predict_args,
        'help-test' : test_args
    }

    for line in pkg.get_metadata_lines("PKG-INFO"):
        line_low = line.lower() # to avoid inconsistency due to letter cases
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
    print(f"prepare_datasets_args['data_input_source'] == {prepare_datasets_args['data_input_source']}")
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
        config_yaml = bf.load_config_yaml(cfg.CONFIG_MODEL_YAML_PATH("train"))
        print(f"prepare_data_train({source_path}, {dest_path}, {dest_path_train_file}, {dest_path_test_file}, {dest_path_validate_file}, {config_yaml})")
        bf.prepare_data_train(source_path, dest_path, dest_path_train_file, dest_path_test_file, dest_path_validate_file, config_yaml)
    elif type_data_process == "test":
        dest_path_test_file = cfg.TEST_FILE
        config_yaml = bf.load_config_yaml(cfg.CONFIG_MODEL_YAML_PATH("test"))
        print(f"prepare_data_test({source_path}, {dest_path}, {dest_path_test_file}, {config_yaml})")
        bf.prepare_data_test(source_path, dest_path, dest_path_test_file, config_yaml)
    elif type_data_process == "predict":
        dest_path_predict_file = cfg.PREDICT_FILE
        config_yaml = bf.load_config_yaml(cfg.CONFIG_MODEL_YAML_PATH("predict"))
        print(f"prepare_data_predict({source_path}, {dest_path}, {dest_path_predict_file}, {config_yaml})")
        bf.prepare_data_predict(source_path, dest_path, dest_path_test_file, config_yaml)
    else:
        message = f"{currentFuncName}: Bad value for type_data_process == {type_data_process}. Use one of 'train', 'test', 'predict'"

    return message


def get_predict_args():
    """
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.get_predict_args
    :return:
    """
    return cfg.PredictArgsSchema().fields


@_catch_error
def predict(**kwargs):
#def train(**kwargs):
    """
    Function to execute prediction
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.predict
    :param kwargs:
    :return:
    """

    #if (not any([kwargs['urls'], kwargs['files']]) or
    #        all([kwargs['urls'], kwargs['files']])):
    #    raise Exception("You must provide either 'url' or 'data' in the payload")

    # output dir
    os.mkdir(cfg.WORK_SAVE_DIR("predict"))

    data_input_targz = os.path.join(cfg.SERVER_DATA_DIR, "input_data.tar.gz")
    config_yaml_path = cfg.CONFIG_YAML_PATH
    config_model_yaml_path = cfg.CONFIG_MODEL_YAML_PATH("predict")
    model_path = os.path.join(cfg.SERVER_DATA_DIR, "model.h5")

    if kwargs['data_tar']:
        kwargs['data_tar'] = [kwargs['data_tar']]
        for a in kwargs['data_tar']:
            data_input_targz = a.filename
    
    if kwargs['config_yaml']:
        kwargs['config_yaml'] = [kwargs['config_yaml']]
        for a in kwargs['config_yaml']:
            config_yaml_path = a.filename

    if kwargs['config_yaml_model']:
        kwargs['config_yaml_model'] = [kwargs['config_yaml_model']]
        for a in kwargs['config_yaml_model']:
            config_model_yaml_path = a.filename

    if kwargs['model_h5_file']:
        kwargs['model_h5_file'] = [kwargs['model_h5_file']]
        for a in kwargs['model_h5_file']:
            model_path = a.filename
    try:
        shutil.rmtree(cfg.DOWNLOADS_TMP)
    except:
    os.mkdir(cfg.DOWNLOADS_TMP)
    shutil.copy(data_input_targz, cfg.DOWNLOADS_TMP + "/input.tar.gz")
    subprocess.run("tar -xzvf " + cfg.DOWNLOADS_TMP + "/input.tar.gz --directory " + cfg.DOWNLOADS_TMP,
                   shell=True, check=True, executable='/bin/bash')
    data_input_source = cfg.DOWNLOADS_TMP + "/input.tar.gz"
    
    shutil.copy(config_yaml_path, cfg.WORK_SAVE_DIR("predict") + "/CONFIG_NN.yaml")
    config_yaml_path = cfg.WORK_SAVE_DIR("predict") + "/CONFIG_NN.yaml"
    shutil.copy(config_model_yaml_path, cfg.WORK_SAVE_DIR("predict") + "/CONFIG_model.yaml")
    config_model_yaml_path = cfg.WORK_SAVE_DIR("predict") + "/CONFIG_model.yaml"
    shutil.copy(model_path, cfg.WORK_SAVE_DIR("predict") + "/model.h5")
    model_path = cfg.WORK_SAVE_DIR("predict") + "/model.h5"

    # CONFIG.yaml
    config_yaml = load_config_yaml_file(config_yaml_path)
    config_model_yaml = load_config_yaml_file(config_model_yaml_path)

    # delete old files in working_dir (only with .csv extension)
    mutils.delete_old_files(cfg.RAW_DATA_DIR, ".csv")
    mutils.delete_old_files(cfg.TRAIN_DIR, ".csv")
    mutils.delete_old_files(cfg.TEST_DIR, ".csv")
    mutils.delete_old_files(cfg.VALIDATION_DIR, ".csv")

    # prepare data
    print(f"prepare_data_predict({cfg.DOWNLOADS_TMP}, {cfg.RAW_DATA_DIR}, {cfg.PREDICT_FILE}, {config_model_yaml})")
    bf.prepare_data_predict(cfg.DOWNLOADS_TMP, cfg.RAW_DATA_DIR, cfg.PREDICT_FILE, config_model_yaml)
    dataPredictX, dataPredictY = mutils.make_dataset(cfg.PREDICT_FILE, config_model_yaml)
    # copy data to output
    shutil.copy(cfg.PREDICT_FILE, cfg.WORK_SAVE_DIR("predict"))
    # load model
    modelLoad = mutils.load_model(model_path, config_yaml)
    # predictions model
    prediction = mutils.test_model(modelLoad, dataPredictX)
    print(prediction)
   
    message = { "status": "ok",
                "training": []}
    return message


def load_files(s, *arg):
    files = []
    for arg in args:
        file_objs = arg['files']
        for f in file_objs:
            files.append(f.filename)
    return files

def _predict_data(*args):
    """
    (Optional) Helper function to make prediction on an uploaded file
    """
    #message = 'Not implemented (predict_data())'
    #message = {"Error": message}
    files = []
    file_inputs = ['data_tar', 'config_yaml', 'config_yaml_model', 'model_h5_file']
    for arg in args:
        for f_i in file_inputs:
            file_objs = arg[f_i]
            for f in file_objs:
                files.append({f_i: f.filename})

    return files #message


def _predict_url(*args):
    """
    (Optional) Helper function to make prediction on an URL
    """
    message = 'Not implemented (predict_url())'
    message = {"Error": message}
    return message


def get_test_args():
    """
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.get_predict_args
    :return:
    """
    return cfg.TestArgsSchema().fields


@_catch_error
def test(**kwargs):
    message = { "status": "ok",
                "training": [],
              }

    # use the schema
    schema = cfg.TestArgsSchema()
    # deserialize key-word arguments
    test_args = schema.load(kwargs)
    
    test_results = { "Error": "No model implemented for test (test())" }
    message["test"].append(test_results)

    # model_path
    model_path = set_kwargs("model_path", arg2="test", **kwargs)

    # rclone_nextcloud
    rclone_nextcloud = set_kwargs("rclone_nextcloud", **kwargs)

    # config_yaml_path
    config_yaml_path = set_kwargs("config_yaml_path", **kwargs)

    # config_model_yaml_path
    config_model_yaml_path = set_kwargs("config_model_yaml_path", arg2="test", **kwargs)

    # CONFIG.yaml
    config_yaml = load_config_yaml_file(config_yaml_path)
    config_model_yaml = load_config_yaml_file(config_model_yaml_path)

    # prepare data
    dataTestX, dataTestY = mutils.make_dataset(cfg.TEST_FILE, config_model_yaml)

    # output dir
    os.mkdir(cfg.WORK_SAVE_DIR("test"))

    # copy data to output
    shutil.copy(cfg.TEST_FILE, cfg.WORK_SAVE_DIR("test"))
    shutil.copy(config_yaml_path, cfg.WORK_SAVE_DIR("test"))
    shutil.copy(config_model_yaml_path, cfg.WORK_SAVE_DIR("test"))
    shutil.copy(model_path, cfg.WORK_SAVE_DIR("test"))

    # load model
    modelLoad = mutils.load_model(model_path, config_yaml)
    # predictions model
    prediction = mutils.test_model(modelLoad, dataTestX)
    print(prediction)
    # compare prediction and dataTestY

    return message


def get_train_args():
    """
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.get_train_args
    :param kwargs:
    :return:
    """
    return cfg.TrainArgsSchema().fields


###
# @flaat.login_required() line is to limit access for only authorized people
# Comment this line, if you open training for everybody
# More info: see https://github.com/indigo-dc/flaat
###
#@flaat.login_required() # Allows only authorized people to train
def train(**kwargs):
#def predict(**kwargs):
    """
    Train network
    https://docs.deep-hybrid-datacloud.eu/projects/deepaas/en/latest/user/v2-api.html#deepaas.model.v2.base.BaseModel.train
    :param kwargs:
    :return:
    """

    message = { "status": "ok",
                "training": [],
              }

    # use the schema
    #schema = cfg.TrainArgsSchema()
    # deserialize key-word arguments
    #train_args = schema.load(kwargs)
    
    os.mkdir(cfg.WORK_SAVE_DIR("train"))

    data_input_targz = os.path.join(cfg.SERVER_DATA_DIR, "input_data.tar.gz")
    config_yaml_path = cfg.CONFIG_YAML_PATH
    config_model_yaml_path = cfg.CONFIG_MODEL_YAML_PATH("train_model")

    try:
        shutil.rmtree(cfg.DOWNLOADS_TMP)
    except:
        print("dbvisu")
    os.mkdir(cfg.DOWNLOADS_TMP)
    shutil.copy(data_input_targz, cfg.DOWNLOADS_TMP + "/input.tar.gz")
    subprocess.run("tar -xzvf " + cfg.DOWNLOADS_TMP + "/input.tar.gz --directory " + cfg.DOWNLOADS_TMP,
                   shell=True, check=True, executable='/bin/bash')
    data_input_source = cfg.DOWNLOADS_TMP + "/input.tar.gz"
    
    shutil.copy(config_yaml_path, cfg.WORK_SAVE_DIR("train") + "/CONFIG_NN.yaml")
    config_yaml_path = cfg.WORK_SAVE_DIR("train") + "/CONFIG_NN.yaml"
    shutil.copy(config_model_yaml_path, cfg.WORK_SAVE_DIR("train") + "/CONFIG_model.yaml")
    config_model_yaml_path = cfg.WORK_SAVE_DIR("train") + "/CONFIG_model.yaml"

    model_path = cfg.WORK_SAVE_DIR("train") + "/model.h5"

    # CONFIG.yaml
    config_yaml = load_config_yaml_file(config_yaml_path)
    config_model_yaml = load_config_yaml_file(config_model_yaml_path)

    # delete old files in working_dir (only with .csv extension)
    mutils.delete_old_files(cfg.RAW_DATA_DIR, ".csv")
    mutils.delete_old_files(cfg.TRAIN_DIR, ".csv")
    mutils.delete_old_files(cfg.TEST_DIR, ".csv")
    mutils.delete_old_files(cfg.VALIDATION_DIR, ".csv")

    # prepare data
    print(f"prepare_data_train({cfg.DOWNLOADS_TMP}, {cfg.RAW_DATA_DIR}, {cfg.TRAIN_FILE}, {cfg.TEST_FILE}, {cfg.VALIDATION_FILE}, {config_model_yaml})")
    bf.prepare_data_train(cfg.DOWNLOADS_TMP, cfg.RAW_DATA_DIR, cfg.TRAIN_FILE, cfg.TEST_FILE, cfg.VALIDATION_FILE, config_model_yaml)

    # prepare data
    dataTrainX, dataTrainY = mutils.make_dataset(cfg.TRAIN_FILE, config_model_yaml)

    # copy data to output
    shutil.copy(cfg.TRAIN_FILE, cfg.WORK_SAVE_DIR("train"))
    shutil.copy(cfg.TEST_FILE, cfg.WORK_SAVE_DIR("train"))
    shutil.copy(cfg.VALIDATION_FILE, cfg.WORK_SAVE_DIR("train"))
    # model training
    trainScores, trainHistories, trainModels = mutils.train_model(
        dataTrainX, dataTrainY, config_yaml
    )
    # model saving
    mutils.save_model(trainModels[1], model_path)
    train_results = { "Error": "No model implemented for training (train())" }
    message["training"].append(train_results)
    return(message)

    # model_path
    model_path = set_kwargs("model_path", arg2="train", **kwargs)

    # rclone_nextcloud
    rclone_nextcloud = set_kwargs("rclone_nextcloud", **kwargs)
    # config_yaml_path
    #config_yaml_path = set_kwargs("config_yaml_path", **kwargs)

    kwargs['files'] = [kwargs['files']]
    #kwargs['config_model_yaml'] = [kwargs['config_model_yaml']]
    config_yaml = load_files(kwargs)
    #config_model_yaml = load_files('config_model_yaml', kwargs)

    # config_model_yaml_path
    #config_model_yaml_path = set_kwargs("config_model_yaml_path", arg2="train", **kwargs)

    # CONFIG.yaml
    #config_yaml = load_config_yaml_file(config_yaml_path)
    #config_model_yaml = load_config_yaml_file(config_model_yaml_path)

    # prepare data
    #dataTrainX, dataTrainY = mutils.make_dataset(cfg.TRAIN_FILE, config_model_yaml)

    # output dir
    #os.mkdir(cfg.WORK_SAVE_DIR("train"))

    # copy data to output
    #shutil.copy(cfg.TRAIN_FILE, cfg.WORK_SAVE_DIR("train"))
    #shutil.copy(cfg.TEST_FILE, cfg.WORK_SAVE_DIR("train"))
    #shutil.copy(cfg.VALIDATION_FILE, cfg.WORK_SAVE_DIR("train"))
    #shutil.copy(config_yaml_path, cfg.WORK_SAVE_DIR("train"))
    #shutil.copy(config_model_yaml_path, cfg.WORK_SAVE_DIR("train"))

    # model training
    #trainScores, trainHistories, trainModels = mutils.train_model(
    #    dataTrainX, dataTrainY, config_yaml
    #)

    # model saving
    #mutils.save_model(trainModels[1], model_path)

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
        #if args.files:
            # create tmp file as later it will be deleted
            #print('abc')
            #temp = tempfile.NamedTemporaryFile()
            #temp.close()
            # copy original file into tmp file
            #print(args.files)
            #with open(args.files, "rb") as f:
            #    print(f)
            #    with open(temp.name, "wb") as f_tmp:
            #        print(temp.name)
            #        for line in f:
            #            f_tmp.write(line)
        
            # create file object to mimic aiohttp workflow
            #file_obj = wrapper.UploadedFile(name="data1", 
            #                                filename = temp.name,
            #                                content_type=mimetypes.MimeTypes().guess_type(args.files)[0],
            #                                original_filename=args.files)
            #args.files = file_obj
        
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
    ## configure parser to call get_metadata()
    get_metadata_parser = subparsers.add_parser('get_metadata', 
                                         help='get_metadata method',
                                         parents=[parser])
    # normally there are no arguments to configure for get_metadata()
    
    # -------------------------------------------------------------------------------------
    ## configure arguments for prepare_datasets()
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
    ## configure arguments for predict()
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

    # -------------------------------------------------------------------------------------
    ## configure arguments for test()
    test_parser = subparsers.add_parser('test', 
                                        help='commands for testing',
                                        parents=[parser]) 
    # one should convert get_test_args() to add them in test_parser
    # For example:
    test_args = _fields_to_dict(get_test_args())
    for key, val in test_args.items():
        test_parser.add_argument('--%s' % key,
                                 default=val['default'],
                                 type=val['type'],
                                 help=val['help'],
                                 required=val['required'])

    args = cmd_parser.parse_args()
    
    main()
