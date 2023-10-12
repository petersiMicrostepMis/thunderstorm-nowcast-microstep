# -*- coding: utf-8 -*-
"""
   Module to define CONSTANTS used across the project
"""

import os
from webargs import fields
from marshmallow import Schema, INCLUDE
from datetime import datetime
import subprocess

# identify basedir for the package
BASE_DIR = os.path.dirname(os.path.normpath(os.path.dirname(__file__)))
NAME = "ai4eosc_thunder_nowcast_ml" #subprocess.run(["python3", BASE_DIR + "/setup.py", "--name"], capture_output = True, text=True).stdout.strip("\n")

# default location for input and output data, e.g. directories 'data' and 'models',
# is either set relative to the application path or via environment setting
IN_OUT_BASE_DIR = BASE_DIR
if 'APP_INPUT_OUTPUT_BASE_DIR' in os.environ:
    env_in_out_base_dir = os.environ['APP_INPUT_OUTPUT_BASE_DIR']
    if os.path.isdir(env_in_out_base_dir):
        IN_OUT_BASE_DIR = env_in_out_base_dir
    else:
        msg = "[WARNING] \"APP_INPUT_OUTPUT_BASE_DIR="
        msg = msg + "{}\" is not a valid directory! ".format(env_in_out_base_dir)
        msg = msg + "Using \"BASE_DIR={}\" instead.".format(BASE_DIR)
        print(msg)

SERVER_DATA_DIR = os.path.join(IN_OUT_BASE_DIR, 'data/raw')
DOWNLOADS_TMP = os.path.join(IN_OUT_BASE_DIR, 'data/downloads_tmp')
NEXTCLOUD_DATA_DIR = ""
WORKING_DATA_DIR = os.path.join(IN_OUT_BASE_DIR, NAME + '/dataset/data_working_directory')
RAW_DATA_DIR = os.path.join(WORKING_DATA_DIR, "raw")
TRAIN_DIR = os.path.join(WORKING_DATA_DIR, "train")
TRAIN_FILE = os.path.join(TRAIN_DIR, "train.csv")
TEST_DIR = os.path.join(WORKING_DATA_DIR, "test")
TEST_FILE = os.path.join(TEST_DIR, "test.csv")
VALIDATION_DIR = os.path.join(WORKING_DATA_DIR, "validate")
VALIDATION_FILE = os.path.join(VALIDATION_DIR, "validate.csv")
PREDICT_DIR = os.path.join(WORKING_DATA_DIR, "predict")
PREDICT_FILE = os.path.join(VALIDATION_DIR, "predict.csv")
CONFIG_DATA_DIR = os.path.join(IN_OUT_BASE_DIR, NAME + '/dataset/config')

TIME_DIR_SUFFIX = datetime.now().strftime("%Y%m%d_%H%M%S")
WORK_SAVE_DIR = lambda x : os.path.join(IN_OUT_BASE_DIR, 'models/' + x + '_model_' + TIME_DIR_SUFFIX)
MODEL_FILE_PATH = lambda x : os.path.join(WORK_SAVE_DIR(x), 'model.h5')
CONFIG_YAML_PATH = os.path.join(BASE_DIR, "CONFIG.yaml")
CONFIG_MODEL_YAML_PATH = lambda x : os.path.join(CONFIG_DATA_DIR, "CONFIG_" + x + ".yaml")


# Input parameters for predict() (deepaas>=1.0.0)
class PredictArgsSchema(Schema):
#class TrainArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # support 'full_paths' parameter

    # full list of fields: https://marshmallow.readthedocs.io/en/stable/api_reference.html
    # to be able to upload a file for prediction
    data_tar = fields.Field(
        required=False,
        missing=None, #os.path.join(SERVER_DATA_DIR, "input_data.tar.gz"), #None,
        type="file",
        data_key="data1",
        location="form",
        description="Select a file tar.gz file with all input files for the prediction. Otherwise, default will be used."
    )

    config_yaml = fields.Field(
        required=False,
        missing=None, #CONFIG_YAML_PATH,
        type="file",
        data_key="data_yaml",
        location="form",
        description="Select a CONFIG yaml file for the prediction. Otherwise, default will be used."
    )

    config_yaml_model = fields.Field(
        required=False,
        missing=None, #CONFIG_MODEL_YAML_PATH("predict"),
        type="file",
        data_key="data_yaml_model",
        location="form",
        description="Select a CONFIG yaml model file for the prediction. Otherwise, default will be used."
    )

    model_h5_file = fields.Field(
        required=False,
        missing=None, #os.path.join(SERVER_DATA_DIR, "model.h5")
        type="file",
        data_key="data_model_h5",
        location="form",
        description="Select a HDF5 model file for the prediction. Otherwise, default will be used."
    )

    # to be able to provide an URL for prediction
    #urls = fields.Url(
    #    required=False,
    #    missing=None,
    #    description="Provide an URL of the data for the prediction"
    #)

    #model_path = fields.String(
    #    required=False, missing=None, description="Select model path"
    #)

    #rclone_nextcloud = fields.Bool(
    #    required=False, missing=False, description="Use nextcloud data source?"
    #)

    #config_yaml_path = fields.String(
    #    required=False, missing=None, description="Select config yaml path"
    #)

    #config_model_yaml_path = fields.String(
    #    required=False, missing=None, description="Select config yaml model path"
    #)
    
    # an input parameter for prediction
    #arg1 = fields.Integer(
    #    required=False,
    #    missing=1,
    #    description="Input argument 1 for the prediction"
    #)

# Input parameters for train() (deepaas>=1.0.0)
class TrainArgsSchema(Schema):
#class PredictArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # support 'full_paths' parameter

    # available fields are e.g. fields.Integer(), fields.Str(), fields.Boolean()
    # full list of fields: https://marshmallow.readthedocs.io/en/stable/api_reference.html

    #model_path = fields.String(
    #    required=False, missing=None, description="Select model path"
    #)

    #rclone_nextcloud = fields.Bool(
    #    required=False, missing=False, description="Use nextcloud data source?"
    #)

    #data_tar_tr = fields.Field(
    #    required=False,
    #    missing=None, #os.path.join(SERVER_DATA_DIR, "input_data.tar.gz"),
    #    type="file",
    #    data_key="data1_tr",
    #    location="form",
    #    description="Select a file tar.gz file with all input files for the prediction. Otherwise, default will be used."
    #)

    #config_yaml_tr = fields.Field(
    #    required=False,
    #    missing=None, #CONFIG_YAML_PATH,
    #   type="file",
    #    data_key="data_yaml_tr",
    #    location="form",
    #    description="Select a CONFIG yaml file for the prediction. Otherwise, default will be used."
    #)

    #config_yaml_model_tr = fields.Field(
    #    required=False,
    #    missing=None, #CONFIG_MODEL_YAML_PATH("predict"),
    #    type="file",
    #    data_key="data_yaml_model_tr",
    #    location="form",
    #    description="Select a CONFIG yaml model file for the prediction. Otherwise, default will be used."
    #)

    arg1 = fields.Integer(
        required=False,
        missing=1,
        description="Input argument 1 for training"
    )

# Input parameters for test() (deepaas>=1.0.0)
class TestArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # support 'full_paths' parameter

    # available fields are e.g. fields.Integer(), fields.Str(), fields.Boolean()
    # full list of fields: https://marshmallow.readthedocs.io/en/stable/api_reference.html

    model_path = fields.String(
        required=False, missing=None, description="Select model path"
    )

    rclone_nextcloud = fields.Bool(
        required=False, missing=False, description="Use nextcloud data source?"
    )

    config_yaml_path = fields.String(
        required=False, missing=None, description="Select config yaml path"
    )

    config_model_yaml_path = fields.String(
        required=False, missing=None, description="Select config yaml model path"
    )

    arg1 = fields.Integer(
        required=False,
        missing=1,
        description="Input argument 1 for testing"
    )

# Input parameters for prepare_datasets() (deepaas>=1.0.0)
class PrepareDatasetsArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # support 'full_paths' parameter

    data_input_source = fields.String(
        required=False, missing=None, description="Select data source type"
    )

    type_data_process = fields.String(
        required=False, missing=None, description="Select one of 'train', 'test', 'predict'"
    )

    arg1 = fields.Integer(
        required=False,
        missing=1,
        description="Input argument 1 for data preparing"
    )

