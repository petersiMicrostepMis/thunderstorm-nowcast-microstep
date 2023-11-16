# -*- coding: utf-8 -*-
"""
   Module to define CONSTANTS used across the project
"""

import os
from webargs import fields, validate
from marshmallow import Schema, INCLUDE
from datetime import datetime
import subprocess

# identify basedir for the package
BASE_DIR = os.path.dirname(os.path.normpath(os.path.dirname(__file__)))
NAME = "ai4eosc_thunder_nowcast_ml"  # subprocess.run(["python3", BASE_DIR + "/setup.py", "--name"], capture_output = True, text=True).stdout.strip("\n")
# NAME = "uc-microstep-mis-ai4eosc_thunder_nowcast_ml"  # subprocess.run(["python3", BASE_DIR + "/setup.py", "--name"], capture_output = True, text=True).stdout.strip("\n")

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
DEFAULT_DATA_TARGZ_FILENAME = "input_data.tar.gz"
DEFAULT_DATA_TARGZ = os.path.join(SERVER_DATA_DIR, DEFAULT_DATA_TARGZ_FILENAME)
DOWNLOADS_TMP = os.path.join(IN_OUT_BASE_DIR, 'data/downloads_tmp')
NEXTCLOUD_DATA_DIR = "/storage/"
NEXTCLOUD_INPUTS = os.path.join(IN_OUT_BASE_DIR, 'data/nextcloud_inputs')
GUI_INPUTS = os.path.join(IN_OUT_BASE_DIR, 'data/gui_inputs')
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
WORK_SAVE_DIR = os.path.join(IN_OUT_BASE_DIR, 'models')
OUTPUT_NAME = lambda x : x + '_model_' + datetime.now().strftime("%Y%m%d_%H%M%S")  # TIME_DIR_SUFFIX

LOG_FILE_PATH = "/tmp/log_" + NAME + ".txt"

MODEL_FILE_NAME = "model.h5"
CONFIG_YAML_NN_FILE_NAME = "CONFIG.yaml"
CONFIG_YAML_DATA_FILE_NAME = lambda x : "CONFIG_" + x + ".yaml"

MODEL_FILE_PATH = os.path.join(SERVER_DATA_DIR, MODEL_FILE_NAME)
CONFIG_YAML_NN_PATH = os.path.join(BASE_DIR, CONFIG_YAML_NN_FILE_NAME)
CONFIG_YAML_DATA_PATH = lambda x : os.path.join(CONFIG_DATA_DIR, CONFIG_YAML_DATA_FILE_NAME(x))


class SerializedField(fields.Field):
    def _serialize(self, value, attr, obj, **kwargs):
        return str(value)


# Input parameters for predict() (deepaas>=1.0.0)
class PredictArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # support 'full_paths' parameter

    # full list of fields: https://marshmallow.readthedocs.io/en/stable/api_reference.html
    # to be able to upload a file for prediction

    get_default_configs = fields.Bool(
        required=False,
        missing=False,
        description="If True is selected, prediction returns default files (configuration, data)."
    )

    use_last_data = fields.Bool(
        required=False,
        missing=True,
        description="If True is selected, new dataset won't be used."
    )

    conf_nn = fields.Field(
        required=False,
        missing=None,
        type="file",
        data_key="conf_nn_pred",
        location="form",
        description="Select config YAML file for neural network architecture. Empty file -> default will be used."
    )

    conf_data = fields.Field(
        required=False,
        missing=None,
        type="file",
        data_key="conf_data_pred",
        location="form",
        description="Select config YAML file for prediction. Empty file -> default will be used."
    )

    model_hdf5 = fields.Field(
        required=False,
        missing=None,
        type="file",
        data_key="model_hdf5",
        location="form",
        description="Select HDF5 model file for prediction. Empty file -> default will be used."
    )

    data_pred = fields.Field(
        required=False,
        missing=None,
        type="file",
        data_key="data_pred",
        location="form",
        description="Select input data tar.gz file for prediction. Empty file -> nextcloud will be used."
    )

#    urls_inp = fields.Url(
#        required=False,
#        missing=None,
#        description="Provide an URL of the data for the prediction. If empty file + no url -> default data will be used."
#    )

    output_name = fields.String(
        required=False, missing=None, description="Write output file name. Otherwise, some default name will be used."
    )

    path_inp = fields.String(
        required=False, missing=None, description="Provide path in Nextcloud for input the training. If empty file + no url -> default data will be used."
    )

    path_out = fields.String(
        required=False, missing=None, description="Provide path in Nextcloud for output the training. If no path -> output on screen."
    )

#    urls_out = fields.Url(
#        required=False,
#        missing=None,
#        description="Provide an URL for output the prediction. If no url -> output on screen."
#    )

    accept = fields.Str(
        load_default="application/json",
        location="headers",
        validate=validate.OneOf(
            ["application/json", "application/zip"]
        ),
        metadata={
            "description": "Returns a zip file with prediction results and used config files."
        },
    )


# Input parameters for train() (deepaas>=1.0.0)
class TrainArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # support 'full_paths' parameter

    # available fields are e.g. fields.Integer(), fields.Str(), fields.Boolean()
    # full list of fields: https://marshmallow.readthedocs.io/en/stable/api_reference.html
    use_last_data = fields.Bool(
        required=False,
        missing=True,
        description="If True is selected, new dataset won't be used."
    )

    conf_nn = SerializedField(
        required=False,
        missing=None,
        # dump_default=open('/home/peto/Downloads/empty.txt', mode='r'), # no effect
        type="file",
        data_key="conf_nn_train",
        # location="form", # serialization error
        description="Select config YAML file for neural network architecture. Empty file -> default will be used."
    )

    conf_data = SerializedField(
        required=False,
        missing=None,
        # dump_default=open('/home/peto/Downloads/empty.txt', mode='r'), # no effect
        type="file",
        data_key="conf_data_train",
        # location="form", # serialization error
        description="Select config YAML file for training. Empty file -> default will be used."
    )

    data_train = SerializedField(
        required=False,
        missing=None,
        # dump_default=open('/home/peto/Downloads/empty.txt', mode='r'), # no effect
        type="file",
        data_key="data_train",
        # location="form", # serialization error
        description="Select input data tar.gz file for training. Empty file -> nextcloud will be used."
    )

#    urls_inp = fields.Url(
#        required=False,
#        missing=None,
#        description="Provide an URL of the data for the training. If empty file + no url -> default data will be used."
#    )

    output_name = fields.String(
        required=False, missing=None, description="Write output file name. Otherwise, some default name will be used."
    )

    path_inp = fields.String(
        required=False, missing=None, description="Provide path in Nextcloud for input the training. If empty file + no url -> default data will be used."
    )

    path_out = fields.String(
        required=False, missing=None, description="Provide path in Nextcloud for output the training. If no path -> output on screen."
    )

#    urls_out = fields.Url(
#        required=False,
#        missing=None,
#        description="Provide an URL for output the training. If no url -> output on screen."
#    )

    accept = fields.Str(
        load_default="application/json",
        location="headers",
        validate=validate.OneOf(
            ["application/json", "application/zip"]
        ),
        metadata={
            "description": "Returns a zip file with prediction results and used config files."
        },
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

