# -*- coding: utf-8 -*-
"""
   Module to define CONSTANTS used across the project
"""

import os
from webargs import fields, validate
from marshmallow import Schema, INCLUDE
# from datetime import datetime
from . import config_layout as cly
from enum import Enum

# default location for input and output data, e.g. directories 'data' and 'models',
# is either set relative to the application path or via environment setting
IN_OUT_BASE_DIR = cly.BASE_DIR
if 'APP_INPUT_OUTPUT_BASE_DIR' in os.environ:
    env_in_out_base_dir = os.environ['APP_INPUT_OUTPUT_BASE_DIR']
    if os.path.isdir(env_in_out_base_dir):
        IN_OUT_BASE_DIR = env_in_out_base_dir
    else:
        msg = "[WARNING] \"APP_INPUT_OUTPUT_BASE_DIR="
        msg = msg + "{}\" is not a valid directory! ".format(env_in_out_base_dir)
        msg = msg + "Using \"BASE_DIR={}\" instead.".format(cly.BASE_DIR)
        print(msg)


def get_config_file_list(config_name, config_name_prefix):
    file_list = []
    config_names = []
    try:
        for x in os.listdir(config_name):
            if x.startswith(config_name_prefix):
                file_list.append(os.path.join(config_name, x))
                name = x.split(config_name_prefix, 1)[1]
                config_names.append(os.path.splitext(name)[0])
    except FileNotFoundError:
        print(f"There is no proper config file in {config_name} directory")
    if len(file_list) == 0:
        file_list = ["---"]
        config_names = ["---"]
    return file_list, config_names


file_list_dtm, config_names_dtm = get_config_file_list(cly.CONFIG_DATA_MANAGEMENT, cly.CONFIG_DATA_MANAGEMENT_PRFX)
file_list_mlo, config_names_mlo = get_config_file_list(cly.CONFIG_MLFLOW_OUTPUTS, cly.CONFIG_MLFLOW_OUTPUTS_PRFX)
file_list_nnw, config_names_nnw = get_config_file_list(cly.CONFIG_NEURAL_NETWORKS, cly.CONFIG_NEURAL_NETWORKS_PRFX)
file_list_ino, config_names_ino = get_config_file_list(cly.CONFIG_INOUTS, cly.CONFIG_INOUTS_PRFX)
file_list_usr, config_names_usr = get_config_file_list(cly.CONFIG_USERS, cly.CONFIG_USERS_PRFX)

# LOG_FILE_PATH = cly.LOG_FILE_PATH


class SerializedField(fields.Field):
    def _serialize(self, value, attr, obj, **kwargs):
        return str(value)


# Input parameters for predict() (deepaas>=1.0.0)
class PredictArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # support 'full_paths' parameter

    # full list of fields: https://marshmallow.readthedocs.io/en/stable/api_reference.html
    # to be able to upload a file for prediction

    select_option_pr = fields.String(
        validate=validate.OneOf(["Prediction",
                                 "Add new config to Data management configs",
                                 "Add new config to MLflow outputs configs",
                                 "Add new config to Neural networks configs",
                                 "Add new config to input/output settings",
                                 "Add new config to MLflow user configs",
                                 "Get all config files"]), missing="Prediction",
        description="Choose what to do"
    )

    select_dtm_pr = fields.String(
        validate=validate.OneOf(config_names_dtm), missing=config_names_dtm[0],
        description="Choose data management config file"
    )

    select_mlo_pr = fields.String(
        validate=validate.OneOf(config_names_mlo), missing=config_names_mlo[0],
        description="Choose MLflow config file"
    )

    select_nnw_pr = fields.String(
        validate=validate.OneOf(config_names_nnw), missing=config_names_nnw[0],
        description="Choose Neural network config file"
    )

    select_ino_pr = fields.String(
        validate=validate.OneOf(config_names_ino), missing=config_names_ino[0],
        description="Choose input/output config file"
    )

    select_usr_pr = fields.String(
        validate=validate.OneOf(config_names_usr), missing=config_names_usr[0],
        description="Choose MLflow user config file"
    )

    cfg_file_pr = fields.Field(
        required=False,
        missing=None,
        type="file",
        data_key="cfg_file",
        location="form",
        description="New configuration file"
    )

    new_config_file_name = fields.String(
        required=False, missing=None, description="Set new config file name"
    )

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


class Season(Enum):
    SPRING = 1
    SUMMER = 2
    AUTUMN = 3
    WINTER = 4


# Input parameters for train() (deepaas>=1.0.0)
class TrainArgsSchema(Schema):
    class Meta:
        unknown = INCLUDE  # support 'full_paths' parameter

    # available fields are e.g. fields.Integer(), fields.Str(), fields.Boolean()
    # full list of fields: https://marshmallow.readthedocs.io/en/stable/api_reference.html

    select_dtm_tr = fields.String(
        validate=validate.OneOf(config_names_dtm), missing=config_names_dtm[0],
        description="Choose data management config file"
    )

    select_mlo_tr = fields.String(
        validate=validate.OneOf(config_names_mlo), missing=config_names_mlo[0],
        description="Choose MLflow config file"
    )

    select_nnw_tr = fields.String(
        validate=validate.OneOf(config_names_nnw), missing=config_names_nnw[0],
        description="Choose Neural network config file"
    )

    select_ino_tr = fields.String(
        validate=validate.OneOf(config_names_ino), missing=config_names_ino[0],
        description="Choose input/output config file"
    )

    select_usr_tr = fields.String(
        validate=validate.OneOf(config_names_usr), missing=config_names_usr[0],
        description="Choose MLflow user config file"
    )

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
