import os
# import datetime
NAME = "ai4eosc_thunder_nowcast_ml"

# working directory
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
WORKING_DATA_DIR = BASE_DIR + "/" + NAME + "/dataset/data_working_directory"
RAW_DATA_DIR = WORKING_DATA_DIR + "/raw"
TRAIN_DIR = WORKING_DATA_DIR + "/train"
TEST_DIR = WORKING_DATA_DIR + "/test"
VALIDATION_DIR = WORKING_DATA_DIR + "/validate"
PREDICT_DIR = WORKING_DATA_DIR + "/predict"
WORK_SAVE_DIR = BASE_DIR + "/models"
NEXTCLOUD = "/storage"
NEXTCLOUD_DATA_DIR = NEXTCLOUD + "/EOSC_runs/data"
SERVER_DATA_DIR = BASE_DIR + "/data/raw"
DOWNLOADS_TMP_DIR = BASE_DIR + "/data/downloads_tmp"
SERVER_DEFAULT_CONFIGS_DIR = BASE_DIR + "/data/configs"
SERVER_USER_CONFIGS_DIR = BASE_DIR + "/data/user_configs"
NEXTCLOUD_CONFIG_DIR = BASE_DIR + "/data/configs_2"  # NEXTCLOUD + "/EOSC_runs/configs"
NEXTCLOUD_USER_CONFIG_DIR = BASE_DIR + "/data/user_configs"  # NEXTCLOUD + "/EOSC_runs/user_configs"
# CONFIG_DATA_DIR = BASE_DIR + "/" + NAME + "/configs"
LOG_FILE_DIR = BASE_DIR + "/" + NAME + "/log_files"

# data on server
DEFAULT_DATA_TARGZ_FILENAME = "input.tar.gz"
DEFAULT_DATA_TARGZ_PATH = SERVER_DATA_DIR + "/" + DEFAULT_DATA_TARGZ_FILENAME
DEFAULT_MODEL_HDF5_FILENAME = "model.h5"
DEFAULT_MODEL_HDF5_PATH = SERVER_DATA_DIR + "/" + DEFAULT_MODEL_HDF5_FILENAME
# data on cloud
# NEXTCLOUD_INPUTS = BASE_DIR + "/data/nextcloud_inputs"
# GUI_INPUTS = BASE_DIR + "/data/gui_inputs"

# files in working directory
TRAIN_FILE = TRAIN_DIR + "/train.csv"
TEST_FILE = TEST_DIR + "/test.csv"
VALIDATION_FILE = VALIDATION_DIR + "/validate.csv"
PREDICT_FILE = VALIDATION_DIR + "/predict.csv"

# output file names
TRAIN_OUTFILENAME = "train_out.csv"
TEST_OUTFILENAME = "test_out.csv"
VALIDATION_OUTFILENAME = "validation_out.csv"
PREDICTION_OUTFILENAME = "prediction_out.csv"

# config directory
CONFIG_DATA_MANAGEMENT = "config_data_management"
CONFIG_MLFLOW_OUTPUTS = "config_mlflow_outputs"
CONFIG_NEURAL_NETWORKS = "config_neural_networks"
CONFIG_INOUTS = "config_inout_settings"
CONFIG_USERS = "config_users"

# config file name preffixes
CONFIG_DATA_MANAGEMENT_PRFX = "CONFIG_DM_"
CONFIG_MLFLOW_OUTPUTS_PRFX = "CONFIG_ML_OUT_"
CONFIG_NEURAL_NETWORKS_PRFX = "CONFIG_NN_"
CONFIG_INOUTS_PRFX = "CONFIG_INOUT_"
CONFIG_USERS_PRFX = "CONFIG_USER_"

# log file
LOG_FILE_PATH = LOG_FILE_DIR + "/log_" + NAME + ".txt"
