NAME = "ai4eosc_thunder_nowcast_ml"

# working directory
BASE_DIR = "/srv/thunderstorm-nowcast-microstep"
WORKING_DATA_DIR = BASE_DIR + "/" + NAME + "/dataset/data_working_directory"
RAW_DATA_DIR = WORKING_DATA_DIR + "/raw"
TRAIN_DIR = WORKING_DATA_DIR + "/train"
TEST_DIR = WORKING_DATA_DIR + "/test"
VALIDATION_DIR = WORKING_DATA_DIR + "/validate"
PREDICT_DIR = WORKING_DATA_DIR + "/predict"
WORK_SAVE_DIR = BASE_DIR + "/models"
NEXTCLOUD_DATA_DIR = "/storage"
SERVER_DATA_DIR = BASE_DIR + "/data/raw"
DOWNLOADS_TMP_DIR = BASE_DIR + "/data/downloads_tmp"

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

# ouput file names
TRAIN_OUTFILENAME = "train_out.csv"
TEST_OUTFILENAME = "test_out.csv"
VALIDATION_OUTFILENAME = "validation_out.csv"
PREDICTION_OUTFILENAME = "prediction_out.csv"

# config directory
CONFIG_DATA_DIR = BASE_DIR + "/" + NAME + "/configs"
CONFIG_DATA_MANAGEMENT = CONFIG_DATA_DIR + "/config_data_management"
CONFIG_MLFLOW_OUTPUTS = CONFIG_DATA_DIR + "/config_mlflow_outputs"
CONFIG_NEURAL_NETWORKS = CONFIG_DATA_DIR + "/config_neural_networks"
CONFIG_INOUTS = CONFIG_DATA_DIR + "/config_inout_settings"
CONFIG_USERS = CONFIG_DATA_DIR + "/config_users"

# config file name preffixes
CONFIG_DATA_MANAGEMENT_PRFX = "CONFIG_DM_"
CONFIG_MLFLOW_OUTPUTS_PRFX = "CONFIG_ML_OUT_"
CONFIG_NEURAL_NETWORKS_PRFX = "CONFIG_NN_"
CONFIG_INOUTS_PRFX = "CONFIG_INOUT_"
CONFIG_USERS_PRFX = "CONFIG_USER_"

# log file (temp)
LOG_FILE_PATH = BASE_DIR + "/log_" + NAME + ".txt"
