# -*- coding: utf-8 -*-
"""
Feature building
"""

import os
import sys
base_directory = os.path.dirname(os.path.abspath(__file__))
base_directory = os.path.dirname(os.path.dirname(base_directory))
sys.path.append(base_directory)

# import project config.py
import uc-microstep-mis-ai4eosc_thunder_nowcast_ml.config as cfg
import yaml
import pandas as pd
import csv
import numpy as np
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


def load_config_yaml(pathYaml, part=""):
    with open(pathYaml) as yamlFile:
        config = yaml.safe_load(yamlFile)
    if part == "":
        return config
    else:
        return config[part]


def csv_data_to_one_file(source_path, dest_path, use_columns, forecast_time=None):
    data_files = os.listdir(source_path)
    data_files = [f for f in data_files if os.path.isfile(source_path + '/' + f)]
    df_csv_append = pd.DataFrame()
    for data_file in data_files:
        if os.path.isfile(source_path + '/' + data_file):
            print_log(f"{currentFuncName()}: Reading file: {source_path}/{data_file} ...")
            df = pd.read_csv(source_path + '/' + data_file, na_filter=False)
            use_columns_2 = [use_columns[i] for i in range(len(use_columns)) if use_columns[i] in df.columns]
            if forecast_time is not None:
                forecast_time_indices = [i for i in range(len(df['forecast'])) if df['forecast'][i] == forecast_time]
                df = df.iloc[forecast_time_indices]
            df = df[use_columns_2]
            if len(df_csv_append) == 0 or list(df.columns) == list(df_csv_append.columns):
                df_csv_append = pd.concat([df_csv_append, df], ignore_index=True)
                print_log(f"{currentFuncName()}: OK")
            else:
                print_log(f"{currentFuncName()}: Error: headers don't match. Skipping this file.")
        else:
            print_log(f"{currentFuncName()}: Warning: Skipping file {source_path}/{data_file}, it does not exist.")
    if len(df_csv_append) > 0:
        df_csv_append.to_csv(dest_path, index=False)
    else:
        print_log(f"{currentFuncName()}: Warning: No such file")


def make_raw_csv_data(source_path, dest_path, data_sources, file_types, use_columns, forecast_time=None):
    if not isinstance(data_sources, list) and not isinstance(data_sources, tuple):
        data_sources = (data_sources,)
    if not isinstance(file_types, list) and not isinstance(file_types, tuple):
        file_types = (file_types,)
    for ds in data_sources:
        for ft in file_types:
            csv_data_to_one_file(source_path + "/" + ds + "/" + ft,
                                 dest_path + "/" + ds + "__" + ft + ".csv",
                                 use_columns, forecast_time)


def load_csv_files(source_path_list, config_yaml, header_list):
    if not isinstance(source_path_list, list) and not isinstance(source_path_list, tuple):
        source_path_list = (source_path_list,)
    output_df = list()
    output_header = list()
    j = 0
    # print(len(source_path_list))
    for i in range(len(source_path_list)):
        # print(source_path_list[i])
        # print(os.path.isfile(source_path_list[i]))
        if os.path.isfile(source_path_list[i]):
            output_df.append(pd.DataFrame())
            output_df[j] = pd.read_csv(source_path_list[i], na_filter=False)
            output_header.append(header_list[i])
            j = j + 1
    return output_df, output_header


def values_in_df_to_classes(dfs, column_list, threshold, val1, val2):
    if not isinstance(dfs, list) and not isinstance(dfs, tuple):
        dfs = (dfs,)
    for i in range(len(dfs)):
        for column in column_list:
            dfs[i][column] = np.float64(dfs[i][column])
            dfs[i][column][np.isnan(dfs[i][column])] = 0
            # dealing with only two classes for now
            dfs[i][column] = np.where(dfs[i][column] >= threshold, val1, val2)
    return dfs


def interval(t, dt):
    return (t - dt, t + dt)


def merge_csv_files(d_files, m_files, tolerance):
    if not isinstance(d_files, list) and not isinstance(d_files, tuple):
        d_files = (d_files,)
    if not isinstance(m_files, list) and not isinstance(m_files, tuple):
        m_files = (m_files,)

    timestamps = d_files[0]['timestamp']
    forecast = d_files[0]['forecast'] * 60 * 1000
    output = -np.ones((len(timestamps), len(d_files) + len(m_files)), dtype=np.int64)
    output[:, 0] = np.array(list(range(len(timestamps))))
    # d_files
    print_log(f"{currentFuncName()}: Merging {len(d_files)} data files")
    for i in range(1, len(d_files)):
        for j in range(len(timestamps)):
            time_interval = interval(timestamps[j] + forecast[j], tolerance)
            time_between_b = d_files[i]['timestamp'].between(time_interval[0], time_interval[1])
            time_between_i = np.where(time_between_b)[0]
            if len(time_between_i) >= 1:
                output[j, i] = time_between_i[0]
    # m_files
    print_log(f"{currentFuncName()}: Merging {len(m_files)} measurement files")
    for i in range(len(m_files)):
        for j in range(len(timestamps)):
            time_interval = interval(timestamps[j] + forecast[j], tolerance)
            time_between_b = m_files[i]['timestamp'].between(time_interval[0], time_interval[1])
            time_between_i = np.where(time_between_b)[0]
            if len(time_between_i) >= 1:
                output[j, i + len(d_files)] = time_between_i[0]

    use_this_rows = np.sum(output > 0, axis=1) == np.shape(output)[1]
    d_files_out = list()
    m_files_out = list()
    for i in range(len(d_files)):
        d_files_out.append(d_files[i].iloc[output[use_this_rows, i]])
    for i in range(len(m_files)):
        m_files_out.append(m_files[i].iloc[output[use_this_rows, i + len(d_files)]])

    return(d_files_out, m_files_out)


def get_proper_dates_indices(timestamps, config_yaml_date_settings):
    dates = [datetime.fromtimestamp(timestamps.iloc[i]) for i in range(len(timestamps))]
    datesY = [dates[i].year for i in range(len(dates))]
    datesM = [dates[i].month for i in range(len(dates))]
    datesD = [dates[i].day for i in range(len(dates))]

    indices = np.zeros((len(dates), 1))

    if config_yaml_date_settings is None:
        return [i for i in range(len(indices)) if indices[i] == 0]
    elif len(config_yaml_date_settings) != 1:
        print_log(f"{currentFuncName()}: Error: Bad data splitting yaml config")
        return [i for i in range(len(indices)) if indices[i] == 0]

    seasons = config_yaml_date_settings[0]
    seasonsY = eval(seasons['years'])
    seasonsM = eval(seasons['months'])
    seasonsD = eval(seasons['days'])

    if len(seasonsY) == 0 and len(seasonsM) == 0 and len(seasonsD) == 0:  # none
        indices = indices
    elif len(seasonsY) == 0 and len(seasonsM) == 0 and len(seasonsD) > 0:  # D
        for i in range(len(indices)):
            if datesD[i] in seasonsD:
                indices[i] = 1

    elif len(seasonsY) == 0 and len(seasonsM) > 0 and len(seasonsD) == 0:  # M
        for i in range(len(indices)):
            if datesM[i] in seasonsM:
                indices[i] = 1

    elif len(seasonsY) == 0 and len(seasonsM) > 0 and len(seasonsD) == len(seasonsM):  # M, D
        for j in range(len(seasonsM)):
            for i in range(len(indices)):
                if datesM[i] == seasonsM[j] and datesD[i] in seasonsD[j]:
                    indices[i] = 1

    elif len(seasonsY) > 0 and len(seasonsM) == 0 and len(seasonsD) == 0:  # Y
        for i in range(len(indices)):
            if datesY[i] in seasonsY:
                indices[i] = 1

    elif len(seasonsY) > 0 and len(seasonsM) == 0 and len(seasonsD) == len(seasonsY):  # Y, D
        for j in range(len(seasonsY)):
            for i in range(len(indices)):
                if datesY[i] == seasonsY[j] and datesD[i] in seasonsD[j]:
                    indices[i] = 1

    elif len(seasonsY) > 0 and len(seasonsM) == len(seasonsY) and len(seasonsD) == 0:  # Y, M
        for j in range(len(seasonsY)):
            for i in range(len(indices)):
                if datesY[i] == seasonsY[j] and datesM[i] in seasonsM[j]:
                    indices[i] = 1

    elif len(seasonsY) == len(seasonsM) == len(seasonsD) > 0:  # Y, M, D
        same_length = True
        for i in range(len(seasonsM)):
            if len(seasonsM[i]) != len(seasonsD[i]):
                same_length = False

        if same_length == True:
            for k in range(len(seasonsY)):
                for j in range(len(seasonsM)):
                    for i in range(len(indices)):
                        if datesY[i] == seasonsY[k] and datesM[i] in seasonsM[k] and datesD[i] in seasonsD[k][j]:
                            indices[i] = 1
        else:
            print_log(f"{currentFuncName()}: Error: Bad input date format")

    else:
        print_log(f"{currentFuncName()}: Error: Bad input date format")

    return [i for i in range(len(indices)) if indices[i] == 1]


def prepare_data_train(source_path, dest_path, dest_path_train_file, dest_path_test_file, dest_path_validate_file, config_yaml):
    print_log(f"{currentFuncName()}:")
    # csv_data_to_one_file
    make_raw_csv_data(source_path, dest_path,
                      eval(config_yaml['all_data_models']),
                      eval(config_yaml['all_data_sources']),
                      eval(config_yaml['use_columns']) + eval(config_yaml['ORP_list']),
                      config_yaml['forecast_time'])
    make_raw_csv_data(source_path, dest_path,
                      config_yaml['measurements'],
                      eval(config_yaml['all_data_sources']),
                      eval(config_yaml['use_columns']) + eval(config_yaml['ORP_list']))

    # load_csv_files
    d_source_path_list = list()
    d_headers = list()
    for dm in eval(config_yaml['all_data_models']):
        for ds in eval(config_yaml['all_data_sources']):
            d_headers.append(dm + "__" + ds)
            d_source_path_list.append(dest_path + "/" + dm + "__" + ds + ".csv")

    m_source_path_list = list()
    m_headers = list()
    for ds in eval(config_yaml['all_data_sources']):
        m_headers.append(config_yaml['measurements'] + "__" + ds)
        m_source_path_list.append(dest_path + "/" + config_yaml['measurements'] + "__" + ds + ".csv")

    d_files, d_headers = load_csv_files(d_source_path_list, config_yaml, d_headers)
    m_files, m_headers = load_csv_files(m_source_path_list, config_yaml, m_headers)

    # data to classes
    threshold, val1, val2 = eval(config_yaml['threshold_value'])
    d_files = values_in_df_to_classes(d_files, eval(config_yaml['ORP_list']), threshold, val1, val2)
    m_files = values_in_df_to_classes(m_files, eval(config_yaml['ORP_list']), threshold, val1, val2)

    # merge_csv_dates
    d_files_out, m_files_out = merge_csv_files(d_files, m_files, config_yaml['time_tolerance'])

    # indices for train, test and validation
    train_i = get_proper_dates_indices(d_files_out[0]['timestamp'] / 1000, config_yaml['train']['seasons'])
    test_i = get_proper_dates_indices(d_files_out[0]['timestamp'] / 1000, config_yaml['test']['seasons'])
    val_i = get_proper_dates_indices(d_files_out[0]['timestamp'] / 1000, config_yaml['validate']['seasons'])

    train_d = [d_files_out[i].iloc[train_i].reset_index(drop=True) for i in range(len(d_files_out))]
    train_m = [m_files_out[i].iloc[train_i].reset_index(drop=True) for i in range(len(m_files_out))]
    test_d = [d_files_out[i].iloc[test_i].reset_index(drop=True) for i in range(len(d_files_out))]
    test_m = [m_files_out[i].iloc[test_i].reset_index(drop=True) for i in range(len(m_files_out))]
    val_d = [d_files_out[i].iloc[val_i].reset_index(drop=True) for i in range(len(d_files_out))]
    val_m = [m_files_out[i].iloc[val_i].reset_index(drop=True) for i in range(len(m_files_out))]
    headers_d = [[d_headers[i], ] * len(d_files_out[i].columns) for i in range(len(d_headers))]
    headers_m = [[m_headers[i], ] * len(m_files_out[i].columns) for i in range(len(m_headers))]
    headers = list()
    for i in range(len(headers_d)):
        headers = headers + headers_d[i]
    for i in range(len(headers_m)):
        headers = headers + headers_m[i]

    # save split data
    train = pd.concat(train_d + train_m, axis=1)
    train.columns = [headers, train.columns]
    train.to_csv(dest_path_train_file, index=False)

    test = pd.concat(test_d + test_m, axis=1)
    test.columns = [headers, test.columns]
    test.to_csv(dest_path_test_file, index=False)

    val = pd.concat(val_d + val_m, axis=1)
    val.columns = [headers, val.columns]
    val.to_csv(dest_path_validate_file, index=False)


def prepare_data_test(source_path, dest_path, dest_path_test_file, config_yaml):
    print_log(f"{currentFuncName()}:")
    # csv_data_to_one_file
    make_raw_csv_data(source_path, dest_path,
                      eval(config_yaml['all_data_models']),
                      eval(config_yaml['all_data_sources']),
                      eval(config_yaml['use_columns']) + eval(config_yaml['ORP_list']),
                      config_yaml['forecast_time'])
    make_raw_csv_data(source_path, dest_path,
                      config_yaml['measurements'],
                      eval(config_yaml['all_data_sources']),
                      eval(config_yaml['use_columns']) + eval(config_yaml['ORP_list']))

    # load_csv_files
    d_source_path_list = list()
    d_headers = list()
    for dm in eval(config_yaml['all_data_models']):
        for ds in eval(config_yaml['all_data_sources']):
            d_headers.append(dm + "__" + ds)
            d_source_path_list.append(dest_path + "/" + dm + "__" + ds + ".csv")

    m_source_path_list = list()
    m_headers = list()
    for ds in eval(config_yaml['all_data_sources']):
        m_headers.append(config_yaml['measurements'] + "__" + ds)
        m_source_path_list.append(dest_path + "/" + config_yaml['measurements'] + "__" + ds + ".csv")

    d_files, d_headers = load_csv_files(d_source_path_list, config_yaml, d_headers)
    m_files, m_headers = load_csv_files(m_source_path_list, config_yaml, m_headers)

    # data to classes
    threshold, val1, val2 = eval(config_yaml['threshold_value'])
    d_files = values_in_df_to_classes(d_files, eval(config_yaml['ORP_list']), threshold, val1, val2)
    m_files = values_in_df_to_classes(m_files, eval(config_yaml['ORP_list']), threshold, val1, val2)

    # merge_csv_dates
    d_files_out, m_files_out = merge_csv_files(d_files, m_files, config_yaml['time_tolerance'])

    # indices for train, test and validation
    test_i = get_proper_dates_indices(d_files_out[0]['timestamp'] / 1000, config_yaml['test']['seasons'])

    test_d = [d_files_out[i].iloc[test_i].reset_index(drop=True) for i in range(len(d_files_out))]
    test_m = [m_files_out[i].iloc[test_i].reset_index(drop=True) for i in range(len(m_files_out))]
    headers_d = [[d_headers[i], ] * len(d_files_out[i].columns) for i in range(len(d_headers))]
    headers_m = [[m_headers[i], ] * len(m_files_out[i].columns) for i in range(len(m_headers))]
    headers = list()
    for i in range(len(headers_d)):
        headers = headers + headers_d[i]
    for i in range(len(headers_m)):
        headers = headers + headers_m[i]

    # save split data
    test = pd.concat(test_d + test_m, axis=1)
    test.columns = [headers, test.columns]
    test.to_csv(dest_path_test_file, index=False)


def prepare_data_predict(source_path, dest_path, dest_path_predict_file, config_yaml):
    print_log(f"{currentFuncName()}:")
    # csv_data_to_one_file
    make_raw_csv_data(source_path, dest_path,
                      eval(config_yaml['all_data_models']),
                      eval(config_yaml['all_data_sources']),
                      eval(config_yaml['use_columns']) + eval(config_yaml['ORP_list']),
                      config_yaml['forecast_time'])

    # load_csv_files
    d_source_path_list = list()
    d_headers = list()
    for dm in eval(config_yaml['all_data_models']):
        for ds in eval(config_yaml['all_data_sources']):
            d_headers.append(dm + "__" + ds)
            d_source_path_list.append(dest_path + "/" + dm + "__" + ds + ".csv")
        
    print_log(f"load_csv_files({d_source_path_list}, {config_yaml}, {d_headers})")
    d_files, d_headers = load_csv_files(d_source_path_list, config_yaml, d_headers)

    # data to classes
    threshold, val1, val2 = eval(config_yaml['threshold_value'])
    d_files = values_in_df_to_classes(d_files, eval(config_yaml['ORP_list']), threshold, val1, val2)

    # merge_csv_dates
    d_files_out, m_files_out = merge_csv_files(d_files, [], config_yaml['time_tolerance'])

    # indices for train, test and validation
    predict_i = get_proper_dates_indices(d_files_out[0]['timestamp'] / 1000, config_yaml['predict']['seasons'])

    predict_d = [d_files_out[i].iloc[predict_i].reset_index(drop=True) for i in range(len(d_files_out))]
    headers_d = [[d_headers[i], ] * len(d_files_out[i].columns) for i in range(len(d_headers))]
    headers = list()

    for i in range(len(headers_d)):
        headers = headers + headers_d[i]

    # save split data
    predict = pd.concat(predict_d, axis=1)
    predict.columns = [headers, predict.columns]
    predict.to_csv(dest_path_predict_file, index=False)

