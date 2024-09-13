# -*- coding: utf-8 -*-
"""
Feature building
"""

import os
import sys
# import project config.py
# from .. import config as cfg
from .. import config_layout as cly
import yaml
import pandas as pd
import numpy as np
import ast
from datetime import datetime


def currentFuncName(n=0):
    return sys._getframe(n + 1).f_code.co_name


def print_log(log_line, verbose=True, time_stamp=True, log_file=cly.LOG_FILE_PATH):
    tm = ""
    if time_stamp:
        tm = datetime.now().strftime("%Y-%m-%d %H:%M:%S: ")
    if verbose:
        if log_file is None:
            print(tm + log_line)
        else:
            print(tm + log_line)
            with open(log_file, 'a') as file:
                file.write(tm + log_line + "\n")


def load_config_yaml(pathYaml, part=""):
    try:
        print_log(f"running {currentFuncName()}")
        with open(pathYaml) as yamlFile:
            config = yaml.safe_load(yamlFile)
        if part == "":
            return config
        else:
            return config[part]
    except Exception as err:
        print_log(f"{currentFuncName()}: Unexpected {err=}, {type(err)=}")


def csv_data_to_one_file(source_path, dest_path, use_columns, forecast_time=None):
    try:
        print_log(f"running {currentFuncName()}")
        print_log(f"source_path == {source_path}")
        data_files = os.listdir(source_path)
        data_files = [f for f in data_files if os.path.isfile(source_path + '/' + f)]
        df_csv_append = pd.DataFrame()
        for data_file in data_files:
            if os.path.isfile(source_path + '/' + data_file):
                print_log(f"{currentFuncName()}: Reading file: {source_path}/{data_file} ...")
                df = pd.read_csv(source_path + '/' + data_file, na_filter=False)
                use_columns_2 = [use_columns[i] for i in range(len(use_columns)) if use_columns[i] in df.columns]
                if forecast_time is not None:
                    forecast_time_indices = [i for i in range(len(df['forecast']))
                                             if df['forecast'][i] == forecast_time]
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
            if os.path.isfile(dest_path):
                print_log(f"{currentFuncName()}: df_csv_append.to_csv({dest_path},mode='a',index=False,header=False)")
                df_csv_append.to_csv(dest_path, mode='a', index=False, header=False)
            else:
                print_log(f"{currentFuncName()}: df_csv_append.to_csv({dest_path}, index=False)")
                df_csv_append.to_csv(dest_path, index=False)
        else:
            print_log(f"{currentFuncName()}: Warning: len(df_csv_append) == 0")
    except Exception as err:
        print_log(f"{currentFuncName()}: Unexpected {err=}, {type(err)=}")


def make_raw_csv_data(source_path, dest_path, data_sources, file_types, use_columns, forecast_time=None):
    try:
        print_log(f"running {currentFuncName()}")
        if not isinstance(data_sources, list) and not isinstance(data_sources, tuple):
            data_sources = (data_sources,)
        if not isinstance(file_types, list) and not isinstance(file_types, tuple):
            file_types = (file_types,)
        for ds in data_sources:
            for ft in file_types:
                csv_data_to_one_file(source_path + "/" + ds + "/" + ft,
                                     dest_path + "/" + ds + "__" + ft + ".csv",
                                     use_columns, forecast_time)
    except Exception as err:
        print_log(f"{currentFuncName()}: Unexpected {err=}, {type(err)=}")


def load_csv_files(source_path_list, config_yaml, header_list, nan_to_zero=True):
    try:
        print_log(f"running {currentFuncName()}")
        if not isinstance(source_path_list, list) and not isinstance(source_path_list, tuple):
            source_path_list = (source_path_list,)
        output_df = list()
        output_header = list()
        j = 0
        for i in range(len(source_path_list)):
            if os.path.isfile(source_path_list[i]):
                output_df.append(pd.DataFrame())
                output_df[j] = pd.read_csv(source_path_list[i], na_filter=False)
                # Fill nan with zeros
                if nan_to_zero:
                    print_log("Fill nan with zeros")
                    for col in output_df[j].columns:
                        output_df[j][col] = output_df[j][col].fillna(0)
                output_header.append(header_list[i])
                j = j + 1
        return output_df, output_header
    except Exception as err:
        print_log(f"{currentFuncName()}: Unexpected {err=}, {type(err)=}")


def values_in_df_to_classes(dfs, column_list, threshold, val1, val2=0):
    try:
        print_log(f"running {currentFuncName()}")
        if not isinstance(dfs, list) and not isinstance(dfs, tuple):
            dfs = (dfs,)
        threshold = (-np.Inf,) + threshold + (np.Inf,)
        for i in range(len(dfs)):
            for column in column_list:
                dfs[i][column] = np.float64(dfs[i][column])
                dfs[i][column][np.isnan(dfs[i][column])] = 0
                # dealing with not only two classes
                # dfs[i][column] = np.where(dfs[i][column] >= threshold, val1, val2)
                df_tmp = np.where(dfs[i][column] >= 0, 0, 0)
                for j in range(len(threshold)-1):
                    print_log(f"threshold[{j}] == {threshold[j]}")
                    print_log(f"threshold[{j+1}] == {threshold[j+1]}")
                    print_log(f"val1[{j}] == {val1[j]}")
                    df_tmp = df_tmp + np.where((threshold[j] < dfs[i][column]) &
                                               (dfs[i][column] <= threshold[j+1]), val1[j], 0)
                    print_log(f"df_tmp = df_tmp + np.where((threshold[{j}] < dfs[{i}][column]) &")
                    print_log(f"                           (dfs[{i}][column] <= threshold[{j+1}]), {val1[j]}, 0)")
                dfs[i][column] = df_tmp
        return dfs
    except Exception as err:
        print_log(f"{currentFuncName()}: Unexpected {err=}, {type(err)=}")


def interval(t, dt):
    return (t - dt, t + dt)


def merge_csv_files(d1_files, d2_files, m_files, tolerance):
    try:
        print_log(f"running {currentFuncName()}")
        if not isinstance(d1_files, list) and not isinstance(d1_files, tuple):
            d1_files = (d1_files,)
        if not isinstance(d2_files, list) and not isinstance(d2_files, tuple):
            d2_files = (d2_files,)
        if not isinstance(m_files, list) and not isinstance(m_files, tuple):
            m_files = (m_files,)

        forecast = d1_files[0]['forecast'][0] * 60 * 1000
        print_log(f"forecast (milliseconds) = {forecast}")

        # sort timestamps
        print_log("Sort timestamps for d1_files")
        for i in range(len(d1_files)):
            d1_files[i] = d1_files[i].sort_values('timestamp')
            d1_files[i]['timestamp'] = d1_files[i]['timestamp'] + forecast
        print_log("Sort timestamps for d2_files")
        for i in range(len(d2_files)):
            d2_files[i] = d2_files[i].sort_values('timestamp')
            d2_files[i]['timestamp'] = d2_files[i]['timestamp'] + forecast
        print_log("Sort timestamps for m_files")
        for i in range(len(m_files)):
            m_files[i] = m_files[i].sort_values('timestamp')
            m_files[i]['timestamp'] = m_files[i]['timestamp'] + forecast

        # getting timestamp and indices
        d1_timestamps = list()
        d2_timestamps = list()
        m_timestamps = list()
        j = 0
        print_log("Getting timestamp and indices for d1_files")
        for i in range(len(d1_files)):
            d1_timestamps.append(d1_files[i].reset_index().sort_values(by=['timestamp', 'index']))
            d1_timestamps[i] = d1_timestamps[i][['timestamp', 'index']]
            d1_timestamps[i].columns = ['timestamp', 'index_d1_' + str(j)]
            j = j + 1
        print_log("Getting timestamp and indices for d2_files")
        for i in range(len(m_files)):
            m_timestamps.append(m_files[i].reset_index().sort_values(by=['timestamp', 'index']))
            m_timestamps[i] = m_timestamps[i][['timestamp', 'index']]
            m_timestamps[i].columns = ['timestamp', 'index_m_' + str(j)]
            j = j + 1
        print_log("Getting timestamp and indices for m_files")
        for i in range(len(d2_files)):
            d2_timestamps.append(d2_files[i].reset_index().sort_values(by=['timestamp', 'index']))
            d2_timestamps[i] = d2_timestamps[i][['timestamp', 'index']]
            d2_timestamps[i].columns = ['timestamp', 'index_d2_' + str(j)]
            j = j + 1
            # adding auxiliary row to d2 files - all zeros
            print_log("Adding auxiliary row to d2_files")
            d2_files[i].loc[len(d2_files[i].index)] = len(d2_files[i].columns) * [0]

        # merging
        print_log("Merging d1_timestamps")
        timestamps_indices = d1_timestamps[0]
        for i in range(1, len(d1_timestamps)):
            timestamps_indices = pd.merge_asof(timestamps_indices, d1_timestamps[i], on="timestamp",
                                               tolerance=tolerance, direction='nearest')
        print_log("Merging m_timestamps")
        for i in range(len(m_timestamps)):
            timestamps_indices = pd.merge_asof(timestamps_indices, m_timestamps[i], on="timestamp",
                                               tolerance=tolerance, direction='nearest')
        print_log("timestamps_inidces.dropna()")
        timestamps_indices = timestamps_indices.dropna()

        print_log("Merging d2_timestamps")
        for i in range(len(d2_timestamps)):
            timestamps_indices = pd.merge_asof(timestamps_indices, d2_timestamps[i], on="timestamp",
                                               tolerance=tolerance, direction='nearest')
        j = 1 + len(d1_files) + len(m_files)
        for i in range(len(d2_files)):
            print_log(f"{i}")
            print_log(f"{i+j}")
            print_log(f"{timestamps_indices.columns[j+i]}")
            print_log(f"{len(d2_files[i].index)-1}")
            timestamps_indices[timestamps_indices.columns[j + i]] = timestamps_indices[
                timestamps_indices.columns[j + i]].fillna(len(d2_files[i].index) - 1)

        j = 1
        for i in range(len(d1_files)):
            print_log(f"{i}")
            print_log(f"{i+j}")
            print_log(f"{timestamps_indices.columns[i+j]}")
            print_log(f"{timestamps_indices[timestamps_indices.columns[i+j]]}")
            d1_files[i] = d1_files[i].iloc[timestamps_indices[timestamps_indices.columns[i + j]]]
        j = j + len(d1_files)
        for i in range(len(m_files)):
            print_log(f"{i}")
            m_files[i] = m_files[i].iloc[timestamps_indices[timestamps_indices.columns[i + j]]]
        j = j + len(m_files)
        for i in range(len(d2_files)):
            print_log(f"{i}")
            d2_files[i] = d2_files[i].iloc[timestamps_indices[timestamps_indices.columns[i + j]]]

        return d1_files, d2_files, m_files
    except Exception as err:
        print_log(f"{currentFuncName()}: Unexpected {err=}, {type(err)=}")


def get_proper_dates_indices(timestamps, config_yaml_date_settings):
    try:
        print_log(f"running {currentFuncName()}")
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
        seasonsY = ast.literal_eval(seasons['years'])
        seasonsM = ast.literal_eval(seasons['months'])
        seasonsD = ast.literal_eval(seasons['days'])

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

            if same_length is True:
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
    except Exception as err:
        print_log(f"{currentFuncName()}: Unexpected {err=}, {type(err)=}")


def prepare_data_train(source_path, dest_path, dest_path_train_file, dest_path_test_file,
                       dest_path_validate_file, config_yaml):
    try:
        print_log(f"running {currentFuncName()}")
        # csv_data_to_one_file
        ORPs = tuple()
        for i in range(len(config_yaml['dataset'])):
            ORPs = ORPs + ast.literal_eval(config_yaml['dataset'][i]['ORP_list'])
        print_log(f"ORPs == {ORPs}")

        make_raw_csv_data(source_path, dest_path,
                          ast.literal_eval(config_yaml['input_data_d1']) +
                          ast.literal_eval(config_yaml['input_data_d2']),
                          ast.literal_eval(config_yaml['input_data_sources']),
                          ast.literal_eval(config_yaml['use_columns']) + ORPs,
                          config_yaml['forecast_time'])
        make_raw_csv_data(source_path, dest_path,
                          config_yaml['measurements'],
                          ast.literal_eval(config_yaml['input_data_sources']),
                          ast.literal_eval(config_yaml['use_columns']) + ORPs)

        # load_csv_files
        d1_source_path_list = list()
        d1_headers = list()
        for dm in ast.literal_eval(config_yaml['input_data_d1']):
            for ds in ast.literal_eval(config_yaml['input_data_sources']):
                d1_headers.append(dm + "__" + ds)
                d1_source_path_list.append(dest_path + "/" + dm + "__" + ds + ".csv")

        d2_source_path_list = list()
        d2_headers = list()
        for dm in ast.literal_eval(config_yaml['input_data_d2']):
            for ds in ast.literal_eval(config_yaml['input_data_sources']):
                d2_headers.append(dm + "__" + ds)
                d2_source_path_list.append(dest_path + "/" + dm + "__" + ds + ".csv")

        m_source_path_list = list()
        m_headers = list()
        for ds in ast.literal_eval(config_yaml['input_data_sources']):
            m_headers.append(config_yaml['measurements'] + "__" + ds)
            m_source_path_list.append(dest_path + "/" + config_yaml['measurements'] + "__" + ds + ".csv")

        d1_files, d1_headers = load_csv_files(d1_source_path_list, config_yaml, d1_headers, nan_to_zero=True)
        d2_files, d2_headers = load_csv_files(d2_source_path_list, config_yaml, d2_headers, nan_to_zero=True)
        m_files, m_headers = load_csv_files(m_source_path_list, config_yaml, m_headers, nan_to_zero=True)

        # merge_csv_dates
        d1_files, d2_files, m_files = merge_csv_files(d1_files, d2_files, m_files, config_yaml['time_tolerance'])
        print_log(f"d1_files[0].columns == {d1_files[0].columns}")
        print_log(f"d2_files[0].columns == {d2_files[0].columns}")
        print_log(f"m_files[0].columns == {m_files[0].columns}")

        d1_files_tmp, d2_files_tmp, m_files_tmp = list(), list(), list()
        print_log("d1 files, ORP selection")
        for i in range(len(d1_files)):
            print_log(f"i == {i}")
            for j in range(len(config_yaml['dataset'])):
                print_log(f"j == {j}")
                use_columns = list(ast.literal_eval(config_yaml['use_columns']) +
                                   ast.literal_eval(config_yaml['dataset'][j]['ORP_list']))
                print_log(f"use_columns == {use_columns}")
                if j == 0:
                    print_log(f"d1_files_tmp.append({d1_files[i][use_columns]})")
                    d1_files_tmp.append(d1_files[i][use_columns])
                else:
                    print_log(f"tmp_df = {d1_files[i][use_columns]}")
                    tmp_df = d1_files[i][use_columns]
                    print_log(f"tmp_df.columns = {list(d1_files_tmp[0].columns)}")
                    tmp_df.columns = list(d1_files_tmp[0].columns)
                    print_log(f"d1_files_tmp[{i}] = pd.concat([d1_files_tmp[{i}], tmp_df], ignore_index=True, axis=0)")
                    d1_files_tmp[i] = pd.concat([d1_files_tmp[i], tmp_df], ignore_index=True, axis=0)
                    print_log(f"d1_files_tmp[{i}] == {d1_files_tmp[i]}")
                print_log("")

        print_log("d2 files, ORP selection")
        for i in range(len(d2_files)):
            print_log(f"i == {i}")
            for j in range(len(config_yaml['dataset'])):
                print_log(f"j == {j}")
                use_columns = list(ast.literal_eval(config_yaml['use_columns']) +
                                   ast.literal_eval(config_yaml['dataset'][j]['ORP_list']))
                print_log(f"use_columns == {use_columns}")
                if j == 0:
                    print_log(f"d2_files_tmp.append({d2_files[i][use_columns]})")
                    d2_files_tmp.append(d2_files[i][use_columns])
                else:
                    print_log(f"tmp_df = {d2_files[i][use_columns]}")
                    tmp_df = d2_files[i][use_columns]
                    print_log(f"tmp_df.columns = {d2_files_tmp[0].columns}")
                    tmp_df.columns = d2_files_tmp[0].columns
                    print_log(f"d2_files_tmp[{i}] = pd.concat([d2_files_tmp[{i}], tmp_df], ignore_index=True, axis=0)")
                    d2_files_tmp[i] = pd.concat([d2_files_tmp[i], tmp_df], ignore_index=True, axis=0)
                    print_log(f"d2_files_tmp[{i}] == {d2_files_tmp[i]}")
                print_log("")

        print_log("m files, ORP selection")
        for i in range(len(m_files)):
            print_log(f"i == {i}")
            for j in range(len(config_yaml['dataset'])):
                print_log(f"j == {j}")
                use_columns = list(ast.literal_eval(config_yaml['use_columns']) +
                                   ast.literal_eval(config_yaml['dataset'][j]['ORP_list']))
                use_columns = [x for x in use_columns if x in m_files[i].columns]
                if j == 0:
                    print_log(f"m_files_tmp.append({m_files[i][use_columns]})")
                    m_files_tmp.append(m_files[i][use_columns])
                else:
                    print_log(f"tmp_df = {m_files[i][use_columns]}")
                    tmp_df = m_files[i][use_columns]
                    print_log(f"tmp_df.columns = {m_files_tmp[0].columns}")
                    tmp_df.columns = m_files_tmp[0].columns
                    print_log(f"m_files_tmp[{i}] = pd.concat([m_files_tmp[{i}], tmp_df], ignore_index=True, axis=0)")
                    m_files_tmp[i] = pd.concat([m_files_tmp[i], tmp_df], ignore_index=True, axis=0)
                    print_log(f"m_files_tmp[{i}] == {m_files_tmp[i]}")
        print_log("d1_files, d2_files, m_files = d1_files_tmp, d2_files_tmp, m_files_tmp")
        d1_files, d2_files, m_files = d1_files_tmp, d2_files_tmp, m_files_tmp

        # data to classes
        # threshold, val1, val2 = ast.literal_eval(config_yaml['threshold_value'])
        threshold, val1 = ast.literal_eval(config_yaml['threshold_value'])
        val2 = 0
        d1_files_out = values_in_df_to_classes(d1_files, ast.literal_eval(config_yaml['dataset'][0]['ORP_list']),
                                               threshold, val1, val2)
        d2_files_out = values_in_df_to_classes(d2_files, ast.literal_eval(config_yaml['dataset'][0]['ORP_list']),
                                               threshold, val1, val2)
        m_files_out = values_in_df_to_classes(m_files, ast.literal_eval(config_yaml['dataset'][0]['ORP_list']),
                                              threshold, val1, val2)

        # reset indices
        for i in range(len(d1_files_out)):
            d1_files_out[i] = d1_files_out[i].reset_index(drop=True)
        for i in range(len(d2_files_out)):
            d2_files_out[i] = d2_files_out[i].reset_index(drop=True)
        for i in range(len(m_files_out)):
            m_files_out[i] = m_files_out[i].reset_index(drop=True)

        # omit all zeros
        all_zeros = [True] * len(d1_files_out[0])
        all_zeros_timestamp = [True] * len(d1_files_out[0])
        use_columns = list(ast.literal_eval(config_yaml['use_columns']))
        print_log(f"use_columns == {use_columns}")
        for i in range(len(d1_files_out)):
            print_log("d1_files_out")
            cls = [clmn for clmn in d1_files_out[i].columns if clmn not in use_columns]
            tmp = d1_files_out[i][cls]
            all_zeros = np.logical_and(all_zeros, (tmp.T == 0).all())

        for i in range(len(d2_files_out)):
            print_log("d2_files_out")
            cls = [clmn for clmn in d2_files_out[i].columns if clmn not in use_columns]
            tmp = d2_files_out[i][cls]
            all_zeros = np.logical_and(all_zeros, (tmp.T == 0).all())
            all_zeros_timestamp = np.logical_and(all_zeros_timestamp, d2_files_out[i]['timestamp'] == 0)

        for i in range(len(m_files_out)):
            print_log("m_files_out")
            cls = [clmn for clmn in m_files_out[i].columns if clmn not in use_columns]
            tmp = m_files_out[i][cls]
            all_zeros = np.logical_and(all_zeros, (tmp.T == 0).all())

        all_zeros2 = np.logical_and(np.random.choice(2, size=len(all_zeros), p=(0.1, 0.9)).astype(bool),
                                    all_zeros)

        for i in range(len(d1_files_out)):
            print_log("d1_files_out")
            d1_files_out[i] = d1_files_out[i][~np.logical_or(np.logical_and(all_zeros, all_zeros_timestamp),
                                                             all_zeros2)]
        for i in range(len(d2_files_out)):
            print_log("d2_files_out")
            d2_files_out[i] = d2_files_out[i][~np.logical_or(np.logical_and(all_zeros, all_zeros_timestamp),
                                                             all_zeros2)]
        for i in range(len(m_files_out)):
            print_log("m_files_out")
            m_files_out[i] = m_files_out[i][~np.logical_or(np.logical_and(all_zeros, all_zeros_timestamp), all_zeros2)]

        # indices for train, test and validation
        train_i = get_proper_dates_indices(d1_files_out[0]['timestamp'] / 1000, config_yaml['train']['seasons'])
        test_i = get_proper_dates_indices(d1_files_out[0]['timestamp'] / 1000, config_yaml['test']['seasons'])
        val_i = get_proper_dates_indices(d1_files_out[0]['timestamp'] / 1000, config_yaml['validate']['seasons'])

        train_d1 = [d1_files_out[i].iloc[train_i].reset_index(drop=True) for i in range(len(d1_files_out))]
        train_d2 = [d2_files_out[i].iloc[train_i].reset_index(drop=True) for i in range(len(d2_files_out))]
        train_m = [m_files_out[i].iloc[train_i].reset_index(drop=True) for i in range(len(m_files_out))]
        test_d1 = [d1_files_out[i].iloc[test_i].reset_index(drop=True) for i in range(len(d1_files_out))]
        test_d2 = [d2_files_out[i].iloc[test_i].reset_index(drop=True) for i in range(len(d2_files_out))]
        test_m = [m_files_out[i].iloc[test_i].reset_index(drop=True) for i in range(len(m_files_out))]
        val_d1 = [d1_files_out[i].iloc[val_i].reset_index(drop=True) for i in range(len(d1_files_out))]
        val_d2 = [d2_files_out[i].iloc[val_i].reset_index(drop=True) for i in range(len(d2_files_out))]
        val_m = [m_files_out[i].iloc[val_i].reset_index(drop=True) for i in range(len(m_files_out))]
        headers_d1 = [[d1_headers[i], ] * len(d1_files_out[i].columns) for i in range(len(d1_headers))]
        headers_d2 = [[d2_headers[i], ] * len(d2_files_out[i].columns) for i in range(len(d2_headers))]
        headers_m = [[m_headers[i], ] * len(m_files_out[i].columns) for i in range(len(m_headers))]
        headers = list()
        for i in range(len(headers_d1)):
            headers = headers + headers_d1[i]
        for i in range(len(headers_d2)):
            headers = headers + headers_d2[i]
        for i in range(len(headers_m)):
            headers = headers + headers_m[i]

        # save split data
        train = pd.concat(train_d1 + train_d2 + train_m, axis=1)
        train.columns = [headers, train.columns]
        train.to_csv(dest_path_train_file, index=False)

        test = pd.concat(test_d1 + test_d2 + test_m, axis=1)
        test.columns = [headers, test.columns]
        test.to_csv(dest_path_test_file, index=False)

        val = pd.concat(val_d1 + val_d2 + val_m, axis=1)
        val.columns = [headers, val.columns]
        val.to_csv(dest_path_validate_file, index=False)
    except Exception as err:
        print_log(f"{currentFuncName()}: Unexpected {err=}, {type(err)=}")


def prepare_data_test(source_path, dest_path, dest_path_test_file, config_yaml):
    try:
        print_log(f"running {currentFuncName()}")
        # csv_data_to_one_file
        for i in range(len(config_yaml['dataset'])):
            make_raw_csv_data(source_path, dest_path,
                              ast.literal_eval(config_yaml['all_data_models']),
                              ast.literal_eval(config_yaml['all_data_sources']),
                              ast.literal_eval(config_yaml['use_columns']) +
                              ast.literal_eval(config_yaml['dataset'][i]['ORP_list']),
                              config_yaml['forecast_time'])
            make_raw_csv_data(source_path, dest_path,
                              config_yaml['measurements'],
                              ast.literal_eval(config_yaml['all_data_sources']),
                              ast.literal_eval(config_yaml['use_columns']) +
                              ast.literal_eval(config_yaml['dataset'][i]['ORP_list']))

        # load_csv_files
        d_source_path_list = list()
        d_headers = list()
        for dm in ast.literal_eval(config_yaml['all_data_models']):
            for ds in ast.literal_eval(config_yaml['all_data_sources']):
                d_headers.append(dm + "__" + ds)
                d_source_path_list.append(dest_path + "/" + dm + "__" + ds + ".csv")

        m_source_path_list = list()
        m_headers = list()
        for ds in ast.literal_eval(config_yaml['all_data_sources']):
            m_headers.append(config_yaml['measurements'] + "__" + ds)
            m_source_path_list.append(dest_path + "/" + config_yaml['measurements'] + "__" + ds + ".csv")

        d_files, d_headers = load_csv_files(d_source_path_list, config_yaml, d_headers)
        m_files, m_headers = load_csv_files(m_source_path_list, config_yaml, m_headers)

        # data to classes
        # threshold, val1, val2 = ast.literal_eval(config_yaml['threshold_value'])
        threshold, val1 = ast.literal_eval(config_yaml['threshold_value'])
        val2 = 0
        d_files = values_in_df_to_classes(d_files, ast.literal_eval(config_yaml['ORP_list']), threshold, val1, val2)
        m_files = values_in_df_to_classes(m_files, ast.literal_eval(config_yaml['ORP_list']), threshold, val1, val2)

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
    except Exception as err:
        print_log(f"{currentFuncName()}: Unexpected {err=}, {type(err)=}")


def prepare_data_predict(source_path, dest_path, dest_path_predict_file, config_yaml):
    try:
        print_log(f"running {currentFuncName()}")
        # csv_data_to_one_file
        ORPs = tuple()
        for i in range(len(config_yaml['dataset'])):
            ORPs = ORPs + ast.literal_eval(config_yaml['dataset'][i]['ORP_list'])
        print_log(f"ORPs == {ORPs}")

        make_raw_csv_data(source_path, dest_path,
                          ast.literal_eval(config_yaml['input_data_d1']) +
                          ast.literal_eval(config_yaml['input_data_d2']),
                          ast.literal_eval(config_yaml['input_data_sources']),
                          ast.literal_eval(config_yaml['use_columns']) + ORPs,
                          config_yaml['forecast_time'])

        # load_csv_files
        d1_source_path_list = list()
        d1_headers = list()
        for dm in ast.literal_eval(config_yaml['input_data_d1']):
            for ds in ast.literal_eval(config_yaml['input_data_sources']):
                d1_headers.append(dm + "__" + ds)
                d1_source_path_list.append(dest_path + "/" + dm + "__" + ds + ".csv")

        d2_source_path_list = list()
        d2_headers = list()
        for dm in ast.literal_eval(config_yaml['input_data_d2']):
            for ds in ast.literal_eval(config_yaml['input_data_sources']):
                d2_headers.append(dm + "__" + ds)
                d2_source_path_list.append(dest_path + "/" + dm + "__" + ds + ".csv")

        # m_source_path_list = list()
        # m_headers = list()
        # for ds in ast.literal_eval(config_yaml['input_data_sources']):
        #     m_headers.append(dm + "__" + ds)
        #     m_source_path_list.append(dest_path + "/" + dm + "__" + ds + ".csv")

        print_log(f"{currentFuncName()}: load_csv_files({d1_source_path_list}, {config_yaml}, {d1_headers})")
        d1_files, d1_headers = load_csv_files(d1_source_path_list, config_yaml, d1_headers, nan_to_zero=True)
        print_log(f"{currentFuncName()}: load_csv_files({d2_source_path_list}, {config_yaml}, {d2_headers})")
        d2_files, d2_headers = load_csv_files(d2_source_path_list, config_yaml, d2_headers, nan_to_zero=True)
        # print_log(f"{currentFuncName()}: load_csv_files({m_source_path_list}, {config_yaml}, {m_headers})")
        # m_files, m_headers = load_csv_files(m_source_path_list, config_yaml, m_headers, nan_to_zero=True)

        # merge_csv_dates
        d1_files, d2_files, m_files = merge_csv_files(d1_files, d2_files, [], config_yaml['time_tolerance'])
        print_log(f"d1_files[0].columns == {d1_files[0].columns}")
        print_log(f"d2_files[0].columns == {d2_files[0].columns}")
        # print_log(f"m_files[0].columns == {m_files[0].columns}")

        d1_files_tmp, d2_files_tmp, m_files_tmp = list(), list(), list()
        print_log("d1 files, ORP selection")
        for i in range(len(d1_files)):
            print_log(f"i == {i}")
            for j in range(len(config_yaml['dataset'])):
                print_log(f"j == {j}")
                use_columns = list(ast.literal_eval(config_yaml['use_columns']) +
                                   ast.literal_eval(config_yaml['dataset'][j]['ORP_list']))
                print_log(f"use_columns == {use_columns}")
                if j == 0:
                    print_log(f"d1_files_tmp.append({d1_files[i][use_columns]})")
                    d1_files_tmp.append(d1_files[i][use_columns])
                else:
                    print_log(f"tmp_df = {d1_files[i][use_columns]}")
                    tmp_df = d1_files[i][use_columns]
                    print_log(f"tmp_df.columns = {list(d1_files_tmp[0].columns)}")
                    tmp_df.columns = list(d1_files_tmp[0].columns)
                    print_log(f"d1_files_tmp[{i}] = pd.concat([d1_files_tmp[{i}], tmp_df], ignore_index=True, axis=0)")
                    d1_files_tmp[i] = pd.concat([d1_files_tmp[i], tmp_df], ignore_index=True, axis=0)
                    print_log(f"d1_files_tmp[{i}] == {d1_files_tmp[i]}")
                print_log("")

        print_log("d2 files, ORP selection")
        for i in range(len(d2_files)):
            print_log(f"i == {i}")
            for j in range(len(config_yaml['dataset'])):
                print_log(f"j == {j}")
                use_columns = list(ast.literal_eval(config_yaml['use_columns']) +
                                   ast.literal_eval(config_yaml['dataset'][j]['ORP_list']))
                print_log(f"use_columns == {use_columns}")
                if j == 0:
                    print_log(f"d2_files_tmp.append({d2_files[i][use_columns]})")
                    d2_files_tmp.append(d2_files[i][use_columns])
                else:
                    print_log(f"tmp_df = {d2_files[i][use_columns]}")
                    tmp_df = d2_files[i][use_columns]
                    print_log(f"tmp_df.columns = {d2_files_tmp[0].columns}")
                    tmp_df.columns = d2_files_tmp[0].columns
                    print_log(f"d2_files_tmp[{i}] = pd.concat([d2_files_tmp[{i}], tmp_df], ignore_index=True, axis=0)")
                    d2_files_tmp[i] = pd.concat([d2_files_tmp[i], tmp_df], ignore_index=True, axis=0)
                    print_log(f"d2_files_tmp[{i}] == {d2_files_tmp[i]}")
                print_log("")

        print_log("m files, ORP selection")
        for i in range(len(m_files)):
            print_log(f"i == {i}")
            for j in range(len(config_yaml['dataset'])):
                print_log(f"j == {j}")
                use_columns = list(ast.literal_eval(config_yaml['use_columns']) +
                                   ast.literal_eval(config_yaml['dataset'][j]['ORP_list']))
                use_columns = [x for x in use_columns if x in m_files[i].columns]
                if j == 0:
                    print_log(f"m_files_tmp.append({m_files[i][use_columns]})")
                    m_files_tmp.append(m_files[i][use_columns])
                else:
                    print_log(f"tmp_df = {m_files[i][use_columns]}")
                    tmp_df = m_files[i][use_columns]
                    print_log(f"tmp_df.columns = {m_files_tmp[0].columns}")
                    tmp_df.columns = m_files_tmp[0].columns
                    print_log(f"m_files_tmp[{i}] = pd.concat([m_files_tmp[{i}], tmp_df], ignore_index=True, axis=0)")
                    m_files_tmp[i] = pd.concat([m_files_tmp[i], tmp_df], ignore_index=True, axis=0)
                    print_log(f"m_files_tmp[{i}] == {m_files_tmp[i]}")
        print_log("d1_files, d2_files, m_files = d1_files_tmp, d2_files_tmp, m_files_tmp")
        d1_files, d2_files, m_files = d1_files_tmp, d2_files_tmp, m_files_tmp

        # data to classes
        # threshold, val1, val2 = ast.literal_eval(config_yaml['threshold_value'])
        threshold, val1 = ast.literal_eval(config_yaml['threshold_value'])
        val2 = 0
        d1_files_out = values_in_df_to_classes(d1_files, ast.literal_eval(config_yaml['dataset'][0]['ORP_list']),
                                               threshold, val1, val2)
        d2_files_out = values_in_df_to_classes(d2_files, ast.literal_eval(config_yaml['dataset'][0]['ORP_list']),
                                               threshold, val1, val2)
        # m_files_out = values_in_df_to_classes(m_files, ast.literal_eval(config_yaml['dataset'][0]['ORP_list']),
        #                                       threshold, val1, val2)

        # indices for train, test and validation
        print_log(f"d1_files_out == {d1_files_out}")
        print_log(f"d1_files_out[0] == {d1_files_out[0]}")
        print_log(f"d1_files_out[0]['timestamp'] == {d1_files_out[0]['timestamp']}")
        print_log(f"config_yaml['predict']['seasons'] == {config_yaml['predict']['seasons']}")
        predict_i = get_proper_dates_indices(d1_files_out[0]['timestamp'] / 1000, config_yaml['predict']['seasons'])

        predict_d1 = [d1_files_out[i].iloc[predict_i].reset_index(drop=True) for i in range(len(d1_files_out))]
        headers_d1 = [[d1_headers[i], ] * len(d1_files_out[i].columns) for i in range(len(d1_headers))]
        predict_d2 = [d2_files_out[i].iloc[predict_i].reset_index(drop=True) for i in range(len(d2_files_out))]
        headers_d2 = [[d2_headers[i], ] * len(d2_files_out[i].columns) for i in range(len(d2_headers))]
        headers = list()

        for i in range(len(headers_d1)):
            headers = headers + headers_d1[i]
        for i in range(len(headers_d2)):
            headers = headers + headers_d2[i]

        # save split data
        predict = pd.concat(predict_d1 + predict_d2, axis=1)
        predict.columns = [headers, predict.columns]
        predict.to_csv(dest_path_predict_file, index=False)
    except Exception as err:
        print_log(f"{currentFuncName()}: Unexpected {err=}, {type(err)=}")
