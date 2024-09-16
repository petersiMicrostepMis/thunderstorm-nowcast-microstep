import pandas as pd
import numpy as np
import datetime
import sys
from .. import config_layout as cly


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
            with open(log_file, 'a') as file:
                file.write(tm + log_line + "\n")


def unlist(x):
    if isinstance(x, list):
        v = []
        for u in x:
            if isinstance(u, list):
                v = v + u
            else:
                v = v + [u, ]
        return v
    else:
        return x


def unlist_all(x, max_iter=1000):
    if isinstance(x, list):
        i = 0
        while x != unlist(x) and i < max_iter:
            x = unlist(x)
            i = i + 1
    return x


def contingency_table(measure, prediction):
    print_log(f"running {currentFuncName()}")
    # print_log(f"measure == {measure}")
    # print_log(f"np.shape(measure) == {np.shape(measure)}")
    # print_log(f"np.prod(np.shape(measure) == {np.prod(np.shape(measure))}")
    # print_log(f"(np.prod(np.shape(measure)), 1) == {(np.prod(np.shape(measure)), 1)}")
    measure = np.concatenate(np.reshape(measure, (np.prod(np.shape(measure)), 1)))
    prediction = np.concatenate(np.reshape(prediction, (np.prod(np.shape(prediction)), 1)))
    # length check
    print_log(f"len(measure) == {len(measure)}, len(prediction) == {len(prediction)}")
    return pd.crosstab(measure, prediction)


def length(x):
    try:
        return len(x)
    except Exception as err:
        print_log(f"Error in {currentFuncName()}: Can't return len(x): Error {err}")
        return 0


def zero_denominator(x, y):
    if y == 0:
        return np.NaN
    else:
        return float(x / y)


def table2abcd(table, event=1, no_event=0):
    # for binary classification
    a = unlist_all(table.values[table.transpose().columns == event, table.columns == event])
    b = unlist_all(table.values[table.transpose().columns == event, table.columns == no_event])
    c = unlist_all(table.values[table.transpose().columns == no_event, table.columns == event])
    d = unlist_all(table.values[table.transpose().columns == no_event, table.columns == no_event])
    a = [int(a[0]) if length(a) > 0 else int(0)]
    b = [int(b[0]) if length(b) > 0 else int(0)]
    c = [int(c[0]) if length(c) > 0 else int(0)]
    d = [int(d[0]) if length(d) > 0 else int(0)]
    return a[0], b[0], c[0], d[0]


def metrics_ACC(table):
    # for binary classification
    print_log(f"running {currentFuncName()}")
    # a, b, c, d = table2abcd(table)
    # return zero_denominator(a + d, a + b + c + d)
    return zero_denominator(np.sum(np.diag(table)), np.sum(np.sum(table)))


def metrics_F1(table):
    # for binary classification
    print_log(f"running {currentFuncName()}")
    # a, b, c, d = table2abcd(table)
    # return zero_denominator(a, a + 0.5 * (b + c))
    F1 = list()
    for i in range(np.shape(table)[0]):
        a = table[i][i]
        row = 0
        for j in range(np.shape(table)[0]):
            row = row + table[j][i]
        bc = np.sum(table[i]) + row - 2*a
        F1.append(zero_denominator(a, a + 0.5*(bc)))
    return F1


def metrics_CSI(table):
    # for binary classification
    print_log(f"running {currentFuncName()}")
    # a, b, c, d = table2abcd(table)
    # return zero_denominator(a, a + b + c)
    CSI = list()
    for i in range(np.shape(table)[0]):
        a = table[i][i]
        row = 0
        for j in range(np.shape(table)[0]):
            row = row + table[j][i]
        bc = np.sum(table[i]) + row - 2*a
        CSI.append(zero_denominator(a, a + bc))
    return CSI


def metrics_POD(table):
    # for binary classification
    print_log(f"running {currentFuncName()}")
    # a, b, c, d = table2abcd(table)
    # return zero_denominator(a, a + c)
    POD = list()
    for i in range(np.shape(table)[0]):
        a = table[i][i]
        c = np.sum(table[i]) - a  # ?
        POD.append(zero_denominator(a, a + c))
    return POD


def metrics_MSI(table):
    # for binary classification
    print_log(f"running {currentFuncName()}")
    # a, b, c, d = table2abcd(table)
    # return zero_denominator(c, a + c)
    MSI = list()
    for i in range(np.shape(table)[0]):
        a = table[i][i]
        c = np.sum(table[i]) - a  # ?
        MSI.append(zero_denominator(c, a + c))
    return MSI


def metrics_FAR(table):
    # for binary classification
    print_log(f"running {currentFuncName()}")
    # a, b, c, d = table2abcd(table)
    # return zero_denominator(b, a + b)
    FAR = list()
    for i in range(np.shape(table)[0]):
        a = table[i][i]
        row = 0
        for j in range(np.shape(table)[0]):
            row = row + table[j][i]
        b = row - a
        FAR.append(zero_denominator(b, a + b))
    return FAR


def metrics_HSS(table):
    # for binary classification
    print_log(f"running {currentFuncName()}")
    # a, b, c, d = table2abcd(table)
    # return zero_denominator(2 * (a * d - b * c), (a + c) * (c + d) + (a + b) * (b + d))
    HSS = list()
    for i in range(np.shape(table)[0]):
        a = table[i][i]
        row = 0
        for j in range(np.shape(table)[0]):
            row = row + table[j][i]
        b = row - a
        c = np.sum(table[i]) - a
        d = np.sum(np.sum(table)) - (a + b + c)
        HSS.append(zero_denominator(2 * (a * d - b * c), (a + c) * (c + d) + (a + b) * (b + d)))
    return HSS


def metrics_MAE(table):
    if np.shape(table)[0] == 2:
        # for binary classification
        print_log(f"running {currentFuncName()}")
        a, b, c, d = table2abcd(table)
        return zero_denominator(b + c, a + b + c + d)
    return np.NaN


def metrics_RMSE(table):
    # for binary classification
    print_log(f"running {currentFuncName()}")
    return float(np.sqrt(metrics_MAE(table)))
