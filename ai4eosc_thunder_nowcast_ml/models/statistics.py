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


def contingency_table(measure, prediction, levels=[0, 1]):
    print_log(f"running {currentFuncName()}")
    measure_tmp, prediction_tmp = [], []
    for i in measure:
        measure_tmp.extend(i) if isinstance(i, list) else measure_tmp.extend([i])
    for i in prediction:
        prediction_tmp.extend(i) if isinstance(i, list) else prediction_tmp.extend([i])
    measure, prediction = measure_tmp, prediction_tmp
    # length check
    print_log(f"len(measure) == {len(measure)}, len(prediction) == {len(prediction)}")
    return pd.crosstab(measure, prediction)


def length(x):
    try:
        return len(x)
    except Exception as err:
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
    a, b, c, d = table2abcd(table)
    return zero_denominator(a + d, a + b + c + d)


def metrics_F1(table):
    # for binary classification
    print_log(f"running {currentFuncName()}")
    a, b, c, d = table2abcd(table)
    return zero_denominator(a, a + 0.5 * (b + c))


def metrics_CSI(table):
    # for binary classification
    print_log(f"running {currentFuncName()}")
    a, b, c, d = table2abcd(table)
    return zero_denominator(a, a + b + c)


def metrics_POD(table):
    # for binary classification
    print_log(f"running {currentFuncName()}")
    a, b, c, d = table2abcd(table)
    return zero_denominator(a, a + c)


def metrics_MSI(table):
    # for binary classification
    print_log(f"running {currentFuncName()}")
    a, b, c, d = table2abcd(table)
    return zero_denominator(c, a + c)


def metrics_FAR(table):
    # for binary classification
    print_log(f"running {currentFuncName()}")
    a, b, c, d = table2abcd(table)
    return zero_denominator(b, a + b)


def metrics_HSS(table):
    # for binary classification
    print_log(f"running {currentFuncName()}")
    a, b, c, d = table2abcd(table)
    return zero_denominator(2 * (a * d - b * c), (a + c) * (c + d) + (a + b) * (b + d))


def metrics_MAE(table):
    # for binary classification
    print_log(f"running {currentFuncName()}")
    a, b, c, d = table2abcd(table)
    return zero_denominator(b + c, a + b + c + d)


def metrics_RMSE(table):
    # for binary classification
    print_log(f"running {currentFuncName()}")
    return float(np.sqrt(metrics_MAE(table)))
