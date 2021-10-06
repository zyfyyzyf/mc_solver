from collections import Counter

import numpy as np


def time2score(time_data, cutoff):
    outscore = time_data.copy()
    # print(time_data)
    number_instance = time_data.shape[0]
    number_solver = time_data.shape[1]
    for i in range(number_instance):
        for j in range(number_solver):
            if time_data[i][j] == cutoff:
                outscore[i][j] = time_data[i][j] * 10
            outscore[i][j] = cutoff * 10 - outscore[i][j]
    return outscore


def normal_feature_data_process(feature_data):
    fun_feature_data = feature_data.copy()
    number_instance = feature_data.shape[0]
    number_feature = feature_data.shape[1]
    all_feature_col = []
    constant_col = []
    for i in range(number_feature):
        c = Counter(feature_data[:, i])
        if c[-512] == number_instance or c[0] == number_instance:
            constant_col.append(i)
    for i in range(number_feature):
        all_feature_col.append(i)
    cleaned_feature_data = np.delete(fun_feature_data, constant_col, axis=1)
    cleaned_feature_data_mean = np.mean(cleaned_feature_data, axis=0)
    cleaned_feature_data_std = np.std(cleaned_feature_data, axis=0)
    output_feature_data = (cleaned_feature_data - cleaned_feature_data_mean) / cleaned_feature_data_std
    return output_feature_data
