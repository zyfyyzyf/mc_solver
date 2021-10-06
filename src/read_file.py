import pandas as pd
import numpy as np


def read_file(filepath, number_solver, number_feature):
    df = pd.read_csv(filepath)
    all_data = df.values
    del_index = []
    for i in range(0,number_solver*2+1,2):
        del_index.append(i)
    all_data = np.delete(all_data, del_index, axis=1)
    # shape (全体实例数, 求解器数+特征列数+特征时间列数)  删掉filename列
    '''
    np.random.shuffle(all_data)
    # 将全体数据打散
    '''
    solver_runtime = all_data[:, :number_solver]
    # shape (全体实例数, 求解器数) 取all_data前 求解器数 列
    all_feature = all_data[:, number_solver:number_feature + number_solver]
    # shape (全体实例数, 特征列数) 取all_data中间的特征列
    simple_feature = all_feature[:, :2]
    # shape (全体实例数, 简单特征列数) 取特征列的前两列
    feature_time = all_data[:, -10:]
    feature_time = np.maximum(feature_time, 0)
    feature_time = np.sum(feature_time, axis=1)
    # shape (全体实例数, ) 特征计算时间

    return all_data, solver_runtime, all_feature, simple_feature, feature_time
