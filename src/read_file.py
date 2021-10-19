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

def test_read_model_predict_file(args):
    # 加载标签
    test_result = {}
    test_stage = {}
    test_solved = {}
    test_time = {}
    for i in range(100):
        f_name = str(i) + '.cnf'
        v = np.random.randint(0, 6, 1)[0]
        v2 = np.random.randint(0, 2, 1)[0]
        v2 = bool(v2)
        v3 = np.random.uniform(low=0.0, high=1800.0, size=1)[0]
        v3 = round(v3,2)
        test_result[f_name] = v
        test_solved[f_name] = v2
        test_time[f_name] = v3
        test_stage[f_name] = 'main'
    return test_result ,test_stage ,test_solved, test_time

def test_read_top1_file(args):
    # 读取sharpsat-td
    test_top1_solved = {}
    test_top1_time = {}
    top1_data = pd.read_csv(args.label_file_path).values
    top1_data = top1_data[:,13]
    for i in range(top1_data.shape[0]):
        f_name = str(i)+'.cnf' 
        if top1_data[i] != 1800:
            test_top1_solved[f_name] = True
            test_top1_time[f_name] = top1_data[i]
        else:
            test_top1_solved[f_name] = False
            test_top1_time[f_name] = top1_data[i]
    return test_top1_solved, test_top1_time

def test_read_oracle_file(args):
    # 读取oracle
    test_oracle_choice = {}
    test_oracle_solved = {}
    test_oracle_time = {}
    oracle_data = pd.read_csv(args.label_file_path).values
    del_col = []
    del1 = 0
    del2 = 2
    for i in range(args.NumberSolver):
        del_col.append(del1)
        del_col.append(del2)
        del1 += 3
        del2 += 3 
    all_data = np.delete(oracle_data, del_col, axis = 1)
    for i in range(all_data.shape[0]):
        # 遍历所有的实例
        f_name = str(i)+'.cnf'
        if  np.min(all_data[i]) == 1800:
            test_oracle_solved[f_name] = False
            test_oracle_time[f_name] = -1
            test_oracle_choice[f_name] = -1
        else:
            test_oracle_solved[f_name] = True
            test_oracle_time[f_name] = np.min(all_data[i])
            test_oracle_choice[f_name] = np.argmin(all_data[i])
    return test_oracle_choice, test_oracle_solved, test_oracle_time
