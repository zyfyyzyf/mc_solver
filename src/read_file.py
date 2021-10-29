import pandas as pd
import numpy as np
import pickle
np.set_printoptions(threshold=1e6)
def read_file(filepath, number_solver, number_feature):
    df = pd.read_csv(filepath)
    orig_data = df.values
    del_index = []
    for i in range(0,number_solver*2+1,2):
        del_index.append(i)
    del_index.extend([25, 40, 61,72,78,97,116,128,140,143])
    # 求解器列占用 20列
    # pre:37 basic:52 klb:73 cg:84 d:90 cl:109 sp: 128 ls-saps:140 ls-gsat:152 lobjoin:155
    all_data = np.delete(orig_data, del_index, axis=1)
    # shape (全体实例数, 求解器数+特征列数)  

    np.random.shuffle(all_data)
    # 将全体数据打散

    solver_runtime = all_data[:, :number_solver]
    # shape (全体实例数, 求解器数) 取all_data前 求解器数 列
    all_feature = all_data[:, number_solver:number_feature + number_solver]
    # shape (全体实例数, 特征列数) 取all_data中间的特征列
    simple_feature = all_feature[:, :2]
    # shape (全体实例数, 简单特征列数) 取特征列的前两列
    feature_time = orig_data[:, [25, 40, 61,72,78,97,116,128,140,143]]
    feature_time = np.maximum(feature_time, 0)
    feature_time = np.sum(feature_time, axis=1)
    # shape (全体实例数, ) 特征计算时间

    return all_data, solver_runtime, all_feature, simple_feature, feature_time

def test_read_model_predict_file(args):
    # 加载标签
    with open('save_model/test_model_result.pkl', 'rb') as f:
        test_model_result = pickle.load(f)
        # print("test_model_result", test_model_result)
    
    with open('save_model/test_model_stage.pkl', 'rb') as f:
        test_model_stage = pickle.load(f)
        # print("test_model_stage", test_model_stage)
    
    with open('save_model/test_model_solved.pkl', 'rb') as f:
        test_model_solved = pickle.load(f)
        # print("test_model_solved", test_model_solved)
        
    with open('save_model/test_model_time.pkl', 'rb') as f:
        test_model_time = pickle.load(f)
        # print("test_model_time", test_model_time)
    
    return test_model_result ,test_model_stage ,test_model_solved, test_model_time

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
            test_oracle_time[f_name] = 1800
            test_oracle_choice[f_name] = -1
        else:
            test_oracle_solved[f_name] = True
            test_oracle_time[f_name] = np.min(all_data[i])
            test_oracle_choice[f_name] = np.argmin(all_data[i])
    return test_oracle_choice, test_oracle_solved, test_oracle_time
