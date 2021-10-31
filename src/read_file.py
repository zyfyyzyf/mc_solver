import pandas as pd
import numpy as np
import pickle
# np.set_printoptions(threshold=1e6)
def read_file(label_path,feature_path, number_solver, number_feature):
    label_data_orig = pd.read_csv(label_path).values
    feature_data_orig = pd.read_csv(feature_path).values

    # 求解器时间
    del_col_label = []
    for i in range(0,number_solver*2,2):
        del_col_label.append(i)
    label_data = np.delete(label_data_orig, del_col_label, axis=1)

    # 所有特征
    del_col_feature  = [0, 7, 22, 43, 54, 60, 79, 98, 110, 122, 125]
    feature_data = np.delete(feature_data_orig, del_col_feature, axis=1)

    # 简单特征
    simple_feature_data = feature_data[:, :2]

    # 特征计算时间
    time_col_feature = [7, 22, 43, 54, 60, 79, 110, 122, 125]
    feature_time = feature_data_orig[:,time_col_feature]
    feature_time = np.maximum(feature_time, 0)
    feature_time = np.sum(feature_time, axis=1)
    # remove 0(filename) 7(pre) 22(basic) 43(klb) 54(cg) 60(dia) cl(79) sp(98) ls-sap(110) ls-gast(122) lob(125)  80-98 

    # 将全体数据打散

    return label_data, feature_data, simple_feature_data, feature_time

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
