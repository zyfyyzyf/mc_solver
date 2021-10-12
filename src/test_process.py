import os
from src.util import normal_feature_data_process,sort_key
import subprocess
import numpy as np
def test_presolve(test_result, all_file):   
    for i in range(1):
        # 测试两个预求解器
        file_path = '/home/mc_zilla/data/raw_data/test_data/' + all_file[81]
        print(file_path)
        status = subprocess.call(['timeout', '20', 'time', "./count_noproj_bareganak", '<', file_path])
        print("status1",status)
        if status == 0:
            test_result[all_file[i]] = 2
            continue
        else:
           status = subprocess.call(['timeout', '20', 'time', "./sharpSAT", '-decot', '1','-decow','100','-tmpdir','.','cs','32000', file_path])
           print("status1",status)
           if status == 0:
               test_result[all_file[i]] = 5
    return test_result

def test_backup_solver(test_result, file):
    file_path = '/home/mc_zilla/data/raw_data/test_data/' + file
    status = subprocess.call(['timeout', '20', 'time', "./sharpSAT", '-decot', '1','-decow','100','-tmpdir','.','cs','32000', file_path])
    if status == 0:
        # 使用备份求解器成功
        test_result[file] = 5
def infer(feat_time_model, TestDataset_path):
    # 模型在测试集上的表现
    '''
    test_instance = Test_solver_runtime.shape[0]
    # 制作输入数据
    X = Test_feature_time.copy()
    # 对数据进行归一化
    X = normal_feature_data_process(X)
    # shape (测试实例数,清洗后的特征列数)  删去无用的列，在列的维度上归一化  
    '''
    # 读取测试文件并按顺序排序
    all_file = []
    for file in os.listdir(TestDataset_path):  
        # file_path = os.path.join(TestDataset_path, file)  
        all_file.append(file)
        all_file = sorted(all_file,key=sort_key)
    test_result = {}
    for i in range(len(all_file)):
        test_result[all_file[i]] = -1

    '''
    # 使用两个预求解器预求解
    test_result = test_presolve(test_result, all_file)
    '''

    # 判断能否计算特征
    for i in range(len(all_file)):
        file = '/home/mc_zilla/data/raw_data/test_data/' + all_file[i]
        with open(file, 'r' ) as f:
            # print("判断实例",all_file[i],'特征计算是否超时...')
            content = f.readlines()
            simple_line = content[0]
            simple_line =simple_line[6:]
            v_c = simple_line.split()
            nb_var = v_c[0]
            nb_clause = v_c[1]
            X = np.array([nb_var,nb_clause])
            X = np.expand_dims(X, axis=0)
            # print(X.shape)
            predict = feat_time_model.predict(X)
            if predict == -1:
                # 特征计算不超时
                # ./run -all $c
                status = subprocess.call(['./copmute_feature', '-all', file])
                if status != 0:
                    test_result = test_backup_solver(test_result, all_file[i])
                else:
                    # 读取csv文件
                    pass
            elif predict == 1:
                # 特征计算超时
                # 调用备份求解器
                # subprocess.call(['timeout', '20', 'time', "./count_noproj_bareganak", '<', file_path])
                pass
            



    

