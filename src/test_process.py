import os
from src.util import sort_key
import subprocess
import numpy as np
import pandas as pd
import joblib
import time
def test_presolve(test_result, test_stage, test_solved, test_time, test_label,filename,TestDataset_path):   
    # 测试两个预求解器
    file_path = TestDataset_path + file
    print(file_path)
    print("为实例 " ,file ,' 使用预求解器nus_narasimha...')
    # status = subprocess.call(['timeout', '20', 'time', "./count_noproj_bareganak", '<', file_path],stdout = subprocess.PIPE,stderr = subprocess.STDOUT)
    pre1_solve_time = test_time[filename[0]][1]
    if pre1_solve_time != 1800:
       endtime = time.time()
       dtime = endtime - starttime
       if pre1_solve_time + dtime <= 1800:
            test_result[filename] == 1
            test_stage[filename] == 'pre1'
            test_solved
    else:
        print("预求解器count_bareganak失败...")
        print("为实例" ,file,'使用预求解器sharpsat_td...')
        status = subprocess.call(['timeout', '20', 'time', "./sharpSAT", '-decot', '1','-decow','100','-tmpdir','.','cs','32000', file_path],stdout = subprocess.PIPE,stderr = subprocess.STDOUT)
        if status == 0:
            test_result[file] = 5
            test_solved[file] = True
        else:
            print("预求解器sharpsat_td失败...")
    return test_result ,test_solved

def test_backup_solver(test_result, file,TestDataset_path):
    file_path = TestDataset_path + file
    status = subprocess.call(['timeout', '20', 'time', "./sharpSAT", '-decot', '1','-decow','100','-tmpdir','.','cs','32000', file_path],stdout = subprocess.PIPE,stderr = subprocess.STDOUT)
    if status == 0:
        # 使用备份求解器成功
        test_result[file] = 4

def read_test_file(TestDataset_path):
    # 读取测试文件并按顺序排序
    all_file = []
    for file in os.listdir(TestDataset_path):    
        all_file.append(file)
        all_file = sorted(all_file,key=sort_key)
    test_result = {}
    test_stage = {}
    test_solved = {}
    test_time ={}
    for i in range(len(all_file)):
        test_result[all_file[i]] = -1
        test_stage[all_file[i]] = 'null'
        test_solved[all_file[i]] = False
        test_time[all_file[i]] = -1
    return all_file, test_result, test_stage, test_solved, test_time

def infer(feat_time_model, solver_model, test_label, TestDataset_path):
    # 模型在测试集上的表现

    # 读取测试文件并按顺序排序
    all_file, test_result, test_stage, test_solved, test_time = read_test_file(TestDataset_path)
    # 判断能否计算特征
    for i in range(len(all_file)):
        # 为每个实例计时
        starttime = time.time()
        file = TestDataset_path + all_file[i]
        print("开始求解实例 ",all_file[i],"...")
        # 使用两个预求解器预求解
        # 查表
        test_result ,test_solved= test_presolve(test_result, test_stage, test_solved, test_time, test_label, all_file[i], TestDataset_path, starttime)
        with open(file, 'r' ) as f:
            print("判断实例 ",all_file[i],' 特征计算是否超时...')
            content = f.readlines()
            simple_line = content[0]
            simple_line =simple_line[6:]
            v_c = simple_line.split()
            nb_var = v_c[0]
            nb_clause = v_c[1]
            X = np.array([nb_var,nb_clause])
            X = np.expand_dims(X, axis=0)
            predict = feat_time_model.predict(X)
            if predict == -1:
                print("实例 ", all_file[i], "特征计算不超时，开始计算特征...")
                os.remove('/home/mc_zilla/test_bin/test_feature.csv')
                os.mknod("/home/mc_zilla/test_bin/test_feature.csv")
                status = subprocess.call(['./copmute_feature', '-all', file],stdout = subprocess.PIPE,stderr = subprocess.STDOUT)
                if status != 0:
                    print("实例 ", all_file[i]," 特征计算失败，使用备份求解器...")
                    test_result[all_file[i]] = 4
                    '''
                    加代码，查表增加时间    
                    如果总时间超时就失败
                    '''
                    test_solved[all_file[i]] = False
                    # test_result = test_backup_solver(test_result, all_file[i], TestDataset_path)
                else:
                    # 如果能计算特征
                    # 读取csv文件
                    print("实例 ", all_file[i]," 特征计算成功，进行求解器选择...")
                    feature_data = pd.read_csv("/home/mc_zilla/test_bin/test_feature.csv").values
                    # remove 0 7(pre) 22(basic) 43(klb) 54(cg) 60(dia) cl(79) sp(98) ls-sap(110) ls-gast(121) lob(124)  80-98
                    del_col = [0, 7, 22, 43, 54, 60, 79,  110, 122, 125] 
                    for index in range(80, 99):
                        del_col.append(index)
                    del_col.sort()
                    cleaned_feature_data = np.delete(feature_data, del_col, axis=1)
                    predict = solver_model.predict(cleaned_feature_data)
                    print("mc_zilla的选择是",predict[0])
                    '''
                    加代码，查表增加时间    
                    如果总时间超时就失败
                    '''
                    test_result[all_file[i]] = predict[0]
                    test_solved[all_file[i]] = True
            elif predict == 1:
                print("实例 ", all_file[i], "特征计算超时，使用备份求解器")
                # 特征计算超时
                # 调用备份求解器
                test_result[all_file[i]] = 4
                test_solved[all_file[i]] = False
                '''
                加代码，查表增加时间    
                如果总时间超时就失败
                '''
                # test_result = test_backup_solver(test_result, all_file[i], TestDataset_path)
        endtime = time.time()
        dtime = endtime - starttime
        dtime = str(dtime)[:4]
        test_time[all_file[i]] = dtime
        print("实例 ",all_file[i]," 运行时间：" , dtime)  #显示到微秒
        print("test_result", test_result)
        print("test_solved",test_solved)
        print("test_time", test_time)
    np.savez(r'/home/mc_zilla/save_model/test_result.pkl',test_result,test_solved,test_time)

        
            



    

