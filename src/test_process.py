import os
from src.util import sort_key
import subprocess
import numpy as np
import pandas as pd
import joblib
import time
import pickle
def test_presolve(test_result, test_stage, test_solved, test_time, test_label,filename,TestDataset_path, starttime):   
    # 测试两个预求解器
    print("为实例 " ,filename ,' 使用预求解器nus_narasimha...')
    file_index = int(filename[0])
    pre1_solve_time = test_label[file_index,1]
    endtime = time.time()
    dtime = endtime - starttime
    if pre1_solve_time <= 20:
        print("实例 " ,filename ,' 由预求解器nus_narasimha求解...')
        test_result[filename] = 1
        test_stage[filename] = 'pre1'
        test_solved[filename] = True
        test_time[filename] = round(pre1_solve_time + dtime, 2)
        print("test_result[filename]", test_result[filename])
        print('test_stage[filename]', test_stage[filename])
        print("test_solved[filename]",test_solved[filename])
        print("test_time[filename]", test_time[filename])
        return True
    else:
        print("预求解器count_bareganak失败...")
        print("为实例", filename, '使用预求解器sharpsat_td...')
        pre2_solve_time = test_label[file_index,4]
        endtime = time.time()
        dtime = endtime - starttime
        if pre2_solve_time <= 20:
            print("实例 " ,filename ,' 由预求解器sharpsat_td求解...')
            test_result[filename] = 4
            test_stage[filename] = 'pre2'
            test_solved[filename] = True
            test_time[filename] = round(pre2_solve_time + dtime, 2)
            print("test_result[filename]", test_result[filename])
            print('test_stage[filename]', test_stage[filename])
            print("test_solved[filename]",test_solved[filename])
            print("test_time[filename]", test_time[filename])
            return True

def test_backup_solver(test_result, test_stage, test_solved, test_time, test_label, filename, TestDataset_path, starttime):
    file_path = TestDataset_path + file
    file_index = int(filename[0])
    backup_solve_time = test_label[file_index,4]
    endtime = time.time()
    dtime = endtime - starttime
    test_result[filename] = 4
    test_stage[filename] = 'backup'
    test_time[filename] = round(backup_solve_time + dtime, 2)
    if backup_solve_time + dtime <= 1800:
        test_solved[filename] = True
    print("test_result[filename]", test_result[filename])
    print('test_stage[filename]', test_stage[filename])
    print("test_solved[filename]",test_solved[filename])
    print("test_time[filename]", test_time[filename])
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
        flag = test_presolve(test_result, test_stage, test_solved, test_time, test_label, all_file[i], TestDataset_path, starttime)
        if flag:
            continue
        # 预求解器不能求解，使用简单特征判断能否计算特征时间
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
                if os.path.exists('./test_feature.csv'): 
                   os.remove('./test_feature.csv')
                os.mknod("./test_feature.csv")
                status = subprocess.call(['./compute_feature', '-all', file], stdout = subprocess.PIPE,stderr = subprocess.STDOUT)
                if status != 0:
                    print("实例 ", all_file[i]," 特征计算失败，使用备份求解器...")
                    test_backup_solver(test_result, test_stage, test_solved, test_time, test_label, all_file[i], TestDataset_path, starttime)
                else:
                    # 如果能计算特征
                    # 读取csv文件
                    print("实例 ", all_file[i]," 特征计算成功，进行求解器选择...")
                    feature_data = pd.read_csv("/home/zhangyf/mc_zilla/test_feature.csv").values
                    # remove 0 7(pre) 22(basic) 43(klb) 54(cg) 60(dia) cl(79) sp(98) ls-sap(110) ls-gast(121) lob(124)  80-98
                    del_col = [0, 7, 22, 43, 54, 60, 79,  110, 122, 125] 
                    for index in range(80, 99):
                        del_col.append(index)
                    del_col.sort()
                    cleaned_feature_data = np.delete(feature_data, del_col, axis=1)
                    predict = solver_model.predict(cleaned_feature_data)
                    print("mc_zilla的选择是：",predict[0])
                    file_index = int(all_file[i][0])
                    main_solve_time = test_label[file_index, predict[0]]
                    endtime = time.time()
                    dtime = endtime - starttime
                    test_result[all_file[i]] = predict[0]
                    test_stage[all_file[i]] = 'main'
                    test_time[all_file[i]] = round(main_solve_time + dtime, 2)
                    if main_solve_time + dtime <= 1800:
                        test_solved[all_file[i]] = True
            elif predict == 1:
                print("实例 ", all_file[i], "特征计算超时，使用备份求解器")
                test_backup_solver(test_result, test_stage, test_solved, test_time, test_label, all_file[i], TestDataset_path, starttime)
    print("test_result[filename]", test_result)
    print('test_stage[filename]', test_stage)
    print("test_solved[filename]",test_solved)
    print("test_time[filename]", test_time)
    output = open('save_model/test_model_result.pkl', 'wb')
    pickle.dump(test_result, output)
    output.close()

    output = open('save_model/test_model_stage.pkl', 'wb')
    pickle.dump(test_stage, output)
    output.close()

    output = open('save_model/test_model_solved.pkl', 'wb')
    pickle.dump(test_solved, output)
    output.close()

    output = open('save_model/test_model_time.pkl', 'wb')
    pickle.dump(test_time, output)
    output.close()

        
            



    

