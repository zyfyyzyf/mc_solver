import os
from src.util import normal_feature_data_process,sort_key
import subprocess
def test_presolve(test_result, all_file):   
    for i in range(1):
        # 测试两个预求解器
        file_path = '/home/mc_zilla/data/raw_data/test_data/' + all_file[81]
        print(file_path)
        status =subprocess.call(['timeout', '2', 'time', "./count_noproj_bareganak", '<', file_path])
        print("status1",status)
        if status == 0:
            test_result[all_file[i]] = 2
            continue
        else:
           status =subprocess.call(['timeout', '30', 'time', "./sharpSAT", '-decot', '1','-decow','100','-tmpdir','.','cs','32000', file_path])
           print("status1",status)
           if status == 0:
               test_result[all_file[i]] = 5
    print(test_result)

def infer(TestDataset_path):
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
    test_presolve(test_result, all_file)
    print(test_result)

