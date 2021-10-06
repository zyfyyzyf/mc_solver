import argparse
import numpy as np
from src.read_file import read_file

parser = argparse.ArgumentParser()
parser.add_argument("--TrainDataset",
                    help="训练集位置", type=str, default="data/raw_data/train_data/train-mc-data-raw.csv")
parser.add_argument("--TestDataset",
                    help="测试集位置", type=str, default="data/raw_data/test_data/test-mc-data-raw.csv")
parser.add_argument("--OutputDir",
                    help="输出目录", type=str, default="save_model")
parser.add_argument("--NumberSolver",
                    help="求解器数", type=int, default=6)
parser.add_argument("--NumberFeature",
                    help="特征数", type=int, default=115)
args = parser.parse_args()
Train_data, Train_solver_runtime, Train_feature, Train_simple_feature, Train_feature_time = read_file(args.TrainDataset,
                                                                                                      args.NumberSolver,
                                                                                                      args.NumberFeature)

Test_data, Test_solver_runtime, Test_feature, Test_simple_feature, Test_feature_time = read_file(args.TestDataset,
                                                                                                 args.NumberSolver,
                                                                                                 args.NumberFeature)


np.savez('data/cleaned_data/train_data_cleaned', Train_data=Train_data, Train_solver_runtime=Train_solver_runtime,
         Train_feature=Train_feature, Train_simple_feature=Train_simple_feature,
         Train_feature_time=Train_feature_time, allow_pickle=True)

np.savez('data/cleaned_data/test_data_cleaned', Test_data=Test_data, Test_solver_runtime=Test_solver_runtime,
         Test_feature=Test_feature, Test_simple_feature=Test_simple_feature,
         Test_feature_time=Test_feature_time, allow_pickle=True)
print("数据预处理完毕")