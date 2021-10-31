import argparse
import numpy as np
from src.read_file import read_file
from src.util import del_constant_col, data_normalization
parser = argparse.ArgumentParser()
parser.add_argument("--TrainData_label_path",
                    help="训练集标签位置", type=str, default="data/train-data/train-label.csv")
parser.add_argument("--TrainData_feature_path",
                    help="训练集标签位置", type=str, default="data/train-data/train-feature.csv")
parser.add_argument("--OutputDir",
                    help="输出目录", type=str, default="data/train-data-cleaned/train-data-cleaned.npz")
parser.add_argument("--NumberSolver",
                    help="求解器数", type=int, default=9)
parser.add_argument("--NumberFeature",
                    help="特征数", type=int, default=115)
args = parser.parse_args()
Train_label_data, Train_feature_data, Train_simple_feature_data, Train_feature_time = read_file(args.TrainData_label_path,
                                                                                                    args.TrainData_feature_path,
                                                                                                      args.NumberSolver,
                                                                                                      args.NumberFeature)
# 删除常数列(18个sp列)
Train_feature_data = del_constant_col(Train_feature_data)

# shape (训练实例数, 97)  在列的维度上归一化
# Train_feature_data = data_normalization(Train_feature_data)


np.savez(args.OutputDir, Train_label_data=Train_label_data,
         Train_feature_data=Train_feature_data, Train_simple_feature_data=Train_simple_feature_data,
         Train_feature_time=Train_feature_time, allow_pickle=True)

print("数据预处理完毕")