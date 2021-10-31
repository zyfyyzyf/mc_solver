import argparse
import numpy as np
import pandas as pd 
import joblib
# np.set_printoptions(threshold = 1e6)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import math

from src.model import judge_feature_time, judge_solver
from src.read_file import read_file
from  src.test_process import infer


parser = argparse.ArgumentParser()
parser.add_argument("--TestDataset",
                    help="测试集位置", type=str, default="data/raw_data/test_data/")
parser.add_argument("--TestLabel",
                    help="测试集标签文件", type=str, default="data/test_label.csv")
parser.add_argument("--ModelDir",
                    help="模型目录", type=str, default="save_model")
parser.add_argument("--NumberSolver",
                    help="求解器数", type=int, default=10)
args = parser.parse_args()


# 加载模型
model_feat_time_path = args.ModelDir + '/' + "model_feat_time.npy"
model_solver_path = args.ModelDir + '/' + "model_solver.npz"
feat_time_model = joblib.load("save_model/feat_time_model.pkl")
solver_model = joblib.load("save_model/solver_model.pkl")

# 加载标签(用于查表)
test_label = pd.read_csv(args.TestLabel).values
del_col = []
del1 = 0
del2 = 2
for i in range(args.NumberSolver):
    del_col.append(del1)
    del_col.append(del2)
    del1 += 3
    del2 += 3 
test_label = np.delete(test_label, del_col, axis = 1)
# shape (10,100)
print(test_label)
infer(feat_time_model, solver_model, test_label, args.TestDataset)


