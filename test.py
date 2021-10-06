import argparse
import numpy as np
import joblib
# np.set_printoptions(threshold = 1e6)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import math

from src.model import judge_feature_time, judge_solver, infer
from src.read_file import read_file
from src.util import normal_feature_data_process

parser = argparse.ArgumentParser()
parser.add_argument("--TestDataset",
                    help="测试集位置", type=str, default="data/cleaned_data/test_data_cleaned.npz")
parser.add_argument("--ModelDir",
                    help="模型目录", type=str, default="save_model")
parser.add_argument("--NumberSolver",
                    help="求解器数", type=int, default=6)
args = parser.parse_args()

# 加载模型
model_feat_time_path = args.ModelDir + '/' + "model_feat_time.npy"
model_solver_path = args.ModelDir + '/' + "model_solver.npz"
feat_time_model = joblib.load("save_model/feat_time_model.pkl")
solver_model = joblib.load("save_model/solver_model.pkl")

# 加载测试集
data = np.load(args.TestDataset)

Test_data = data['Test_data']
Test_solver_runtime = data['Test_solver_runtime']
Test_feature = data['Test_feature']
Test_simple_feature = data['Test_simple_feature']
Test_feature_time = data['Test_feature_time']

infer(solver_model, Test_solver_runtime, Test_feature, Test_feature, args)


