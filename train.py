import argparse
import numpy as np
import joblib
# np.set_printoptions(threshold = 1e6)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import math

from src.model import judge_feature_time, judge_solver
from src.read_file import read_file

parser = argparse.ArgumentParser()
parser.add_argument("--TrainDataset",
                    help="训练集位置", type=str, default="/home/mc_zilla/data/cleaned_data/train_data_cleaned.npz")
parser.add_argument("--OutputDir",
                    help="输出目录", type=str, default="/home/mc_zilla/save_model")
parser.add_argument("--ModelType",
                    help="模型类型", type=str, default="RF")
parser.add_argument("--LabelType",
                    help="标签类型(两两成对还是整个一体)", type=str, default="single")
parser.add_argument("--FeatureCutoff",
                    help="特征计算时间阈值", type=int, default=180)
parser.add_argument("--AllCutoff",
                    help="全过程时间阈值", type=int, default=1800)
parser.add_argument("--NumberSolver",
                    help="求解器数", type=int, default=6)
parser.add_argument("--NumberFeature",
                    help="特征数", type=int, default=115)
parser.add_argument("--Seed",
                    help="随机数种子", type=int, default=1234)
parser.add_argument("--MaxPreTime",
                    help="单个预求解器运行时间相对总运行时间阈值最大比例", type=int, default=0.05)
parser.add_argument("--presolver1",
                    help="预求解器1", type=int, default=5)
parser.add_argument("--presolver2",
                    help="预求解器2", type=int, default=2)
parser.add_argument("--presolver1time",
                    help="预求解器1运行时间", type=float, default=17)
parser.add_argument("--presolver2time",
                    help="预求解器2运行时间", type=float, default=17)
parser.add_argument("--Backupsolver",
                    help="备份求解器", type=int, default=5)
parser.add_argument("--NumCrossValidation",
                    help="交叉验证折数", type=int, default=10)
parser.add_argument("--ScoreType",
                    help="分数类型", type=str, default="PAR10")
parser.add_argument("--CostType",
                    help="随机森林权重类型", type=str, default="RAW")
args = parser.parse_args()
'''
# np.random.seed(args.Seed)
# print(np.random.rand())
# np.random.seed(args.Seed)
# print(np.random.rand())
'''
data = np.load(args.TrainDataset)
# 加载数据
Train_data = data['Train_data']
Train_solver_runtime = data['Train_solver_runtime']
Train_feature = data['Train_feature']
Train_simple_feature = data['Train_simple_feature']
Train_feature_time = data['Train_feature_time']

train_instance = Train_data.shape[0]

# 训练随机森林判断实例是否能计算特征时间
feat_time_model = judge_feature_time(Train_simple_feature, Train_feature_time, args)
# 返回随机森林模型

solver_model = judge_solver(Train_solver_runtime, Train_feature_time, Train_feature, args)
# 返回随机森林模型集合 字典形式 对应的求解器组合名作为key 标签为模型
# eg  '0,1': RandomForestClassifier()

joblib.dump(feat_time_model, r'/home/mc_zilla/save_model/feat_time_model.pkl')
joblib.dump(solver_model, r'/home/mc_zilla/save_model/solver_model.pkl')

