import argparse

from src.analysis_process import average_solve_time_model
from src.read_file import test_read_model_predict_file, test_read_oracle_file, test_read_top1_file
parser = argparse.ArgumentParser()
parser.add_argument("--NumberSolver",
                    help="求解器数", type=int, default=6)
parser.add_argument("--predict_file_path",
                    help="预测结果文件地址", type=str, default="save_model/test_result.pkl")
parser.add_argument("--label_file_path",
                    help="测试集标签文件地址", type=str, default="data/test_label.csv")                 
args = parser.parse_args()

# 加载模型测试结果
test_model_result ,test_model_solved, test_model_time = test_read_model_predict_file(args)
# 3个dict 保存模型对100个测试样例的选择结果,是否求解成功,求解时间

# 加载top1结果
test_top1_solved, test_top1_time = test_read_top1_file(args)
# 3个dict 保存top1对100个测试样例的选择结果,是否求解成功,求解时间

# 加载oracle结果
test_oracle_result ,test_oracle_solved, test_oracle_time = test_read_oracle_file(args)
# 3个dict 保存top1对100个测试样例的选择结果,是否求解成功,求解时间

average_solve_time_model(test_model_result ,test_model_solved, test_model_time)


# 分析1.3
# average_solve_time(test_result ,test_solved, test_time, test_label)
