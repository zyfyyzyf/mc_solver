import argparse

from src.analysis_process import average_model_solve_time, average_top1_solve_time, average_oracle_solve_time
from src.analysis_process import analysis_2, analysis_3_1, analysis_3_2
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
test_model_result , test_model_stage, test_model_solved,  test_model_time = test_read_model_predict_file(args)
# 3个dict 保存模型对100个测试样例的选择结果,是否求解成功,求解时间

# 加载top1结果
test_top1_solved, test_top1_time = test_read_top1_file(args)
# 3个dict 保存top1对100个测试样例的选择结果,是否求解成功,求解时间

# 加载oracle结果
test_oracle_result ,test_oracle_solved, test_oracle_time = test_read_oracle_file(args)
# 3个dict 保存top1对100个测试样例的选择结果,是否求解成功,求解时间

'''
# 分析1
average_model_solve_time(test_model_result ,test_model_solved, test_model_time)
average_top1_solve_time(test_top1_solved, test_top1_time)
average_oracle_solve_time(test_oracle_solved, test_oracle_time)
'''
'''
# 分析2
analysis_2(test_model_result, test_model_stage, test_model_solved, test_model_time, args)
'''
'''
# 分析3.1 每个组件求解器能力
analysis_3_1(args)
'''
'''
# 分析3.2 组件求解器的斯皮尔曼相关系数
analysis_3_2(args)
'''
'''
# 分析3.3 主阶段求解器被选择的频率
# 分析3.4 各个求解器的求解比率(pre和主阶段和备份分开)(被选择并能求解)