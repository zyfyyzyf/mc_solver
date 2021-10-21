import pandas as pd
import numpy as np
import matplotlib.mlab as mlab  
import matplotlib.pyplot as plt  
import seaborn as sns
from src.util import draw_pie, draw_CDF, create_cdf_data, draw_CDF_analysis_4

def average_model_solve_time(test_model_result ,test_model_solved, test_model_time):
    all_model_time = 0
    solved_count = 0
    for i in range(100):
        # 如果可以求解
        if test_model_solved[i] == True:
           all_model_time += test_model_time[i]
           solved_count += 1
    average_model_time = all_model_time / solved_count

def average_top1_solve_time(test_top1_solved, test_top1_time):
    all_top1_time = 0
    solved_count = 0
    print(test_top1_solved)
    for i in range(100):
        # 如果可以求解
        f_name = str(i) + '.cnf'
        if test_top1_solved[f_name] == True:
           all_top1_time += test_top1_time[f_name]
           solved_count += 1
    average_top1_time = all_top1_time / solved_count
    print("top1求解实例数:",solved_count)
    print("top1超时实例数:",100 - solved_count)
    print("top1平均求解时间(s):",average_top1_time)
    all_top1_time = all_top1_time / 60 / 60
    print("top1总求解时间(h):",all_top1_time)

def average_oracle_solve_time(test_oracle_solved, test_oracle_time):
    all_oracle_time = 0
    solved_count = 0
    for i in range(100):
        # 如果可以求解
        f_name = str(i) + '.cnf'
        if test_oracle_solved[f_name] == True:
           all_oracle_time += test_oracle_time[f_name]
           solved_count += 1
    average_oracle_time = all_oracle_time / solved_count
    print("oracle求解实例数:",solved_count)
    print("oracle超时实例数:",100 - solved_count)
    print("oracle平均求解时间(s):",average_oracle_time)
    all_oracle_time = all_oracle_time / 60 / 60
    print("oracle总求解时间(h):",all_oracle_time)   

def analysis_2(test_model_result, test_model_stage, test_model_solved, test_model_time, args):
    print("test_model_result", test_model_result)
    print("test_model_solved", test_model_solved)
    print("test_model_solved", test_model_solved)
    component_solver_choice = {}
    component_solver_solved = {}
    component_solver_average_runtime = {}
    for i in range(args.NumberSolver):
        component_solver_choice[i] = 0
        component_solver_solved[i] = 0
        component_solver_average_runtime[i] = 0
    for i in range(100):
        f_name = str(i) + '.cnf'
        component_solver_choice[test_model_result[f_name]] += 1
        if test_model_solved[f_name] == True:
            component_solver_solved[test_model_result[f_name]] += 1
            component_solver_average_runtime[test_model_result[f_name]] += test_model_time[f_name]
    for i in range(args.NumberSolver):
        component_solver_average_runtime[i] = component_solver_average_runtime[i] / component_solver_solved[i]
    print("组件求解器被选择情况",component_solver_choice)
    print("组件求解器平均求解时间", component_solver_average_runtime)
    print("被选择的组件求解器求解情况", component_solver_solved)

def analysis_3_1(args):
    component_solver_choice = {}
    for i in range(args.NumberSolver):
        component_solver_choice[i] = 0
    analysis_data = pd.read_csv(args.label_file_path).values
    del_col = []
    del1 = 0
    del2 = 2
    for i in range(args.NumberSolver):
        del_col.append(del1)
        del_col.append(del2)
        del1 += 3
        del2 += 3 
    analysis_data = np.delete(analysis_data, del_col, axis = 1)
    for i in range(args.NumberSolver):
        for j in range(100):
            if analysis_data[j][i] != 1800:
                component_solver_choice[i] += 1
    pie_input = []
    labels = []
    for i in range(args.NumberSolver):
        pie_input.append(component_solver_choice[i])
    labels = ['c2d','nus_narasimha','d4', 'MCSim','sharpsat-td','count_bareganak']
    explode =[0,0,0,0,0.4,0]
    colors = ["blue","red","coral","green","yellow","orange"] 
    draw_pie(pie_input, labels, explode, colors)

def analysis_3_2(args):
    analysis_data = pd.read_csv(args.label_file_path).values
    del_col = []
    del1 = 0
    del2 = 2
    for i in range(args.NumberSolver):
        del_col.append(del1)
        del_col.append(del2)
        del1 += 3
        del2 += 3 
    analysis_data = np.delete(analysis_data, del_col, axis = 1)
    analysis_data = analysis_data.astype(int)
    col_name = ['c2d','nus_narasimha','d4', 'MCSim','sharpsat-td','count_bareganak']
    analysis_data = pd.DataFrame(analysis_data, columns=col_name)
    corr = analysis_data.corr('spearman')
    print(corr)
    f,ax = plt.subplots(figsize=(12,6))
    fig = sns.heatmap(corr,annot=True)
    scatter_fig = fig.get_figure()
    scatter_fig.savefig("analysis_3_2.eps", dpi = 600)

def analysis_3_5(test_oracle_time, test_top1_time):
    top1_time_data = create_cdf_data(test_top1_time, flag = True)
    oracle_time_data = create_cdf_data(test_oracle_time, flag = True)
    draw_CDF(oracle_time_data, top1_time_data) 

def analysis_4(args):
    time_col = [7, 22, 43, 54, 60, 79,  110, 122, 125]
    data = pd.read_csv(args.feature_file_path).values
    data = data[:,time_col]
    data = np.sum(data, axis = 1)
    data = create_cdf_data(data.tolist(),flag = False)
    draw_CDF_analysis_4(data) 

def analysis_5(args):
    data = pd.read_csv(args.feature_file_path).values
    var_data = data[:,1]
    cls_data = data[:,2]
    print("变量数最小值:",np.min(var_data))
    print("变量数最大值:",np.max(var_data))
    print("变量数平均值:",np.mean(var_data))
    print("变量数中位数:",np.median(var_data))
    print("子句数最小值:",np.min(cls_data))
    print("子句数最大值:",np.max(cls_data))
    print("子句数平均值:",np.mean(cls_data))
    print("子句数中位数:",np.median(cls_data))
    '''
    # 分析5.2 柱状图
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.set_yscale("log")
    ax1.set_adjustable("datalim")
    ax1.axhline(y = np.mean(var_data),c="g", ls="--",lw=1)
    ax1.axhline(y = np.median(var_data),c="r", ls="dotted",lw=1)
    x_label = [i for i in range(0,100)]
    y_label = var_data
    ax1.bar(x_label, y_label,width=0.5)
    plt.xticks([i for i in range(0,101,10)])
    plt.savefig("analysis_5.png")
    '''
    # 分析5.3 变量字句比 散点图
    x = var_data.tolist()
    y = cls_data.tolist()
    c = []
    for i in range(100):
        tmp = x[i] / y[i]
        c.append(tmp)
    fig= plt.figure()
    plt.scatter(x = x, y = y, c = c)
    plt.yscale("log")
    plt.xscale("log")
    # plt.show()
    plt.colorbar(label="Like/Dislike Ratio", orientation="horizontal")
    # plt.savefig("analysis_5_3.eps", format='eps') 
    plt.savefig("analysis.jpg")
    '''
    # fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_adjustable("datalim")
    ax.scatter(x, y, c, cmap="summer") 
    # sax.colorbar(label="Like/Dislike Ratio", orientation="horizontal")
    '''
