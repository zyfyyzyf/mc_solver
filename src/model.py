import math
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from src.util import time2score, normal_feature_data_process
import sklearn
import random

# np.set_printoptions(threshold=1e6)


# 训练随机森林判断实例是否能计算特征时间
def judge_feature_time(Train_feature_time, args):
    train_instance = Train_feature_time.shape[0]
    # 生成随机森林模型
    model_feat_time = RandomForestClassifier(n_estimators=100)

    # 制作标签和训练数据
    y = np.zeros(shape=(train_instance,))
    y -= 1
    for i in range(train_instance):
        if Train_feature_time[i] >= args.FeatureCutoff:
            y[i] = 1
    # y shape (训练实例数, ) 值为-1表明特征计算不超时 1表明特征计算超时

    # 制作输入数据
    X = np.expand_dims(Train_feature_time, axis=1).copy()
    # X shape (训练实例数, 1)  每个实例的特征计算时间

    # 进行交叉验证
    kf = KFold(n_splits=args.NumCrossValidation)
    for train_index, val_index in kf.split(X):
        train_X, train_y = X[train_index], y[train_index]
        val_X, val_y = X[val_index], y[val_index]
        model_feat_time.fit(train_X, train_y)
        print("特征计算时间，交叉验证分数: ", model_feat_time.score(val_X, val_y))
    # 返回模型
    return model_feat_time


def choice_ids(Train_feature_time, Train_solver_runtime, args):
    train_instance = Train_feature_time.shape[0]
    # 制作特征时间标签
    feature_time_label = np.zeros(shape=(train_instance,), dtype=int)
    feature_time_label -= 1
    for i in range(train_instance):
        if Train_feature_time[i] > args.FeatureCutoff:
            feature_time_label[i] = 1
    # 取出使用预求解器超时并且可以进行特征计算的实例
    # 因为只有这些实例需要进行求解器选择
    need_choice_id = []
    for i in range(train_instance):
        if Train_solver_runtime[i, args.presolver1] > args.presolver1time or \
                Train_solver_runtime[i, args.presolver2] > args.presolver2time:
            if feature_time_label[i] == -1:
                need_choice_id.append(i)
    return need_choice_id

    # 训练随机森林为每个实例选择求解器


def get_weight_input_label(Train_solver_runtime, Train_feature_time, Train_feature, args):
    train_instance = Train_feature_time.shape[0]
    # 先选择使用两个预求解器超时并且可以进行特征计算的实例
    choice = choice_ids(Train_feature_time, Train_solver_runtime, args)
    # list len 需要进行主求解器选择的实例  只有这些实例才需要进入求解器选择阶段

    # 制作输入数据
    X = Train_feature.copy()
    # 对数据进行归一化
    # X = normal_feature_data_process(X)
    # shape (训练实例数,清洗后的特征列数)  删去无用的列，在列的维度上归一化
    X = X[choice, :]
    # shape (需要训练实例数,清洗后的特征列数)

    # 制作标签
    # 标签模式: 是两两成对模式还是整体模式
    if args.LabelType == 'pair':
        # 先制作权重
        weights = {}
        score = time2score(Train_solver_runtime, args.AllCutoff)
        labels = {}
        for i in range(args.NumberSolver):
            for j in range(i + 1, args.NumberSolver):
                y = np.ones(shape=(train_instance,), dtype=int)
                # shape (训练实例数, ) 为每一种组合生成一个标签 求解器i比求解器j快 标签为1 否则为-1
                weight = np.zeros(shape=(train_instance,))
                # shape (训练实例数, ) 为每一种组合生成一个权重 值为求解器i的PAR10分数与求解器j的PAR10分数差
                for index in range(train_instance):
                    if Train_solver_runtime[index, i] > Train_solver_runtime[index, j]:
                        y[index] = -1
                    weight[index] = np.abs(score[index, i] - score[index, j])
                # 对应的求解器名作为key 标签为value 保存到字典label中
                # eg: '3,4': array (训练实例数, )
                y = y[choice,]
                weight = weight[choice,]
                key = str(i) + "," + str(j)
                labels[key] = y
                weights[key] = weight
        return weights, X, labels

    elif args.LabelType == 'single':
        # 单标签类型
        y = np.zeros(shape=(train_instance,), dtype=int)
        weight = np.zeros(shape=(train_instance,))
        score = time2score(Train_solver_runtime, args.AllCutoff)
        foo = np.argmin(Train_solver_runtime, axis=1)
        for i in range(train_instance):
            y[i] = foo[i]
            for j in range(args.NumberSolver):
                weight[i] += (score[i, foo[i]] - score[i, j])
            weight[i] /= (args.NumberSolver - 1)
        y = y[choice,]
        weight = weight[choice,]
        return weight, X, y


def pair_val(models, keys, val_input, val_label, args):
    # models list 长度 求解器组合数  每个都是一个GBDT
    # eg {'0,1': GradientBoostingClassifier(n_estimators=10), '0,2': Gradie
    val_X = val_input
    # shape (验证集实例数,特征列数) 验证集输入
    val_y = val_label
    # list 每个都是 (验证集实例数,)
    # eg '0,1': array([ 1,  1, -1,  1,  1,  1,
    ans = []
    solvers_counter ={}
    # 多候选求解器
    val_final_winner = []
    val_instance = len(val_input)
    print("验证集输入", val_input)
    for i_inst in range(val_instance):
        # 对每个验证集实例进行预测
        ss_solvers_counter = {}
        for index in range(args.NumberSolver):
            solvers_counter[str(index)] = 0
            # 初始化求解器的获胜数 计数器
        for j in range(len(models)):
            # 遍历求解器组合数 这么多的模型
            key = keys[j]
            solver1 = key[0]
            solver2 = key[2]
            predict_y = models[key].predict(val_X)
            print("第"+str(i_inst)+'个实例的15种组合预测',predict_y)
            # list 长度 验证集实例数 值为1/-1
            # 验证实例在每一种求解器组合下都有一个预测
            if predict_y[i_inst] == 1:
                solvers_counter[solver1] += 1
            if predict_y[i_inst] == -1:
                solvers_counter[solver2] += 1
        # 如何处理相同胜出数的求解器
        print("求解器计数器",solvers_counter)
        max_solved = max(solvers_counter.values())
        # 最强求解器的获胜数
        good_solvers = [k for k, v in solvers_counter.items() if v == max_solved]
        print("第"+str(i_inst)+'个实例的最佳求解器',good_solvers)
        # 获取最强求解器列表
        if len(good_solvers) != 1:
            # 有多个拥有同样胜出数的求解器 再互相比一次 不行就随机选
            ss_solver = len(good_solvers)
            ss_solvers_counter = {}
            for i in range(ss_solver):
                ss_solvers_counter[str(good_solvers[i])] = 0
            # 初始化强强争霸 计数器
            for i in range(ss_solver):
                for j in range(i+1, ss_solver):
                    # 遍历所有的强强求解器组合
                    key = str(good_solvers[i]) + ',' +str(good_solvers[j])
                    solver1 = key[0]
                    26391X
                    solver2 = key[2]
                    # 取模型
                    print("小key", key)
                    ss_predict_y = models[key].predict(val_X)
                    print("第"+str(i_inst)+'个实例的小predict_y' ,ss_predict_y)
                    # list 长度 验证集实例数 值为1/-1
                    if ss_predict_y[i_inst] == 1:
                        ss_solvers_counter[solver1] += 1
                    if ss_predict_y[i_inst] == -1:
                        ss_solvers_counter[solver2] += 1
            print("小计数器",ss_solvers_counter)
            ss_max_solved = max(ss_solvers_counter.values())
            ss_good_solvers = [k for k, v in ss_solvers_counter.items() if v == ss_max_solved]
            print("小胜者",ss_good_solvers)
            print("old", good_solvers)
            if len(ss_good_solvers) != 1:
                good_solvers = random.sample(ss_good_solvers, 1)
            else:
                good_solvers = ss_good_solvers
            print("new", good_solvers)
            print("第"+str(i_inst)+'个实例的最终胜者' ,good_solvers)
        val_final_winner.extend(good_solvers)
    print("最终结果",val_final_winner)
    input()

def model_choice(Train_solver_runtime, Train_feature_time, Train_feature, args):
    # 进行模型选择
    global models
    if args.LabelType == 'single':
        weight, X, y = get_weight_input_label(Train_solver_runtime, Train_feature_time, Train_feature, args)
        if args.ModelType == 'RF':
            # 随机森林
            model = RandomForestClassifier(n_estimators=100)
            # 在260颗树下 没归一化0.717 # 归一化后0.69
            # 在100颗树下 没归一化0.705 # 归一化后0.709
            # 十折交叉验证
            score = []
            time = 1
            kf = KFold(n_splits=args.NumCrossValidation)
            for train_index, val_index in kf.split(X):
                print('第' + str(time) + "折交叉验证...")
                train_X, train_y = X[train_index], y[train_index]
                val_X, val_y = X[val_index], y[val_index]
                train_weight = weight[train_index,]
                model.fit(train_X, train_y, sample_weight=train_weight)
                score.append(model.score(val_X, val_y))
                time += 1
                # 在最后一折的数据很难验证
            print("模型 " + args.ModelType + " 在" + args.LabelType + " 标签下的10折平均交叉验证分数是: ", np.mean(score))

        if args.ModelType == 'AdaBoost':
            # SVM模型
            model = AdaBoostClassifier() # 0.39
            # 十折交叉验证
            score = []
            time = 1
            kf = KFold(n_splits=args.NumCrossValidation)
            for train_index, val_index in kf.split(X):
                print('第' + str(time) + "折交叉验证...")
                train_X, train_y = X[train_index], y[train_index]
                val_X, val_y = X[val_index], y[val_index]
                train_weight = weight[train_index,]
                model.fit(train_X, train_y, sample_weight=train_weight)
                score.append(model.score(val_X, val_y))
                time += 1
                # 在最后一折的数据很难验证
            print("模型 " + args.ModelType + " 在" + args.LabelType + " 标签下的10折平均交叉验证分数是: ", np.mean(score))

        if args.ModelType == 'SVM':
            # SVM模型
            # model = SVC(kernel='rbf', C=1.0, gamma='auto') # 0.50
            # model = SVC(kernel="linear", C=0.025) # 归一化后 0.569
            model = SVC(gamma=2, C=1) # 归一化后 0.625
            # 十折交叉验证
            score = []
            time = 1
            kf = KFold(n_splits=args.NumCrossValidation)
            for train_index, val_index in kf.split(X):
                print('第' + str(time) + "折交叉验证...")
                train_X, train_y = X[train_index], y[train_index]
                val_X, val_y = X[val_index], y[val_index]
                train_weight = weight[train_index,]
                model.fit(train_X, train_y, sample_weight=train_weight)
                score.append(model.score(val_X, val_y))
                time += 1
                # 在最后一折的数据很难验证
            print("模型 " + args.ModelType + " 在" + args.LabelType + " 标签下的10折平均交叉验证分数是: ", np.mean(score))

        if args.ModelType == 'LR':
            # 逻辑回归模型
            model = LogisticRegression(penalty='l1',
                                       dual=False, tol=0.0001, C=1.0, fit_intercept=True,
                                       intercept_scaling=1, class_weight=None,
                                       random_state=None, solver='liblinear', max_iter=10000,
                                       multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)
            # 10折交叉验证
            score = []
            time = 1
            kf = KFold(n_splits=args.NumCrossValidation)
            for train_index, val_index in kf.split(X):
                print('第' + str(time) + "折交叉验证...")
                train_X, train_y = X[train_index], y[train_index]
                val_X, val_y = X[val_index], y[val_index]
                train_weight = weight[train_index,]
                model.fit(train_X, train_y, sample_weight=train_weight)
                score.append(model.score(val_X, val_y))
                # 在最后一折的数据很难验证
                time += 1
            print("模型 " + args.ModelType + " 在 " + args.LabelType + " 标签下的10折平均交叉验证分数是: ", np.mean(score))

        if args.ModelType == 'GBDT':
            # 梯度提升决策树
            model = GradientBoostingClassifier(n_estimators=200) # 未归一化 0.675 # 归一化 0.688
            # 10折交叉验证
            score = []
            time = 1
            kf = KFold(n_splits=args.NumCrossValidation)
            for train_index, val_index in kf.split(X):
                print('第' + str(time) + "折交叉验证...")
                train_X, train_y = X[train_index], y[train_index]
                val_X, val_y = X[val_index], y[val_index]
                train_weight = weight[train_index,]
                model.fit(train_X, train_y, sample_weight=train_weight)
                score.append(model.score(val_X, val_y))
                time += 1
                # 在最后一折的数据很难验证
            print("模型 " + args.ModelType + " 在 " + args.LabelType + " 标签下的10折平均交叉验证分数是: ", np.mean(score))

        if args.ModelType == 'QDA':
            # 二次判别分析 0.50
            model = QuadraticDiscriminantAnalysis()
            # 10折交叉验证
            score = []
            time = 1
            kf = KFold(n_splits=args.NumCrossValidation)
            for train_index, val_index in kf.split(X):
                print('第' + str(time) + "折交叉验证...")
                train_X, train_y = X[train_index], y[train_index]
                val_X, val_y = X[val_index], y[val_index]
                train_weight = weight[train_index,]
                model.fit(train_X, train_y)
                score.append(model.score(val_X, val_y))
                time += 1
                # 在最后一折的数据很难验证
            print("模型 " + args.ModelType + " 在" + args.LabelType + " 标签下的10折平均交叉验证分数是: ", np.mean(score))

        if args.ModelType == 'MLP':
            # 多层感知机 0.41
            model = MLPClassifier(alpha=1)
            # 10折交叉验证
            score = []
            time = 1
            kf = KFold(n_splits=args.NumCrossValidation)
            for train_index, val_index in kf.split(X):
                print('第' + str(time) + "折交叉验证...")
                train_X, train_y = X[train_index], y[train_index]
                val_X, val_y = X[val_index], y[val_index]
                train_weight = weight[train_index,]
                model.fit(train_X, train_y)
                score.append(model.score(val_X, val_y))
                time += 1
                # 在最后一折的数据很难验证
            print("模型 " + args.ModelType + " 在" + args.LabelType + " 标签下的10折平均交叉验证分数是: ", np.mean(score))

        if args.ModelType == 'MNBC':
            # 多项朴素贝叶斯分类器
            model = MultinomialNB(alpha=0.01) # 归一化后 0.67+
            # 10折交叉验证
            score = []
            time = 1
            kf = KFold(n_splits=args.NumCrossValidation)
            for train_index, val_index in kf.split(X):
                print('第' + str(time) + "折交叉验证...")
                train_X, train_y = X[train_index], y[train_index]
                val_X, val_y = X[val_index], y[val_index]
                train_weight = weight[train_index,]
                model.fit(train_X, train_y, sample_weight=train_weight)
                score.append(model.score(val_X, val_y))
                time += 1
                # 在最后一折的数据很难验证
            print("模型 " + args.ModelType + " 在" + args.LabelType + " 标签下的10折平均交叉验证分数是: ", np.mean(score))

        if args.ModelType == 'XGB':
            # 0.46  归一化 0.64
            model = XGBClassifier()
            # 10折交叉验证
            score = []
            time = 1
            kf = KFold(n_splits=args.NumCrossValidation)
            for train_index, val_index in kf.split(X):
                print('第' + str(time) + "折交叉验证...")
                train_X, train_y = X[train_index], y[train_index]
                val_X, val_y = X[val_index], y[val_index]
                train_weight = weight[train_index,]
                model.fit(train_X, train_y, sample_weight=train_weight)
                score.append(model.score(val_X, val_y))
                time += 1
                # 在最后一折的数据很难验证
            print("模型 " + args.ModelType + " 在" + args.LabelType + " 标签下的10折平均交叉验证分数是: ", np.mean(score))

        if args.ModelType == 'KNN':
            # 感知机 0.36
            model = KNeighborsClassifier(n_neighbors=6)
            # 10折交叉验证
            score = []
            time = 1
            kf = KFold(n_splits=args.NumCrossValidation)
            for train_index, val_index in kf.split(X):
                print('第' + str(time) + "折交叉验证...")
                train_X, train_y = X[train_index], y[train_index]
                val_X, val_y = X[val_index], y[val_index]
                train_weight = weight[train_index,]
                model.fit(train_X, train_y)
                score.append(model.score(val_X, val_y))
                time += 1
                # 在最后一折的数据很难验证
            print("模型 " + args.ModelType + " 在" + args.LabelType + " 标签下的10折平均交叉验证分数是: ", np.mean(score))

    elif args.LabelType == 'pair':
        weights, X, ys = get_weight_input_label(Train_solver_runtime, Train_feature_time, Train_feature, args)
        if args.ModelType == 'GBDT':
            models = {}
            val_input = []
            val_label = {}
            keys = []
            scores = []
            kf = KFold(n_splits=args.NumCrossValidation)
            time = 1
            for train_index, val_index in kf.split(X):
                print('第' + str(time) + "折交叉验证...")
                for i in range(args.NumberSolver):
                    for j in range(i + 1, args.NumberSolver):
                        print("为求解器"+str(i) + ' 和 ' + "求解器 "+str(j)+" 生成模型")
                        # 取出标签和权重
                        key = str(i) + ',' + str(j)
                        y = ys[key]
                        weight = weights[key]
                        model = GradientBoostingClassifier(n_estimators=100)
                        train_X, train_y = X[train_index], y[train_index]
                        val_X, val_y = X[val_index], y[val_index]
                        train_weight = weight[train_index,]
                        model.fit(train_X, train_y, sample_weight=train_weight)
                        models[key]=model
                        keys.append(key)
                        val_input.append(val_X)
                        val_label[key] = val_y
                time += 1
                score = pair_val(models, keys, val_input[0], val_label, args)
                scores.append(score)
            print("模型 " + args.ModelType + " 在" + args.LabelType + " 标签下的10折平均交叉验证分数是: ", np.mean(scores))


def judge_solver(Train_solver_runtime, Train_feature_time, Train_feature, args):
    # 进行模型选择
    model_choice(Train_solver_runtime, Train_feature_time, Train_feature, args)
    # '''
    # train_instance = Train_feature_time.shape[0]
    # models = {}
    # # 先选择使用两个预求解器超时并且可以进行特征计算的实例
    # choice = choice_ids(Train_feature_time, Train_solver_runtime, args)
    # # list len 需要进行主求解器选择的实例  只有这些实例才需要进入求解器选择阶段
    #
    # # 制作输入数据
    # X = Train_feature.copy()
    # # 对数据进行归一化
    # X = normal_feature_data_process(X)
    # # shape (训练实例数,清洗后的特征列数)  删去无用的列，在列的维度上归一化
    # X = X[choice, :]
    # # shape (需要训练实例数,清洗后的特征列数)
    #
    # # 制作标签
    # label = {}
    # for i in range(args.NumberSolver):
    #     for j in range(i + 1, args.NumberSolver):
    #         y = np.ones(shape=(train_instance,), dtype=int)
    #         # shape (训练实例数, ) 为每一种组合生成一个标签 求解器i比求解器j快 标签为1 否则为-1
    #         for index in range(train_instance):
    #             if Train_solver_runtime[index, i] > Train_solver_runtime[index, j]:
    #                 y[index] = -1
    #         # 对应的求解器名作为key 标签为value 保存到字典label中
    #         # eg: '3,4': array (训练实例数, )
    #         key = str(i) + "," + str(j)
    #         label[key] = y
    #
    # # 生成 求解器数 * (求解器数-1) 个随机森林
    # for i in range(args.NumberSolver):
    #     for j in range(i + 1, args.NumberSolver):
    #         print("为求解器 " + str(i) + " 和求解器 " + str(j) + "构建随机森林")
    #         # kf = KFold(n_splits=args.NumCrossValidation)
    #         # model_solver = SVC(kernel='rbf', C=1.0, gamma='auto')
    #         model_solver = RandomForestClassifier(n_estimators=1000)
    #         # 每一种组合一个随机森林
    #         key = str(i) + "," + str(j)
    #         y = label[key]
    #         y = y[choice]
    #         print(X.shape)
    #         print(y.shape)
    #         score = cross_val_score(model_solver, X, y, cv=10).mean()
    #         print(score)
    #         '''
    #         for train_index, val_index in kf.split(X):
    #             key = str(i) + "," + str(j)
    #             y = label[key]
    #             train_X, train_y = X[train_index], y[train_index]
    #             val_X, val_y = X[val_index], y[val_index]
    #             model_solver.fit(train_X, train_y)
    #             score.append(model_solver.score(val_X, val_y))
    #             # 在最后一折的数据很难验证
    #         '''
    #         # 保存模型
    #         key = str(i) + "," + str(j)
    #         models[key] = model_solver
    #
    # return models


def infer(models, Test_solver_runtime, Test_feature_time, Test_feature, args):
    # 模型在测试集上的表现

    test_instance = Test_solver_runtime.shape[0]
    # 制作输入数据
    X = Test_feature_time.copy()
    # 对数据进行归一化
    X = normal_feature_data_process(X)
    # shape (测试实例数,清洗后的特征列数)  删去无用的列，在列的维度上归一化

    for index in range(test_instance):
        for i in range(args.NumberSolver):
            for j in range(i + 1, args.NumberSolver):
                key = str(i) + "," + str(j)
                print(key)
                model = models[key]
