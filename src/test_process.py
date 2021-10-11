def infer(models, Test_solver_runtime, Test_feature_time, Test_feature, args):
    # 模型在测试集上的表现

    test_instance = Test_solver_runtime.shape[0]
    # 制作输入数据
    X = Test_feature_time.copy()
    # 对数据进行归一化
    X = normal_feature_data_process(X)
    # shape (测试实例数,清洗后的特征列数)  删去无用的列，在列的维度上归一化

    for index in range(test_instance):
        # 对每个实例预测
        # 先判断实例是否能由预求解器求解器
        test
        for i in range(args.NumberSolver):
            for j in range(i + 1, args.NumberSolver):
                key = str(i) + "," + str(j)
                print(key)
                model = models[key]