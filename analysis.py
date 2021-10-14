import joblib

# 加载测试结果
test_result = joblib.load(r'/home/mc_zilla/save_model/test_result.pkl')
# dict 保存模型对100个测试样例的选择结果
print("test_result", test_result)