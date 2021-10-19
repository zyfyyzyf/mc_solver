from collections import Counter
import matplotlib.mlab as mlab  
import matplotlib.pyplot as plt  
import numpy as np
# np.set_printoptions(threshold = 1e6)
import re
def sort_key(s):
    #sort_strings_with_embedded_numbers
    re_digits = re.compile(r'(\d+)')
    pieces = re_digits.split(s)  # 切成数字与非数字
    pieces[1::2] = map(int, pieces[1::2])  # 将数字部分转成整数
    return pieces
    
def time2score(time_data, cutoff):
    outscore = time_data.copy()
    number_instance = time_data.shape[0]
    number_solver = time_data.shape[1]
    for i in range(number_instance):
        for j in range(number_solver):
            if time_data[i][j] == cutoff:
                outscore[i][j] = time_data[i][j] * 10
            outscore[i][j] = cutoff * 10 - outscore[i][j]
    return outscore


def del_constant_col(feature_data):
    fun_feature_data = feature_data.copy()
    number_instance = feature_data.shape[0]
    number_feature = feature_data.shape[1]
    all_feature_col = []
    constant_col = []
    for i in range(number_feature):
        c = Counter(feature_data[:, i])
        if c[-512] == number_instance or c[0] == number_instance:
            constant_col.append(i)
    for i in range(number_feature):
        all_feature_col.append(i)
    cleaned_feature_data = np.delete(fun_feature_data, constant_col, axis=1)
    return cleaned_feature_data

def data_normalization(feature_data):
    cleaned_feature_data = feature_data.copy()
    cleaned_feature_data_mean = np.mean(cleaned_feature_data, axis=0)
    cleaned_feature_data_std = np.std(cleaned_feature_data, axis=0)
    output_feature_data = (cleaned_feature_data - cleaned_feature_data_mean) / cleaned_feature_data_std
    return output_feature_data

def draw_pie(input_data, labels, explode, colors):
    #不显示边框
    plt.gca().spines['right'].set_color('none')
    plt.gca().spines['top'].set_color('none')
    plt.gca().spines['left'].set_color('none')
    plt.gca().spines['bottom'].set_color('none')
    #绘制饼图
    plt.pie(x=input_data, #绘制数据
    labels=labels,#添加编程语言标签
    explode=explode,#突出显示Python
    colors=colors, #设置自定义填充色
    autopct='%.2f%%',#设置百分比的格式,保留3位小数
    radius=4.2,#设置饼图的半径(相当于X轴和Y轴的范围)
    # counterclock= False,#是否为逆时针方向,False表示顺时针方向
    wedgeprops= {'linewidth':0.5,'edgecolor':'black'},#设置饼图内外边界的属性值
    frame=1) #是否显示饼图的圆圈,1为显示
    #不显示X轴、Y轴的刻度值
    plt.xticks(())
    plt.yticks(())
    #plt.title('2018年8月的编程语言指数排行榜',fontproperties=my_font)
    plt.show()
    plt.savefig('analysis_3_1.eps', format='eps') 
    # plt.savefig('analysis_3_1.jpg') 