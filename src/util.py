from collections import Counter
import matplotlib.mlab as mlab  
import matplotlib.pyplot as plt  
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import MultipleLocator
import pandas as pd 
# np.set_printoptions(threshold = 1e6)
import re
def create_cdf_data(time_data,flag):
    if flag == True:
        input_data = []
        for i in range(100):
            f_name = str(i) + '.cnf'
            if time_data[f_name] != 1800:
                input_data.append(time_data[f_name])
    else:
        input_data = time_data
    data = pd.DataFrame(input_data)
    denominator = 100
    Data=pd.Series(data[0])
    Fre=Data.value_counts()
    Fre_sort=Fre.sort_index(axis=0,ascending=True)
    Fre_df=Fre_sort.reset_index()
    Fre_df[0]=Fre_df[0]/denominator
    Fre_df.columns=['Rds','Fre']
    Fre_df['cumsum']=np.cumsum(Fre_df['Fre'])
    return Fre_df

def draw_CDF(oracle_time_data, top1_time_data,f_name):
    #创建画布
    plot=plt.figure()
    #只有一张图，也可以多张
    ax1=plot.add_subplot(1,1,1)
    #按照Rds列为横坐标，累计概率分布为纵坐标作图
    ax1.plot(oracle_time_data['Rds'],oracle_time_data['cumsum'])
    ax1.plot(top1_time_data['Rds'],top1_time_data['cumsum'])
    font = FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc', size=16)
    #图的标题
    ax1.set_title("solve_time_CDF")
    #横轴名
    ax1.set_xlabel("solve_time")
    #纵轴名
    ax1.set_ylabel("solved %")
    #横轴的界限
    ax1.set_xlim(-100,1800)
    ax1.set_ylim(0,1)
    x_major_locator=MultipleLocator(200)
    y_major_locator=MultipleLocator(0.1)
    ax1.xaxis.set_major_locator(x_major_locator)
    ax1.yaxis.set_major_locator(y_major_locator)
    #图片显示
    plt.grid()
    plt.show()
    # plt.savefig("analysis_3_5.eps", format='eps') 
    plt.savefig("analysis_3_5.png")

def draw_CDF_analysis_4(data):
    #创建画布
    plot=plt.figure()
    #只有一张图，也可以多张
    ax1=plot.add_subplot(1,1,1)
    #按照Rds列为横坐标，累计概率分布为纵坐标作图
    ax1.plot(data['Rds'],data['cumsum'])
    font = FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc', size=16)
    #图的标题
    ax1.set_title("feature_time_CDF")
    #横轴名
    ax1.set_xlabel("feature_time")
    #纵轴名
    ax1.set_ylabel("%")
    #横轴的界限
    ax1.set_xlim(-4,80)
    ax1.set_ylim(0,1)
    x_major_locator=MultipleLocator(5)
    y_major_locator=MultipleLocator(0.1)
    ax1.xaxis.set_major_locator(x_major_locator)
    ax1.yaxis.set_major_locator(y_major_locator)
    #图片显示
    plt.grid()
    plt.show()
    # plt.savefig("analysis_4.eps", format='eps') 
    plt.savefig("analysis_4.png")
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