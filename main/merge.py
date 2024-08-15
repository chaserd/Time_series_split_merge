import matplotlib
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal
from scipy.signal import argrelextrema
matplotlib.rcParams.update({'font.size': 12})
from aeon.datasets import load_classification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X, Y = load_classification("Libras", split="test")

## 重庆邮电大学  蒲世威  Telephone:18290248811   Email:1220399075.com
### 采用 l1 滤波 然后求极值
def inputek(dfi):
    y = np.array(dfi)
    n = y.size
    x = y

    ####  线段找到极值点
    arg = []
    arg.append([0])
    arg.append(argrelextrema(x, np.greater)[0].tolist())
    arg.append(argrelextrema(x, np.less)[0].tolist())
    arg.append([n])
    list_2 = [int(x) for arg in arg for x in arg]
    index = sorted(list_2)
    xi = []
    count = 0
    temp_k = 1
    temp_b = 1
    fc = 0
    for i in range(len(index) - 1):
        y = x[index[i]:index[i + 1]]
        value2 = y[-1]
        value1 = y[0]
        leng = index[i + 1] - index[i]
        x1 = range(count, count + len(y))
        count = leng + count
        if (len(y) >= 2):
            slope, intercept = np.polyfit(x1, y, 1)
            fc = np.var(y)
            xi.append([slope, intercept, fc, leng, [value1, value2], count, y])  # k,b
            temp_k = slope
            temp_b = intercept
        else:
            xi.append([temp_k, temp_b, fc, 1, [value1, value2], count, y])  # k,b
    return xi


### 采用 l1 滤波 然后求极值 ，带参数的线段分割
def inputek1(dfi, va):
    y = np.array(dfi)
    n = y.size
    x = y
    arg = []
    arg.append([0])
    arg.append(argrelextrema(x, np.greater)[0].tolist())
    arg.append(argrelextrema(x, np.less)[0].tolist())
    arg.append([n])
    list_2 = [int(x) for arg in arg for x in arg]
    index = sorted(list_2)
    xi = []
    si = 0.2
    ys = []
    lidus = []
    # print("一共有线段数 ",len(index))
    for i in range(len(index) - 1):
        y = x[index[i]:index[i + 1]]
        leng = index[i + 1] - index[i]
        x1 = range(1, len(y) + 1)  # 时间t
        if (len(y) > 1):
            slope, intercept = np.polyfit(x1, y, 1)  #
            lu = np.mean(y) - va * np.std(y)
            up = np.mean(y) + va * np.std(y)
            fc = np.var(y)
            lidu = []
            for i in range(len(x1)):
                lidus.append([slope, intercept, fc, leng, lu, up,
                              np.exp(-((y[i] - (slope * x1[i] + intercept)) ** 2) / (2 * (lu ** 2) * (1 - si))),
                              1 - np.exp(-((y[i] - (slope * x1[i] + intercept)) ** 2) / (2 * (lu ** 2) * (1 + si))),
                              np.exp(-((y[i] - (slope * x1[i] + intercept)) ** 2) / (2 * (up ** 2) * (1 - si))),
                              1 - np.exp(-((y[i] - (slope * (x1[i]) + intercept)) ** 2) / (2 * (up ** 2) * (1 + si)))])
            xi.append(lidu)  # k,b，var，
            res = pd.DataFrame(lidus, columns=['k', 'b', 'var', 'length', 'lu', 'up', 'l1', 'l2', 'u1', 'u2'])
    ys = dfi['y']
    return res, ys

### 对线段合并函数
def merge_l1(l1, k_1, s_1):
    ### l_.columns=["k", "b","var","leng","y","count","yx"]
    ilist = []
    ## 根据2筛选
    k1 = pd.DataFrame(l1)[0].diff().mean() / k_1
    s = pd.DataFrame(l1)[3].mean() / s_1
    len1 = len(l1)
    print("len1==========", len1)
    l2 = l1
    for i in range(len1 - 1):
        if (abs(l1[i][0] - l1[i + 1][0])) < k1:  ## k
            ##print(l1[i][0]-l1[i+1][0])
            ####
            tana = (l1[i][0] - l1[i + 1][0]) / 1 + (l1[i][0] * l1[i + 1][0])
            k = (l1[i + 1][4][1] - l1[i][4][0]) / (l1[i][3] + l1[i + 1][3])  ##新的K
            b = l1[i][4][1] - k * l1[i][5]  ## 新的b
            var = l1[i][2]
            lens = l1[i][3] + l1[i + 1][3]

            # list1=np.concatenate(l1[i][5],l1[i+1][5])

            l1[i + 1][0] = k
            l1[i + 1][1] = b
            l1[i + 1][2] = var
            l1[i + 1][3] = lens
            # l1[i+1][4]=list1
            ilist.append(i)
        elif ((abs(l1[i][3])) < s):  ## 根据1 筛选
            tana = (l1[i][0] - l1[i + 1][0]) / 1 + (l1[i][0] * l1[i + 1][0])
            k = (l1[i + 1][4][1] - l1[i][4][0]) / (l1[i][3] + l1[i + 1][3])  ##新的K
            b = l1[i][4][1] - k * l1[i][5]  ## 新的b
            var = l1[i][2]
            lens = l1[i][3] + l1[i + 1][3]

            # list2=np.concatenate(l1[i][5],l1[i+1][5])
            l1[i + 1][0] = k
            l1[i + 1][1] = b
            l1[i + 1][2] = var
            l1[i + 1][3] = lens
            # l1[i+1][4]=list2
            ilist.append(i)
    return ilist

def line_mergre(df_temp, k_1, s_1):
    b, a = signal.butter(4, 0.2, 'lowpass')  # 8表示滤波器的阶数
    # 通过 l1 滤波过滤
    # 对于每一行，通过列名name访问对应的元素
    df_temp = df_temp

    ###  只对第二列进行了   假设一个 m*n 矩阵 变成了一个 5（粒属性）*k (粒个数） * j（样本数）
    ###  合并前进行滤波去噪
    filtedData = signal.filtfilt(b, a, df_temp + 0.1)
    #     print(filtedData)
    l1 = inputek(filtedData)  # 输出每一
    l_ = pd.DataFrame(l1)
    l_.columns = ["k", "b", "var", "leng", "y", "count", "yx"]
    index = l_['count'].tolist()
    index = list(map(int, index))
    print("合并前一共有线段数 ", len(index))
    ###  保存合并之前的线段
    l_.to_csv("l1.csv")
    ###########   进行线段合并
    ilist = merge_l1(l1, k_1, s_1)

    #     ilistd=pd.DataFrame(ilist)
    #     ilistd.to_csv("ilist.csv")

    ### 删除多余线段
    l2 = np.delete(l1, ilist, axis=0)  # 二维数组删除0+1、1+1行
    # l2=l1
    l2 = pd.DataFrame(l2)
    l2.columns = ["k", "b", "var", "leng", "y", "count", "yx"]
    index = l2['count'].tolist()
    index = list(map(int, index))
    print("合并后一共有线段数 ", len(index))

    return l2

red_sum = []
##  假设采用本地数据  可删除
list_name = [
    "ArticularyWordRecognition",
    # "BasicMotions",
    # "Cricket",
    # "EigenWorms",
    # "FingerMovements",
    # "HandMovementDirection",
    # "JapaneseVowels",
    # "LSST",
    # "NATOPS",
    # "PhonemeSpectra",
    # "RacketSports",
    # "SelfRegulationSCP1",
    # "SelfRegulationSCP2",
    # "SpokenArabicDigits",
]

# list_a = []
# dicts = dict()
# dictts = dict()
# for str1 in list_name:
#     file_name = 'C:\\Users\\dh\\Downloads\\Multivariate_arff\\' + str1 + '\\' + str1 + '_TRAIN.arff'
#     dict1 = dict()
#     # file_name=r'C:\Users\dh\Desktop\Univariate_arff\Haptics\Haptics_TRAIN.arff'
#     print(file_name)
#     data, meta = arff.loadarff(open(file_name, encoding="utf-8"))
#     df = pd.DataFrame(data)
#     arr, _ = pd.factorize(df.iloc[:, 1])
#     list1 = []
#     for i in range(len(data)):
#         list1.append(pd.DataFrame(data[i][0]).astype("float64").values)
#     tensor1 = torch.from_numpy(np.array(list1))
#
#     for leni in range(tensor1.shape[1]):
#         #         print("tensor1",tensor1.shape[1])
#         print(leni)
#         print("dataFrame", pd.DataFrame(line_mergre(pd.DataFrame(tensor1[:, leni, :]), 0.5, 0.5)).fillna(0).values)
#         # dict1[leni+1]=
#         # torch.from_numpy(pd.DataFrame(line_mergre(pd.DataFrame(tensor1[:,leni,:]),0.5,0.5)).fillna(0).values)
#
#     # print(np.mean(dict1[1]))
#
#     # df = pd.read_csv("C:\\Users\\dh\\Desktop\\模糊和领域粗糙集\\代码\\fncf-far-master\\fncf-far-master\\code\\DataSet\\" + dataa[k] + ".csv", header=None)
#     #         data = df.values
#     data = dictToDf(dict1, arr)


# 类型0  一个一维数组   类型 1 一个二维 dataFrame 结构  类型3 一个三维结构
def liner_granulate(data,type):

    if type == 0:
        return line_mergre(data,0.5,0.5)
    else:

        return  0
# 使用优化后的函数
if __name__ == "__main__":
    ###  进行合并
    input_data=X[:,1,1]
    l2 = liner_granulate(input_data,0)

    ###  进行数据填补
    lss=[]
    index=l2["count"]
    x=input_data
    for i in range(len(index)):
        if(i==0):
            y=x[0:index[i]]
            lss.append(y.tolist() )
        else:
            y=x[index[i-1]:index[i]]
            lss.append(y.tolist() )
    l2["yx"]=lss

    data = l2

    # 数据画图
    x_end = 0
    y_end = 0
    # 绘制每条线段
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 1, 2)
    #data =pd.read_csv(r"C:\Users\dh\Desktop\模糊和领域粗糙集\代码\线性粒化结果Construction.csv").iloc[:,1:]
    for i in range(len(data)):
        k, b, var, leng,y1,count,yx = data.values[i]
        #print(yx)
        #yx = ast.literal_eval(yx)
        # yx_values = yx[1:-1].split()
        #
        # # 将字符串列表转换为浮点数列表
        # yx = [float(value) for value in yx_values]
        #yx=[yx[0],yx[-1]]
        #print(yx)
        x_start = count-leng
        y_start = x_start* k +b
        x_end = count
        y_end = k * x_end + b
        #print([x_start, x_end])
        plt.plot([x_start, x_end], [y_start, y_end], linewidth=15)
        plt.plot(range(x_start, x_end),yx)

    # 添加标题和标签
    # 添加标题和标签
    plt.title('After merging',fontsize=16)
    plt.xlabel('X',fontsize=14)
    plt.ylabel('Y', rotation=0,fontsize=14)
    plt.grid(True)
    x_end = 0
    y_end = 0
    # 绘制每条线段
    plt.subplot(2, 1, 1)
    data =pd.read_csv(r"C:\Users\dh\Desktop\模糊和领域粗糙集\代码\l1.csv").iloc[:,1:]
    for i in range(len(data)):
        k, b, var, leng,y1,count,yx = data.values[i]
        #print(yx)
        #yx = ast.literal_eval(yx)
        yx_values = yx[1:-1].split()

        # 将字符串列表转换为浮点数列表
        yx = [float(value) for value in yx_values]
        #print(yx)
        x_start = count-leng
        y_start = x_start* k +b
        x_end = count
        y_end = k * x_end + b
        #print([x_start, x_end])
        plt.plot([x_start, x_end], [y_start, y_end], linewidth=15)
        plt.plot(range(x_start, x_end),yx)

    # 添加标题和标签
    plt.title('Before merging',fontsize=16)
    #plt.xlabel('X-axis')
    plt.ylabel('Y', rotation=0,fontsize=14)


    # 显示图形
    plt.grid(True)
    plt.show()

    ####  对数据合并前进行的数据噪声平滑 （画图）

    y = input_data
    n = y.size
    x = y
    arg = []
    arg.append([0])
    arg.append(argrelextrema(x, np.greater)[0].tolist())
    arg.append(argrelextrema(x, np.less)[0].tolist())
    arg.append([n - 1])
    list_2 = [int(x) for arg in arg for x in arg]
    index = sorted(list_2)
    print(len(index))
    plt.plot(y)
    plt.plot(index, y[index], 'x')
    plt.show()

    b, a = signal.butter(4, 0.2, 'lowpass')  # 8表示滤波器的阶数
    # 通过 l1 滤波过滤
    # 对于每一行，通过列名name访问对应的元素
    df_temp = input_data

    filtedData = signal.filtfilt(b, a, df_temp + 0.1)

    y = filtedData
    n = y.size
    x = y
    arg = []
    arg.append([0])
    arg.append(argrelextrema(x, np.greater)[0].tolist())
    arg.append(argrelextrema(x, np.less)[0].tolist())
    arg.append([n - 1])
    list_2 = [int(x) for arg in arg for x in arg]
    index = sorted(list_2)
    print(len(index))
    plt.plot(y)
    plt.plot(index, y[index], 'x')
    plt.show()
