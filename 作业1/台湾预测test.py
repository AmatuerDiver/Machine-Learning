import sys
import pandas as pd
import numpy as np

# data = pd.read_csv('gdrive/My Drive/hw1-regression/train.csv', header = None, encoding = 'big5')
data = pd.read_csv('./dataset/train.csv', encoding='big5')

"""# **Preprocessing** 
取需要的數值部分，將 'RAINFALL' 欄位全 部補 0。
另外，如果要在 colab 重覆這段程式碼的執行，請從頭開始執行(把上面的都重新跑一次)，以避免跑出不是自己要的結果（若自己寫程式不會遇到，但 colab 重複跑這段會一直往下取資料。意即第一次取原本資料的第三欄之後的資料，第二次取第一次取的資料掉三欄之後的資料，...）。
"""

data = data.iloc[:, 3:]
# print(data)
data[data == 'NR'] = 0
# print(data)
raw_data = data.to_numpy()
# print(raw_data.shape)
# print(raw_data)

month_data = {}
for month in range(12):  # month 从0-11 共12个月
    sample = np.empty([18, 480])  # 返回一个18行480列的数组，用来保存一个月的数据（一个月只有20天，一天24个小时）
    for day in range(20):  # day从0-19 共20天
        sample[:, day * 24: (day + 1) * 24] = raw_data[18 * (20 * month + day): 18 * (20 * month + day + 1), :]
    month_data[month] = sample

# print(month_data)
x = np.empty([12 * 471, 18 * 9],
             dtype=float)  # 一共480个小时，每9个小时一个数据（480列最后一列不可以计入，因为如果取到最后一行那么最后一个数据便没有了结果{需要9个小时的输入和第10个小时的第10行作为结果}），480-1-9+1=471。471*12个数据集按行排列，每一行一个数据；数据是一个小时有18个特征，而每个数据9个小时，一共18*9列
y = np.empty([12 * 471, 1], dtype=float)  # 结果是471*12个数据，每个数据对应一个结果，即第10小时的PM2.5浓度
for month in range(12):  # month 0-11
    for day in range(20):  # day 0-19
        for hour in range(24):  # hour 0-23
            if day == 19 and hour > 14:  # 取到raw_data中的最后一块行为18，列为9的块之后，就不可以再取了，再取就会超过界限了，具体看Extract Features (2)1图片
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:, day * 24 + hour: day * 24 + hour + 9].reshape(1,
                                                                                                                     # 取对应month：行都要取，列取9个，依次进行，最后将整个数据reshape成一行数据(列数无所谓)。然后赋给x，x内的坐标只是为了保证其从0-471*12
                                                                                                                     -1)  # vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][
                9, day * 24 + hour + 9]  # value,结果对应的行数一直是第9列（即第10行PM2.5）然后列数随着取得数据依次往后进行
# print(x.shape)
# print(y.shape)

# print(x)
# print(y)

"""# **Normalize (1)**"""
"""
数据的标准化（normalization）是将数据按比例缩放，使之落入一个小的特定区间。
在某些比较和评价的指标处理中经常会用到，去除数据的单位限制，将其转化为无量纲的纯数值，
便于不同单位或量级的指标能够进行比较和加权。
最常见的标准化方法就是Z标准化，也是SPSS中最为常用的标准化方法，spss默认的标准化方法就是z-score标准化。
也叫标准差标准化，这种方法给予原始数据的均值（mean）和标准差（standard deviation）进行数据的标准化。
经过处理的数据符合标准正态分布，即均值为0，标准差为1，注意，一般来说z-score不是归一化，而是标准化，归一化只是标准化的一种[lz]。
其转化函数为：
x* = (x - μ ) / σ
其中μ为所有样本数据的均值，σ为所有样本数据的标准差。
https://www.cnblogs.com/chenyusheng0803/p/9867579.html
"""
mean_x = np.mean(x, axis=0)  # 18 * 9 求均值，axis = 0表示对各列求均值，返回 1* 列数 的矩阵
# print(mean_x.shape)
# print(mean_x)
std_x = np.std(x, axis=0)  # 18 * 9 求标准差，axis = 0表示对各列求均值，返回 1* 列数 的矩阵
# print(std_x.shape)
# print(std_x)
for i in range(len(x)):  # 12 * 471
    for j in range(len(x[0])):  # 18 * 9
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

# 将训练数据拆成训练数据：验证数据=8:2，这样用来验证
import math

x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8):, :]
y_validation = y[math.floor(len(y) * 0.8):, :]
print(x_train_set)
print(y_train_set)
print(x_validation)
print(y_validation)
print(len(x_train_set))
print(len(y_train_set))
print(len(x_validation))
print(len(y_validation))

"""# **Training**
因為常數項的存在，所以 dimension (dim) 需要多加一欄；eps 項是避免 adagrad 的分母為 0 而加的極小數值。

每一個 dimension (dim) 會對應到各自的 gradient, weight (w)，透過一次次的 iteration (iter_time) 學習。
"""
#
# dim = 18 * 9 + 1  # 用来做参数vector的维数，加1是为了对bias好处理（还有个误差项）。即最后的h(x)=w1x1+w2x2+'''+WnXn+b
# w = np.ones([dim, 1])  # 生成一个dim行1列的数组用来保存参数值，对比源码我这里改成了ones而不是zeros
# x_train_set = np.concatenate((np.ones([4521, 1]), x_train_set), axis=1).astype(
#     float)  # np.ones来生成12*471行1列的全1数组，np.concatenate，axis=1表示按列将两个数组拼接起来，即在x最前面新加一列内容，之前x是12*471行18*9列的数组，新加一列之后变为12*471行18*9+1列的数组
# learning_rate = 100  # 学习率
# iter_time = 10000  # 迭代次数
# adagrad = np.zeros([dim, 1])  # 生成dim行即163行1列的数组，用来使用adagrad算法更新学习率
# eps = 0.0000000001  # 因为新的学习率是learning_rate/sqrt(sum_of_pre_grads**2),而adagrad=sum_of_grads**2,所以处在分母上而迭代时adagrad可能为0，所以加上一个极小数，使其不除0
# for t in range(iter_time):
#     loss = np.sqrt(np.sum(np.power(np.dot(x_train_set, w) - y_train_set,
#                                    2)) / 4521)  # rmse loss函数是从0-n的(X*W-Y)**2之和/(471*12)再开根号，即使用均方根误差(root mean square error),具体可百度其公式，/471/12即/N(次数)
#     if (t % 100 == 0):  # 每一百次迭代就输出其损失
#         print(str(t) + ":" + str(loss))
#     gradient = 2 * np.dot(x_train_set.transpose(), np.dot(x_train_set,
#                                                 w) - y_train_set)  # dim*1 x.transpose即x的转置，后面是X*W-Y,即2*(x的转置*(X*W-Y))是梯度，具体可由h(x)求偏微分获得.最后生成1行18*9+1列的数组。转置后的X，其每一行是一个参数，与h(x)-y的值相乘之后是参数W0的修正值，同理可得W0-Wn的修正值保存到1行18*9+1列的数组中，即gradient
#     adagrad += gradient ** 2  # adagrad用于保存前面使用到的所有gradient的平方，进而在更新时用于调整学习率
#     w = w - learning_rate * gradient / np.sqrt(adagrad + eps)  # 更新权重
# np.save('weight.npy', w)  # 将参数保存下来

w = np.load('weight.npy')
# 使用x_validation和y_validation来计算模型的准确率，因为X已经normalize了，所以不需要再来一遍，只需在x_validation上添加新的一列作为bias的乘数即可
x_validation = np.concatenate((np.ones([1131, 1]), x_validation), axis=1).astype(float)
ans_y = np.dot(x_validation, w)
# print(ans_y)
# print(y_validation)
loss = np.sqrt(np.sum(np.power(ans_y - y_validation, 2)) / 1131)
print(loss)
