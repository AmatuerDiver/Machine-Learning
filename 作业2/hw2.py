import sys
import pandas as pd
import numpy as np
import math

# 处理X_train
data = pd.read_csv('./dataset/X_train', encoding='big5')
data = data.iloc[:, 1:]  # 去掉第一列id，剩下的全部都是有用数据，即510维
raw_data = data.to_numpy()  # 转换为数组

##处理Y_train
y_data = pd.read_csv('./dataset/Y_train', encoding='big5')
y = y_data.iloc[:, 1:]  # 去掉第一列id，剩下的全部都是有用数据，即510维
y = y.to_numpy()  # 转换为数组

# mean_x = np.mean(raw_data, axis=0)  # 18 * 9
# std_x = np.std(raw_data, axis=0)  # 18 * 9
# for i in range(len(raw_data)):  # 12 * 471
#     for j in range(len(raw_data[0])):  # 18 * 9
#         if std_x[j] != 0:
#             raw_data[i][j] = (raw_data[i][j] - mean_x[j]) / std_x[j]

# 将x与y根据8:2拆成训练数据和验证数据
x_train_set = raw_data[: math.floor(len(raw_data) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = raw_data[math.floor(len(raw_data) * 0.8):, :]
y_validation = y[math.floor(len(y) * 0.8):, :]


print(len(x_train_set))
print(len(y_train_set))
print(len(x_validation))
print(len(y_validation))

# 定义sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 梯度下降算法初始一些参数
dim = 510 + 1  # 定义参数维数
bias = np.ones([43404, 1])  # 在训练集中新加一列全1数组作为bias
x = np.concatenate((x_train_set, bias), axis=1).astype(float)  # 将bias加入训练集最后
# print(data)
# print(data.shape)
w = np.ones([dim, 1])  # 创建一个511行一列的全1数组保存权重
eps = 0.0000000001  # 防止adagrad取0导致分母为0
learning_rate = 100  # 学习率
iter_time = 600  # 迭代次数

adagrad = np.zeros([dim, 1])  # 定义adagrad参数，便于之后的学习率更新
# print(gradient)
# print(gradient.shape)
for t in range(iter_time):
    gradient = np.dot(x.transpose(), sigmoid(np.dot(x, w)) - y_train_set)  # 计算梯度公式，跟线性回归几乎一样，只是需要先将其填入sigmoid函数
    adagrad += gradient ** 2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)  # 其余步骤跟线性回归一样
    if (t % 100 == 0):
        print("第" + str(t) + "次迭代了，再等会。。。")
np.save('weight.npy', w)  # 将参数保存下来

"""
使用validation来评估模型好坏
"""
# w = np.load('weight.npy')  # 加载参数
# bias_validation = np.ones([10852, 1])  # 跟训练集的处理方法一样
# x_validation = np.concatenate((x_validation, bias_validation), axis=1).astype(float)
# ans_y = sigmoid(np.dot(x_validation, w))  # 将验证集与参数做积并通过sigmoid函数得到预测值
# # print(ans_y)
# # print(ans_y.shape)
# # print(y_validation)
# # print(y_validation.shape)
# marx = ans_y - y_validation  # marx保存预测值与真实值之差，当差为0时代表预测正确，否则预测错误，所以只需统计marx矩阵中0元素的个数即可得到正确的预测数量
# # print(marx.shape)
# # print(marx)
# num = 0  # 正确的个数
# for i in range(len(marx)):  # 统计marx中0的个数
#     for j in range(len(marx[0])):
#         if marx[i][j] == 0:
#             num += 1
# # 正确率的计算公式是预测正确的数量/数据集总数量
# print(num / 10852)  # 100次迭代0.77，300次0.78，400次0.77，500次迭代时是0.79，600次0.8，1000次迭代时正确率为0.76，2000次迭代0.74，10000次迭代时正确率为0.60


# 处理X_test
data = pd.read_csv('./dataset/X_test', encoding='big5')
data = data.iloc[:, 1:]  # 去掉第一列id，剩下的全部都是有用数据，即510维
x_test = data.to_numpy()  # 转换为数组

bias_test = np.ones([27622, 1])  # 在训练集中新加一列全1数组作为bias
x_test = np.concatenate((x_test, bias_test), axis=1).astype(float)
w = np.load('weight.npy')  # 加载参数
ans_y = sigmoid(np.dot(x_test, w))  # 得到预测值

import csv

# 将预测的值填入submit中
with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'label']
    # print(header)
    csv_writer.writerow(header)
    for i in range(27622):
        row = [str(i), int(ans_y[i][0])]
        csv_writer.writerow(row)
        # print(row)
