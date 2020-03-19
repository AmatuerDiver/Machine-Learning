import sys
import pandas as pd
import numpy as np

# data = pd.read_csv('gdrive/My Drive/hw1-regression/train.csv', header = None, encoding = 'big5')
data = pd.read_csv('./dataset/train.csv', encoding='big5')

"""# **Preprocessing** 
取需要的數值部分，將 'RAINFALL' 欄位全 部補 0。
另外，如果要在 colab 重覆這段程式碼的執行，請從頭開始執行(把上面的都重新跑一次)，以避免跑出不是自己要的結果（若自己寫程式不會遇到，但 colab 重複跑這段會一直往下取資料。意即第一次取原本資料的第三欄之後的資料，第二次取第一次取的資料掉三欄之後的資料，...）。
"""

data = data.iloc[:, 3:]  # 行保留所有，列从第三列开始往后才保留，这样去除了数据中的时间、地点、参数等信息
# print(data)
data[data == 'NR'] = 0  # 将所有NR的值全部置为0方便之后处理
# print(data)
raw_data = data.to_numpy()  # 将data的所有数据转换为二维数据并用raw_data来保存
# print(raw_data)
"""# **Extract Features (1)**
![圖片說明](https://drive.google.com/uc?id=1LyaqD4ojX07oe5oDzPO99l9ts5NRyArH)
![圖片說明](https://drive.google.com/uc?id=1ZroBarcnlsr85gibeqEF-MtY13xJTG47)

將原始 4320 * 18 的資料依照每個月分重組成 12 個 18 (features) * 480 (hours) 的資料。
"""

month_data = {}
for month in range(12):  # month 从0-11 共12个月
    sample = np.empty([18, 480])  # 返回一个18行480列的数组，用来保存一个月的数据（一个月只有20天，一天24个小时）
    for day in range(20):  # day从0-19 共20天
        sample[:, day * 24: (day + 1) * 24] = raw_data[18 * (20 * month + day): 18 * (20 * month + day + 1),
                                              :]  # raw的行每次取18行，列取全部列。送到sample中（sample是18行480列）行给全部行，列只给24列，然后列往后增加
    month_data[month] = sample

"""# **Extract Features (2)**
![alt text](https://drive.google.com/uc?id=1wKoPuaRHoX682LMiBgIoOP4PDyNKsJLK)
![alt text](https://drive.google.com/uc?id=1FRWWiXQ-Qh0i9tyx0LiugHYF_xDdkhLN)

每個月會有 480hrs，每 9 小時形成一個 data，每個月會有 471 個 data，故總資料數為 471 * 12 筆，而每筆 data 有 9 * 18 的 features (一小時 18 個 features * 9 小時)。

對應的 target 則有 471 * 12 個(第 10 個小時的 PM2.5)
"""

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
# print(x)
# print(y)

"""# **Normalize (1)**"""

mean_x = np.mean(x, axis=0)  # 18 * 9
std_x = np.std(x, axis=0)  # 18 * 9
for i in range(len(x)):  # 12 * 471
    for j in range(len(x[0])):  # 18 * 9
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

"""#**Split Training Data Into "train_set" and "validation_set"**
這部分是針對作業中 report 的第二題、第三題做的簡單示範，以生成比較中用來訓練的 train_set 和不會被放入訓練、只是用來驗證的 validation_set。
"""

import math

x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8):, :]
y_validation = y[math.floor(len(y) * 0.8):, :]
# print(x_train_set)
# print(y_train_set)
# print(x_validation)
# print(y_validation)
# print(len(x_train_set))
# print(len(y_train_set))
# print(len(x_validation))
# print(len(y_validation))

"""# **Training**
![alt text](https://drive.google.com/uc?id=1xIXvqZ4EGgmxrp7c9r0LOVbcvd4d9H4N)
![alt text](https://drive.google.com/uc?id=1S42g06ON5oJlV2f9RukxawjbE4NpsaB6)
![alt text](https://drive.google.com/uc?id=1BbXu-oPB9EZBHDQ12YCkYqtyAIil3bGj)

(和上圖不同處: 下面的 code 採用 Root Mean Square Error)

因為常數項的存在，所以 dimension (dim) 需要多加一欄；eps 項是避免 adagrad 的分母為 0 而加的極小數值。

每一個 dimension (dim) 會對應到各自的 gradient, weight (w)，透過一次次的 iteration (iter_time) 學習。
"""

dim = 18 * 9 + 1  # 用来做参数vector的维数，加1是为了对bias好处理（还有个误差项）。即最后的h(x)=w1x1+w2x2+'''+WnXn+b
w = np.ones([dim, 1])  # 生成一个dim行1列的数组用来保存参数值，对比源码我这里改成了ones而不是zeros
x = np.concatenate((np.ones([12 * 471, 1]), x), axis=1).astype(
    float)  # np.ones来生成12*471行1列的全1数组，np.concatenate，axis=1表示按列将两个数组拼接起来，即在x最前面新加一列内容，之前x是12*471行18*9列的数组，新加一列之后变为12*471行18*9+1列的数组
learning_rate = 100  # 学习率
iter_time = 10000  # 迭代次数
adagrad = np.zeros([dim, 1])  # 生成dim行即163行1列的数组，用来使用adagrad算法更新学习率
eps = 0.0000000001  # 因为新的学习率是learning_rate/sqrt(sum_of_pre_grads**2),而adagrad=sum_of_grads**2,所以处在分母上而迭代时adagrad可能为0，所以加上一个极小数，使其不除0
for t in range(iter_time):
    loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y,
                                   2)) / 471 / 12)  # rmse loss函数是从0-n的(X*W-Y)**2之和/(471*12)再开根号，即使用均方根误差(root mean square error),具体可百度其公式，/471/12即/N(次数)
    if (t % 100 == 0):  # 每一百次迭代就输出其损失
        print(str(t) + ":" + str(loss))
    gradient = 2 * np.dot(x.transpose(), np.dot(x,
                                                w) - y)  # dim*1 x.transpose即x的转置，后面是X*W-Y,即2*(x的转置*(X*W-Y))是梯度，具体可由h(x)求偏微分获得.最后生成1行18*9+1列的数组。转置后的X，其每一行是一个参数，与h(x)-y的值相乘之后是参数W0的修正值，同理可得W0-Wn的修正值保存到1行18*9+1列的数组中，即gradient
    adagrad += gradient ** 2  # adagrad用于保存前面使用到的所有gradient的平方，进而在更新时用于调整学习率
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)  # 更新权重
np.save('weight.npy', w)  # 将参数保存下来

"""# **Testing**
![alt text](https://drive.google.com/uc?id=1165ETzZyE6HStqKvgR0gKrJwgFLK6-CW)

載入 test data，並且以相似於訓練資料預先處理和特徵萃取的方式處理，使 test data 形成 240 個維度為 18 * 9 + 1 的資料。
"""

# testdata = pd.read_csv('gdrive/My Drive/hw1-regression/test.csv', header = None, encoding = 'big5')
testdata = pd.read_csv('./dataset/test.csv', header=None, encoding='big5')
test_data = testdata.iloc[:, 2:]  # 取csv文件中的全行数即第3列到结束的列数所包含的数据
test_data[test_data == 'NR'] = 0  # 将testdata中的NR替换为0
test_data = test_data.to_numpy()  # 将其转换为数组
test_x = np.empty([240, 18 * 9], dtype=float)  # 创建一个240行18*9列的空数列用于保存textdata的输入
for i in range(240):  # 共240个测试输入数据
    test_x[i, :] = test_data[18 * i: 18 * (i + 1), :].reshape(1, -1)
# 下面是Normalize,且必须跟training data是同一种方法进行Normalize
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float)  # 在test_x前面拼接一列全1数组，构成240行，163列数据
"""# **Prediction**
說明圖同上

![alt text](https://drive.google.com/uc?id=1165ETzZyE6HStqKvgR0gKrJwgFLK6-CW)

有了 weight 和測試資料即可預測 target。
"""

w = np.load('weight.npy')
ans_y = np.dot(test_x, w)  # test data的预测值ans_y=test_x与W的积
# print(ans_y)


"""# **Save Prediction to CSV File**"""

import csv

with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)
