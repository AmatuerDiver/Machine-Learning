import numpy as np
import pandas as pd


# 数据预处理
def dataProcess(df):
    x_list, y_list = [], []
    # df替换指定元素，将空数据填充为0
    df = df.replace(['NR'], [0.0])
    # astype() 转换array中元素数据类型
    array = np.array(df).astype(float)
    # 将数据集拆分为多个数据帧
    for i in range(0, 4320, 18):  # i表i天的第一个参数，每一天有18个参数，所以步长是18
        for j in range(24 - 9):  # j表示每一天的第j组数据，（第一组数据是0-8，label是9的PM2.5；第二组数据是1-9，label是10的PM2.5）
            mat = array[i:i + 18, j:j + 9] #这里的+18和+9指步长，因为从0开始，所以第一组应该是[17,8]
            label = array[i + 9, j + 9]  # PM2.5是第十个参数，相对于j的话，j的前9个都用来做输入（0-8），所以标签就是第j+9个
            x_list.append(mat)
            y_list.append(label)
    x = np.array(x_list) # 3维数组，第一个数字表示有几个二维数组，后面两个分别是二维数组的行和列，一天可以有15个二维数组数据，一共240天就是3600个二维数组数据
    y = np.array(y_list)
    print(x.shape)
    print(y.shape)
    return x, y, array


def main():
    # 从csv中读取有用的信息
    # 由于大家获取数据集的渠道不同，所以数据集的编码格式可能不同
    # 若读取失败，可在参数栏中加入encoding = 'gb18030'
    df = pd.read_csv('./dataset/train.csv', usecols=range(3, 27), encoding='big5')
    x, y, _ = dataProcess(df)
    # #划分训练集与验证集
    # x_train, y_train = x[0:3200], y[0:3200]
    # x_val, y_val = x[3200:3600], y[3200:3600]
    # epoch = 2000 # 训练轮数
    # # 开始训练
    # w, b = train(x_train, y_train, epoch)
    # # 在验证集上看效果
    # loss = validate(x_val, y_val, w, b)
    # print('The loss on val data is:', loss)


if __name__ == '__main__':
    main()
