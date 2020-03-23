from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, adam

import numpy as np


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    number = 10000
    x_train = x_train[0:number]
    y_train = y_train[0:number]
    x_train = x_train.reshape(number, 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train
    x_test = x_test

    x_train = x_train / 255
    x_test = x_test / 255

    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data()

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

model = Sequential()  # 建立一个神经网络
model.add(Dense(input_dim=28 * 28, units=689, activation='relu'))  # 增加第一层输入层，维度是28*28，后面连接的第一层隐藏层的神经元个数是689，激活函数是relu
model.add(Dropout(0.7))  # 使用dropout
model.add(Dense(units=689, activation='relu'))  # 第二层隐藏层神经元个数689，激活函数是relu
model.add(Dropout(0.7))
model.add(Dense(units=10, activation='softmax'))  # 最后一层因为是手写数字识别，那么激活函数一定要用softmax

# 对模型进行设置，设置loss function，learning rate的优化函数，metrics用于设定评估当前训练模型的性能的评估函数
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#进行训练，使用x_train作为训练集y_train作为训练集对应的标签，设置batch_size和epochs从而进行小批次循环训练，并且可以使用GPU加速
model.fit(x_train, y_train, batch_size=100, epochs=20)

"""
一定要看train set的表现，因为test set表现不好可能是因为没有train好或者overfitting，
但train set的表现不好就说明不是overfitting，所以再根据需要换函数
手写数字分类时，loss func选择mse时的表现是
Train Acc: 0.31349998712539673
Test Acc: 0.31049999594688416
而选择categorical_crossentropy表现是
Train Acc: 0.899399995803833
Test Acc: 0.8867999911308289
激活函数从sigmoid调整为relu时得到
Train Acc: 0.9965000152587891
Test Acc: 0.9521999955177307
而不做Normalize的时候也是train不起来的
optimizer选择从SGD换成adam
Train Acc: 1.0
Test Acc: 0.9663000106811523
drpoout是train set上跑的太好了，而test set跑的相对差些，才可以使用.但此时train后 train set表现会降低，而test set上会增高
Train Acc: 0.9934999942779541
Test Acc: 0.9635999798774719
"""
# 输出模型在train set的正确率,一定要看train set的表现，因为test set表现不好可能是因为没有train好或者overfitting，
# 但train set的表现不好就说明不是overfitting，所以再根据需要换函数
result = model.evaluate(x_train, y_train, batch_size=10000)
print('\nTrain Acc:', result[1])

# 输出模型在test set的正确率
result = model.evaluate(x_test, y_test, batch_size=10000)
print('\nTest Acc:', result[1])
