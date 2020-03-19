import numpy as np

# 数据预处理，将读到的数据分别存入X_train，Y_train,X_test中
np.random.seed(0)
X_train_fpath = './dataset/X_train'
Y_train_fpath = './dataset/Y_train'
X_test_fpath = './dataset/X_test'
output_fpath = './output_{}.csv'

# Parse csv files to numpy array
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype=float)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype=float)

# print(X_train)
# print(X_train.shape)
#
# print(Y_train)
# print(Y_train.shape)
#
# print(X_test)
# print(X_test.shape)

"""
定义接下来Normalize和划分数据集会用到的函数
"""


# Normalize函数，对数据进行正规化
def _normalize(X, train=True, specified_column=None, X_mean=None, X_std=None):  # 函数内定义的都是默认的参数，参数的意义底下有说明
    # This function normalizes specific columns of X.
    # The mean and standard variance of training data will be reused when processing testing data.
    #
    # Arguments:
    #     X: data to be processed
    #     train: 'True' when processing training data, 'False' for testing data
    #     specific_column: indexes of the columns that will be normalized. If 'None', all columns
    #         will be normalized.
    #     X_mean: mean value of training data, used when train = 'False'
    #     X_std: standard deviation of training data, used when train = 'False'
    # Outputs:
    #     X: normalized data
    #     X_mean: computed mean value of training data
    #     X_std: computed standard deviation of training data

    if specified_column == None:  # 如果等于None的话，意味着所有列都需要正规化
        specified_column = np.arange(X.shape[1])  # 新建一个数组，是0-X.shape[1]即0-509
    if train:  # 如果train为True，那么表示处理training data，否则就处理testing data,即不再另算X_mean和X_std
        X_mean = np.mean(X[:, specified_column], 0).reshape(1, -1)  # 对X的所有行以及特定列的数组中求各列的平均值（因为axis的参数为0），然后重组为一行的数组
        X_std = np.std(X[:, specified_column], 0).reshape(1, -1)  # 同X_mean

    X[:, specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)  # X_std加入一个很小的数防止分母除以0

    return X, X_mean, X_std


# 将训练集拆成训练集和验证集，默认值是0.25，可以调
def _train_dev_split(X, Y, dev_ratio=0.25):
    # This function spilts data into training set and development set.
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]


# Normalize training and testing data
X_train, X_mean, X_std = _normalize(X_train, train=True)
X_test, _, _ = _normalize(X_test, train=False, X_mean=X_mean, X_std=X_std)

# Split data into training set and development set，按9:1进行拆分
dev_ratio = 0.1
X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio=dev_ratio)

# print(X_train.shape)
# print(Y_train.shape)
# print(X_dev.shape)
# print(Y_dev.shape)

# 用来看看拆开的数据，以及数据参数维度是否正确
train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]
print('Size of training set: {}'.format(train_size))
print('Size of development set: {}'.format(dev_size))
print('Size of testing set: {}'.format(test_size))
print('Dimension of data: {}'.format(data_dim))

"""
会被训练中重复用到的函数
"""


# 按顺序打乱X和Y，即打乱后，X[i]对应的仍是Y[i]
def _shuffle(X, Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(len(X))  # 建立一个0-X的行-1的数组
    np.random.shuffle(randomize)  # 生成大小为randomize的随机列表，
    return (X[randomize], Y[randomize])


# 定义sigmoid函数
def _sigmoid(z):
    # Sigmoid function can be used to calculate probability.
    # To avoid overflow, minimum/maximum output value is set.
    # 为避免溢出，设置了最大最小值，即如果sigmoid函数的最小值比1e-8小，只会输出1e-8；而比1 - (1e-8)大，则只输出1 - (1e-8)
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))


# 逻辑回归的方程，输入是X，参数是w，bias是b，注意X与w都是数组，而b是一个数
# 实际上这跟在X中新加一列全1作为b的相乘数的结果是一样的
def _f(X, w, b):
    # This is the logistic regression function, parameterized by w and b
    #
    # Arguements:
    #     X: input data, shape = [batch_size, data_dimension]
    #     w: weight vector, shape = [data_dimension, ]
    #     b: bias, scalar
    # Output:
    #     predicted probability of each row of X being positively labeled, shape = [batch_size, ]
    # 在np.matmul(X, w)的基础上，数列中的每个值都加b得到最终的数列 matmul=dot
    return _sigmoid(np.matmul(X, w) + b)


# 将sigmoid中获得的值四舍五入转换成0或1(int型)，注意如果正好为0.5，(虽然几率很小)结果是0
# 实际上
def _predict(X, w, b):
    # This function returns a truth value prediction for each row of X
    # by rounding the result of logistic regression function.
    return np.round(_f(X, w, b)).astype(np.int)


# 返回模型正确率
def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - np.mean(np.abs(
        Y_pred - Y_label))  # np.abs(Y_pred - Y_label) 如果预测正确，则结果是0，否则结果是1，那么我们求mean平均值的话所得值是1的概率(mean相当于 1的个数/总个数),那么我们求0的概率就是1-it。这比我的方法两个for循环快多了
    return acc


# 交叉熵
def _cross_entropy_loss(y_pred, Y_label):
    # This function computes the cross entropy.
    #
    # Arguements:
    #     y_pred: probabilistic predictions, float vector，即还未放入_predict函数中的_f函数的结果
    #     Y_label: ground truth labels, bool vector 真正的结果，只有0和1两个元素
    # Output:
    #     cross entropy, scalar 输出是交叉熵是一个数,具体公式与推导可看笔记或Logistic Regression ppt11页
    cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred))
    return cross_entropy


# 返回要调整的w参数的gradient与b参数的gradient，具体公式与推导请看笔记或Logistic Regression ppt11页
def _gradient(X, Y_label, w, b):
    # This function computes the gradient of cross entropy loss with respect to weight w and bias b.
    y_pred = _f(X, w, b)  # 预测值，是float类型而不是bool
    pred_error = Y_label - y_pred  # 真实值-预测值，即误差
    w_grad = -np.sum(pred_error * X.T, 1)  # X.T就是X的转置，axis取值为1时代表将每一行的元素相加，实际上返回的是1行510列的数组
    b_grad = -np.sum(pred_error)  # 对b求偏微分后的结果，黑板上没有，但因为逻辑回归和线性回归的损失函数相似，可由线性回归对b进行求偏微分得到
    return w_grad, b_grad


"""
使用小批次进行训练时，当整个训练集的所有小批次都用过后，将所有训练集打散并重新分成很多小批次，继续进行一次，知道特定的次数结束为止。
"""
# Zero initialization for weights ans bias
# 使用0初始化w和b参数
w = np.zeros((data_dim,))
b = np.zeros((1,))

# print(w)
# print(w.shape)
# print(b)
# print(b.shape)

# Some parameters for training
max_iter = 20  # 迭代次数
batch_size = 8  # 训练的批次中的数据个数
learning_rate = 0.05  # 学习率
lamda = 0.004  # 正则化项系数0.004 0.8792

# Keep the loss and accuracy at every iteration for plotting
# 将每次迭代的损失和正确率都保存，以方便画出来
train_loss = []  # 训练集损失
dev_loss = []  # 验证集损失
train_acc = []  # 训练集正确率
dev_acc = []  # 验证集正确率

# Calcuate the number of parameter updates
# 记录参数更新的次数
step = 1

# Iterative training
for epoch in range(max_iter):  # max_iter迭代次数
    # Random shuffle at the begging of each epoch
    # 随机的将训练集X和Y按顺序打乱
    X_train, Y_train = _shuffle(X_train, Y_train)

    # Mini-batch training
    for idx in range(int(np.floor(train_size / batch_size))):  # 每个批次8个数据，一共48830个数据，共48830/8=6103次批次
        X = X_train[idx * batch_size:(idx + 1) * batch_size]  # 分别取X和Y中的对应8个数据(每个批次8个数据)
        Y = Y_train[idx * batch_size:(idx + 1) * batch_size]

        # Compute the gradient
        # 计算w参数和b参数的梯度
        w_grad, b_grad = _gradient(X, Y, w, b)

        # gradient descent update
        # learning rate decay with time
        # 更新参数，自适应学习率这次使用的是非常简单的学习率除以更新次数的根
        w = w - learning_rate / np.sqrt(step) * (w_grad + lamda * w)
        b = b - learning_rate / np.sqrt(step) * b_grad

        step = step + 1  # 更新次数+1

    # Compute loss and accuracy of training set and development set
    y_train_pred = _f(X_train, w, b)  # 计算预测的值，注意此时数据格式为float
    Y_train_pred = np.round(y_train_pred)  # 将数据格式转换为bool类型
    train_acc.append(_accuracy(Y_train_pred, Y_train))  # 将这一轮迭代的正确率记录下来
    train_loss.append(_cross_entropy_loss(y_train_pred, Y_train) / train_size)  # 将这一次迭代的损失记录下来

    y_dev_pred = _f(X_dev, w, b)  # 同样的方法处理验证集
    Y_dev_pred = np.round(y_dev_pred)
    dev_acc.append(_accuracy(Y_dev_pred, Y_dev))
    dev_loss.append(_cross_entropy_loss(y_dev_pred, Y_dev) / dev_size)

print('Training loss: {}'.format(train_loss[-1]))  # 输出最后依次迭代的结果
print('Development loss: {}'.format(dev_loss[-1]))
print('Training accuracy: {}'.format(train_acc[-1]))
print('Development accuracy: {}'.format(dev_acc[-1]))

np.save('weight_hw2.npy', w)  # 将参数保存下来

w = np.load('weight_hw2.npy')  # 加载参数

"""
调参结果：对相同的批次(8)，相同的迭代次数(10)时，学习率0.05最好 0.8765 loss 0.2837
        相同迭代次数10，学习率0.05，批次为10时，结果0.8759，loss0.2837
        学习率0.05，批次为8时,迭代次数20 ，结果0.8789，loss0.2834
"""
"""将最后的损失曲线和正确率曲线画出来"""

import matplotlib.pyplot as plt

# Loss curve
plt.plot(train_loss)
plt.plot(dev_loss)
plt.title('Loss')
plt.legend(['train', 'dev'])
plt.savefig('loss.png')
plt.show()

# Accuracy curve
plt.plot(train_acc)
plt.plot(dev_acc)
plt.title('Accuracy')
plt.legend(['train', 'dev'])
plt.savefig('acc.png')
plt.show()

# Predict testing labels
predictions = _predict(X_test, w, b)
with open(output_fpath.format('logistic'), 'w') as f:
    f.write('id,label\n')
    for i, label in enumerate(predictions):
        f.write('{},{}\n'.format(i, label))

# # Print out the most significant weights
# # 找到权重中最大的前十项，即关联结果的最紧密的参数
# ind = np.argsort(np.abs(w))[::-1] #将数组从小到大排好后从最后往前取
# with open(X_test_fpath) as f:
#     content = f.readline().strip('\n').split(',')
# features = np.array(content)
# for i in ind[0:10]:
#     print(features[i], w[i])
