import numpy as np

# Size of the points dataset.
m = 20

# Points x-coordinate and dummy value (x0, x1).
X0 = np.ones((m, 1))
X1 = np.arange(1, m + 1).reshape(m, 1)
X = np.hstack((X0, X1)) #X本来应该是仅仅是输入，但考虑到在h(x)=⊙0+⊙1x中我们还有个⊙0是需要考虑的参数，所以我们在每个输入前面加上1，即(1,Xi)，这样相乘时1乘参数⊙0还是⊙0，而Xi乘参数⊙1就是⊙1x,

# print(X0)
# print(X1)
# print(X)

# Points y-coordinate
y = np.array([
    3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
    11, 13, 13, 16, 17, 18, 17, 19, 21
]).reshape(m, 1)
# print(y)

# The Learning Rate alpha.
alpha = 0.01


def loss_function(theta, X, y):
    '''Error function J definition.'''
    diff = np.dot(X, theta) - y
    return (1. / 2 * m) * np.dot(np.transpose(diff), diff) #根据损失函数的定义（均方误差代价函数），来计算，diff是h(x)-y,diff的转置与diff的乘积即从1到m h(x)-y的平方的和


def gradient_function(theta, X, y):# 该函数返回一个1行2列的矩阵保存⊙1和⊙2的数值
    '''Gradient of the function J definition.'''
    diff = np.dot(X, theta) - y
    return (1. / m) * np.dot(np.transpose(X), diff) #将两个参数结合了，因为X的第一个值一定是1，所以即⊙0


def gradient_descent(X, y, alpha):
    '''Perform gradient descent.'''

    theta = np.array([1, 1]).reshape(2, 1)
    print('start loss :', loss_function(theta, X, y)[0, 0])
    gradient = gradient_function(theta, X, y)
    count = 0
    while not np.all(np.absolute(gradient) <= 1e-5):
        count += 1
        if (count % 10 == 0):
            print(count, end='')
            print("itr times loss:", loss_function(theta, X, y)[0, 0])
        theta = theta - alpha * gradient #theta即⊙，我们需要找的参数，使loss function得到最小值
        gradient = gradient_function(theta, X, y)
    return theta


optimal = gradient_descent(X, y, alpha)
print('optimal:', optimal) #⊙0和⊙1的最后数值
print('end loss :', loss_function(optimal, X, y)[0, 0])
