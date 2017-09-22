import numpy as np
from urllib.request import urlopen

def getRawDataSet(url):
    dataSet = urlopen(url)
    filename = 'MLFex1_' + url.split('_')[1] + '_' + url.split('_')[2]
    with open(filename, 'wb') as fr:
        fr.write(dataSet.read())
    return filename
def getDataSet(filename):
    dataSet = open(filename, 'r')
    dataSet = dataSet.readlines()   # 将训练数据读出，存入dataSet变量中
    num = len(dataSet)  # 训练数据的组数
    # 提取X, Y
    X = np.zeros((num, 5))
    Y = np.zeros((num, 1))
    for i in range(num):
        data = dataSet[i].strip().split()
        X[i, 0] = 1.0
        X[i, 1] = np.float(data[0])
        X[i, 2] = np.float(data[1])
        X[i, 3] = np.float(data[2])
        X[i, 4] = np.float(data[3])
        Y[i, 0] = np.int(data[4])
    return X, Y
# sigmoid函数，返回函数值
def sign(x, w):
    if np.dot(x, w)[0] >= 0:
        return 1
    else:
        return -1
# 最原始的PLA训练算法
# X, Y，存储训练数据的矩阵，shape分别是(n+1)*m, m*1
# w，最初的系数矩阵，shape是(n+1)*1
# eta，参数
# updates，迭代次数
# 函数返回一个标志位（flag，用以说明训练是否结束，即最终得到的w是否完全fit训练数据），训练结果w， 实际迭代次数iterations
# 具体执行过程请阅读函数
def trainPLA_Naive(X, Y, w, eta, updates):
    iterations = 0  # 记录实际迭代次数
    num = len(X)    # 训练数据的个数
    flag = True
    for i in range(updates):
        flag = True
        for j in range(num):
            if sign(X[j], w) != Y[j, 0]:
                flag = False
                w += eta * Y[j, 0] * np.matrix(X[j]).T
                break
            else:
                continue
        if flag == True:
            iterations = i
            break
    return flag, iterations, w

# 使用此函数可直接解答15题
def question15():
    url = 'https://d396qusza40orc.cloudfront.net/ntumlone%2Fhw1%2Fhw1_15_train.dat'
    filename = getRawDataSet(url)
    X, y = getDataSet(filename)
    w0 = np.zeros((5, 1))
    eta = 1
    updates = 80
    flag, iterations, w = trainPLA_Naive(X, y, w0, eta, updates)
    print(flag)
    print(iterations)
    print(w)

question15()