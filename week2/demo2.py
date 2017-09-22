import scipy.io
import numpy as np
from random import randint
data = scipy.io.loadmat("T1.mat")["data"]
target = scipy.io.loadmat("T1.mat")["target"]
data_test = scipy.io.loadmat("T1.mat")["data_test"]
target_test = scipy.io.loadmat("T1.mat")["target_test"]

N = np.shape(data_test)[0]
total = np.shape(data)[0]

def sign(x):
    new = []
    for i in range(np.shape(x)[0]):
        if x[i][0]>0:
            new.append([1])
        else:
            new.append([-1])
    return new

w = w_p = np.zeros((5,1))
h_w = sign(data.dot(w))
h_p = sign(data.dot(w_p))

#error_w = np.sum(np.absolute(h_w - target) / 2)

loop = 0
while (loop <= 100):
    index = np.where(np.absolute(h_w - target) != 0)
    j = randint(0, len(index[0])-1)
    w = w + np.reshape(data[index[0][j]]*target[index[0][j]],(5,1))
    loop+=1
    h_w = sign(data.dot(w))
    h_p = sign(data.dot(w_p))
    error_w = np.sum(np.absolute(h_w - target) / 2)
    error_p = np.sum(np.absolute(h_p - target) / 2)
    if error_w < error_p:
        w_p = w

print(error_p/500)




