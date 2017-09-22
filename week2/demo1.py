import scipy.io
import numpy as np
import random

x = scipy.io.loadmat("data.mat")["x"]
target = scipy.io.loadmat("data.mat")["target"]

w = np.zeros((5,1))
loop =0

def sign(x):
    new = []
    for i in range(np.shape(x)[0]):
        if x[i][0]>0:
            new.append([1])
        else:
            new.append([-1])
    return new

N = np.shape(x)[0]

h_w = sign(x.dot(w))

error_w = np.sum(np.absolute(h_w - target) / 2)

while (error_w != 0):
    index = np.where(np.absolute(h_w - target) != 0)
    j = random.randint(0, len(index[0]))
    w = w + np.reshape(0.5* x[index[0][j]]*target[index[0][j]],(5,1))
    loop+=1
    h_w = sign(x.dot(w))
    error_w = np.sum(np.absolute(h_w - target) / 2)


print(loop)




