from __future__ import print_function
import numpy as np 
import matplotlib.pyplot as plt
np.random.seed(22)

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N) # lớp 1
X1 = np.random.multivariate_normal(means[1], cov, N) # lớp -1 
X = np.concatenate((X0.T, X1.T), axis = 1) # tất cả dữ liệu 
y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1) # nhãn
print(y)