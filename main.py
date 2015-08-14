__author__ = 'Freeman'

import numpy as np
from numpy import ma

with open('C:\Users\Freeman\Downloads\Test for web-analyst (2).csv', 'rb') as rh:
    reader = rh.readlines()
    for line in reader:
        l = line.split('\r')

l_data = []
for i in l[1:]:
    values = [int(_) for _ in i.split(';')]
    l_data.append(values[1:])

matrix_freq = np.matrix(l_data)
zero_mask_matrix = matrix_freq == 0
n_features = 10
reg_alpha = 0.1
learning_rate = 0.0000005

X = np.random.rand(matrix_freq.shape[1],n_features)  # report params
theta = np.random.rand(matrix_freq.shape[0],n_features) # user params

# matrix_freq_masked = ma.array(matrix_freq, mask=zero_mask_matrix)
# matrix_freq_masked = np.log(matrix_freq_masked)
# matrix_freq_log = np.log(matrix_freq+1)
matrix_freq_log = matrix_freq / matrix_freq


for i in range(100):
    y_matrix = theta.dot(X.T)
    # error = ma.array(y_matrix-matrix_freq_masked, mask=zero_mask_matrix)
    # error = y_matrix-matrix_freq
    # error = ma.array(y_matrix-matrix_freq, mask=zero_mask_matrix)
    # error = y_matrix-matrix_freq_log
    error = ma.array(y_matrix-matrix_freq_log, mask=zero_mask_matrix)

    s_error = np.power(error, 2).sum()
    reg_user = np.power(theta, 2).sum()
    reg_report = np.power(X, 2).sum()
    cost = 0.5 *  (s_error + reg_alpha * (reg_user + reg_report))
    print (cost.sum(), i)

    theta_grad = error.dot(X) + (reg_alpha * theta)
    X_grad = error.T.dot(theta) + (reg_alpha * X)

    X = X - (learning_rate * X_grad)
    theta = theta - (learning_rate * theta_grad)






matrix_freq = np.matrix([[5,0,0],[0,5,5],[0,5,5]])
n_features = 2
reg_alpha = 0.1
learning_rate = 0.05

X = np.random.rand(matrix_freq.shape[1],n_features)  # report params
theta = np.random.rand(matrix_freq.shape[0],n_features) # user params


for i in range(100):
    y_matrix = theta.dot(X.T)
    error = y_matrix-matrix_freq


    s_error = np.power(error,2).sum()
    reg_user = np.power(theta,2).sum()
    reg_report = np.power(X,2).sum()
    cost = 0.5 * (s_error + reg_alpha * (reg_user + reg_report))
    print (cost.sum(), i)

    theta_grad = error.dot(X) + (reg_alpha * theta)
    X_grad = error.T.dot(theta) + (reg_alpha * X)

    X = X - (learning_rate * X_grad)
    theta = theta - (learning_rate * theta_grad)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

heatmap, xedges, yedges = np.histogram2d(matrix_freq_log, bins=50)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.clf()
plt.imshow(heatmap, extent=extent)
plt.show()

fig, ax = plt.subplots()
heatmap = ax.pcolor(matrix_freq_log)
plt.show()




a = 0.01
mask = np.random.choice([False, True], matrix_freq_log.shape[0], p=[1-a, a])
matrix_freq_log[mask]

matrix_freq_over_3 = (matrix_freq > 3).astype(int)
axis_sum_more_1 = matrix_freq_over_3.sum(axis=1) != 0
matrix_freq_over_3[np.array(axis_sum_more_1)].shape
# matrix_freq_ones = matrix_freq / matrix_freq

mask = np.random.choice([False, True], matrix_freq_over_3.shape[0], p=[1-a, a])
matrix_freq_over_3[mask]

model = TSNE(n_components=2, random_state=0)
dt = model.fit_transform(matrix_freq_over_3[mask])
plt.scatter(dt[:,0],dt[:,1])