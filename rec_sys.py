__author__ = 'Freeman'

import numpy as np


with open('C:\Users\Freeman\Downloads\Test for web-analyst (2).csv', 'rb') as rh:
    reader = rh.readlines()
    for line in reader:
        l = line.split('\r')

l_data = []
for i in l[1:]:
    values = [int(_) for _ in i.split(';')]
    l_data.append(values[1:])

names = l[0].split(';')[1:]

##### preprocessing ######
freq_array = np.array(l_data)
low_val_array = freq_array < 5
freq_array[low_val_array] = 0
freq_array = freq_array[freq_array.sum(axis = 1) != 0]
zero_array_mask = freq_array != 0
freq_array_log = np.log(freq_array+1)

##### model params ######

n_features = 7
reg_alpha = 0.1
learning_rate_X = 0.00002
learning_rate_theta = 0.02

class CollRec:

    def __init__(self, n_features = 7, regularization = 0.1, learning_rate_X = 0.00001, learning_rate_theta = 0.01, iterations = 100):
        self.n_features = n_features
        self.regularization = regularization
        self.learning_rate_X = learning_rate_X
        self.learning_rate_theta = learning_rate_theta
        self.iter = iterations

    def fit(self, Y):
        Y_non_zero_array = Y != 0
        X_mean = Y.sum(axis = 0) / Y_non_zero_array.sum(axis = 0)
        # Y = Y - X_mean

        X = np.random.rand(Y.shape[1],self.n_features)  # report params
        theta = np.random.rand(Y.shape[0],self.n_features) # user params

        for _ in range(self.iter):
            y_array = theta.dot(X.T) * Y_non_zero_array
            error = y_array - Y

            s_error = np.power(error, 2).sum()
            s_theta = np.power(theta, 2).sum()
            s_X = np.power(X, 2).sum()
            cost = 0.5 *  (s_error + self.regularization * (s_theta + s_X))
            print (round(cost.sum(), 3), _)

            theta_grad = error.dot(X) + (self.regularization * theta)
            X_grad = error.T.dot(theta) + (self.regularization * X)

            theta = theta - (self.learning_rate_theta * theta_grad)
            X = X - (self.learning_rate_X * X_grad)

        self.X = X
        self.theta = theta
        self.cost = cost
        self._X_mean = X_mean

    def predict_new_user(self):
        return self._X_mean

    def predict_old_user(self, uid):
        return self.X.dot(self.theta[uid,:])  # + self._X_mean

def recommend_top_n_items(uid, top_n):
    if uid != 'new':
        rec_array = rec_sys.predict_old_user(uid)
        given_array = freq_array[uid, :]
        rec_array[given_array != 0] = 0
    else:
        rec_array = rec_sys.predict_new_user()
    idx = (-rec_array).argsort()[:top_n]
    return np.array(names)[idx]


rec_sys = CollRec(learning_rate_theta=learning_rate_theta, learning_rate_X=learning_rate_X)
rec_sys.fit(freq_array_log)
rec_sys.predict_old_user(13)
inx, report_name = recommend_top_n_items(13, 3)
inx, report_name = recommend_top_n_items('new', 3)


