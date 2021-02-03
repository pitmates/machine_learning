import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import linear_model

from linear_model import LDA, Logit


scriptpath = os.path.abspath(__file__)
rootpath = os.path.dirname(os.path.dirname(scriptpath))

datapath = os.path.join(rootpath, 'data/LinearModel/watermelon3_0_Ch.csv')


def testLogistic():
    data = pd.read_csv(datapath).values

    X = data[:, 7:9].astype(float)
    y = data[:, 9]

    y[y == '是'] = 1
    y[y == '否'] = 0
    y = y.astype(int)
    
    logistic = Logit(X, y)
    A = logistic.regression()

    isGood = data[:, 9] == '是'
    isBad = data[:, 9] == '否'

    plt.scatter(data[:, 7][isGood], data[:, 8][isGood], c='k', marker='o')
    plt.scatter(data[:, 7][isBad], data[:, 8][isBad], c='r', marker='x')
    plt.xlabel("密度")
    plt.ylabel("含糖量")

    w1, w2, ic = A
    x1 = np.linspace(0, 1)
    y1 = -(w1*x1 + ic) / w2

    ax1, = plt.plot(x1, y1, label=r'my_regression')

    lr = linear_model.LogisticRegression(solver='lbfgs', C=10000)
    lr.fit(X, y)    
    print(lr.coef_[0])
    print(lr.intercept_)
    # lr_beta = np.concatenate((lr.coef_[0], lr.intercept_), axis=1)
    lr_beta = np.c_[lr.coef_, lr.intercept_]
    print(lr_beta)
    print(logistic.J_cost(lr_beta))

    w1_sk, w2_sk = lr.coef_[0, :]

    x2 = np.linspace(0, 1)
    y2 = -(w1_sk*x2 + lr.intercept_) / w2_sk

    ax2, = plt.plot(x2, y2, label=r'sklearn_logit')

    plt.legend(loc='upper right')
    plt.show()
    


if __name__ == "__main__":
    testLogistic()