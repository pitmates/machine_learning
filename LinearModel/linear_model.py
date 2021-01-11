import numpy as np
import pandas as pd
import os


class Logit:
    def __init__(self, X, Y):
        self.X = X
        W = np.random.rand(X.shape[1]) * 2 - 1
        self.W = W[:-1]
        self.Y = Y

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    def get_z(self):
        m, n = self.X.shape
        print(m, n, self.X.shape)

    def gradient(self):
        pass


    def logistic_regression(self, iteration=100, lr=1.0, method='gradient'):
        # self.beta = self.init_beta(n)
        pass

    def test(self):
        print("x:\n", self.X)
        print("w:\n", self.W)
        # print(self.b)
        self.get_z()



class LDA:
    def __init__(self):
        pass



if __name__ == "__main__":
    a = np.random.rand(3, 5)
    a = a.T
    logistic = Logit(a, [])
    logistic.test()
