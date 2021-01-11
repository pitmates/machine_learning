import numpy as np
import os


class Logit:
    def __init__(self, X, Y):
        self.X = X
        W = np.random.rand(X.shape[1]+1) * 2 - 1
        self.W = W[:-1]
        self.b = W[-1]
        self.Y = Y

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def get_z(self):
        return np.dot(self.X, self.W.T) + self.b

    def test(self):
        print("x:\n", self.X)
        print("w:\n", self.W)
        print("b:\n", self.b)
        # print(self.b)
        print("\n", self.get_z())



class LDA:
    def __init__(self):
        pass



if __name__ == "__main__":
    a = np.random.rand(3, 5)
    a = a.T
    logistic = Logit(a, [])
    logistic.test()
