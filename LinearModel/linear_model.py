import numpy as np
import os


class Logit:
    def __init__(self, X, Y, lr=1.2, it=1000, method='grad'):
        self.X = X
        self.Y = Y
        self.lr = self.initLR(lr)
        self.epoch = it
        self.method = method
        self.Beta = self.initBeta()
        
    def initLR(self, lr):
        _lr = []
        for i in range(10):
            _lr.append(lr - i*0.1)
        print(_lr)
        return _lr

    def initBeta(self):
        return np.random.randn(self.X.shape[1]+1, 1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def J_cost(self):
        X_Aug = np.concatenate((self.X, np.ones((self.X.shape[0], 1))), axis=1)
        beta = self.Beta.reshape(-1, 1)
        y = self.Y.reshape(-1, 1)
        betaX = np.dot(X_Aug, beta)

        lBeta = -y * betaX + np.log(1 + np.exp(betaX))

        return lBeta.sum()

    def gradient(self, beta):
        X_Aug = np.concatenate((self.X, np.ones((self.X.shape[0], 1))), axis=1)
        beta = beta.reshape(-1, 1)
        y = self.Y.reshape(-1, 1)
        sig = self.sigmoid(np.dot(X_Aug, beta))

        gra = (-X_Aug * (y - sig)).sum(axis=0)
        return gra.reshape(-1, 1)

    def gradDesc(self):
        beta = self.Beta
        change_point = self.epoch / 10
        for i in range(self.epoch):
            grad = self.gradient(beta)
            lr = self.lr[int(i / change_point)]
            beta = beta - lr * grad
        self.Beta = beta
        return beta

    def logistic_regression(self):
        
        if self.method == 'grad':
            return self.gradDesc()
        else:
            raise ValueError('Unknow method', self.method)

    def predict(self):
        X_Aug = np.concatenate((self.X, np.ones((self.X.shape[0], 1))), axis=1)
        sig = self.sigmoid(np.dot(X_Aug, self.Beta))

        sig[sig>0.5] = 1
        sig[sig<=0.5] = 0



class LDA:
    def __init__(self):
        pass



if __name__ == "__main__":
    a = np.random.randn(5, 3)
    b = np.random.randint(2, size=(5,1))
    print(a)
    print(b)
    logistic = Logit(a, b)
    
