import numpy as np
import pandas as pd
import os

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
    
    print(X)
    print(y)
    


if __name__ == "__main__":
    testLogistic()