import numpy as np
import pandas as pd
import os

scriptpath = os.path.abspath(__file__)
rootpath = os.path.dirname(os.path.dirname(scriptpath))

datapath = os.path.join(rootpath, 'data/LinearModel/watermelon3_0_Ch.csv')


def testLogistic():
    data = pd.read_csv(datapath)
    print(data.values)


if __name__ == "__main__":
    testLogistic()