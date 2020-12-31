import numpy as np


class NormData:
    def __init__(self, dataset):
        self.dataSet = dataset

    def linearNorm(self):
        minVal = self.dataSet.min(0)
        maxVal = self.dataSet.max(0)
        ranges = maxVal - minVal
        normData = np.zeros(self.dataSet.shape)
        m = self.dataSet.shape[0]
        normData = self.dataSet - np.tile(minVal, (m, 1))
        normData = normData / np.tile(ranges, (m, 1))
        return normData, ranges, minVal

    