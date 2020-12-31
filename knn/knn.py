import numpy as np


class KNN:
    def __init__(self, k, dataset, label):
        self.topK = k
        self.dataSet = dataset
        self.label = label

    def knn(self, inX):
        dataSize = self.dataSet.shape[0]
        diffMat = np.tile(inX, (dataSize, 1)) - self.dataSet
        sqDiffMat = diffMat ** 2
        sqDistance = sqDiffMat.sum(axis=1)
        distance = sqDistance ** 0.5

        sortedDist = distance.argsort()
        # print(sortedDist)
        classCount = {}
        for i in range(self.topK):
            voteLabel = self.label[sortedDist[i]]
            classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
        sortedClassCount = sorted(classCount.items(), key=lambda x : x[1], reverse=True)
        return sortedClassCount[0][0]

