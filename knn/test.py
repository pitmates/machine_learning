import numpy as np
import os
import sys

scriptpath = os.path.abspath(__file__)
rootpath = os.path.dirname(os.path.dirname(scriptpath))

sys.path.append(rootpath)
from norm import NormData

from knn import KNN


class DatingTest:
    def __init__(self):
        datapath = "data/knn/datingTestSet2.txt"
        self.filepath = os.path.join(rootpath, datapath)

    def file2matrix(self):
        with open(self.filepath, "r") as fr:
            lines = fr.readlines()

        lens = len(lines)
        returnMat = np.zeros((lens, 3))
        labels = []

        for id, line in enumerate(lines):
            line = line.strip()
            lineList = line.split("\t")

            returnMat[id, :] = lineList[:3]
            labels.append(lineList[-1])

        return returnMat, labels
        

    def drawPlot(self):
        import matplotlib.pyplot as plt 

        mats, labels = self.file2matrix()        
        normMethod = NormData(mats)
        M, _, _ = normMethod.linearNorm()
        print(M )
        colors = ['r'] * len(labels)
        for i, la in enumerate(labels):
            if la == '2': colors[i] = 'g'
            if la == '3': colors[i] = 'b'
        # print(colors)
        # print(mats)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(mats[:, 0], mats[:, 1], s=10, c=colors)
        plt.show()

    def testClassify(self):
        testRatio = 0.1
        mats, labels = self.file2matrix()        
        normMethod = NormData(mats)
        dataset, _, _ = normMethod.linearNorm()

        m = dataset.shape[0]
        testVec = int(m * testRatio)
        classifier = KNN(10, dataset, labels)
        errorCount = 0
        for i in range(testVec):
            classifyRes = classifier.knn(dataset[i])
            # print("the classifier came back with: {}, the real answer is: {}, is correct: {}".format(classifyRes, labels[i], classifyRes == labels[i]))
            if (classifyRes != labels[i]):
                errorCount += 1
                print("the classifier came back with: {}, the real answer is: {}, the error id is: {}".format(classifyRes, labels[i], i))
        print("the total error count is: {}, the error rate is: {}".format(errorCount, errorCount / testVec))


class HandWrittingDigitalTest:
    def __init__(self):
        trainingDir = 'data/knn/trainingDigits'
        testingDir = 'data/knn/testDigits'
        self.trainingDir = os.path.join(rootpath, trainingDir)
        self.testingDir = os.path.join(rootpath, testingDir)

    def img2vector(self, filepath):
        with open(filepath, 'r') as fr:
            lines = fr.readlines()
        numVec = np.zeros((1, 1024))
        
        for i, line in enumerate(lines):
            line = line.strip()
            numVec[0, i*32 : (i+1)*32] = [int(i) for i in line]

        return numVec

    def testClassify(self):
        hwLabels = []
        trainingList = os.listdir(self.trainingDir)

        trainingMat = np.zeros((len(trainingList), 1024))

        for i, name in enumerate(trainingList):
            num = name.split('_')[0]
            hwLabels.append(num)
            trainingMat[i] = self.img2vector(os.path.join(self.trainingDir, name))

        testList = os.listdir(self.testingDir)
        errorCount = 0

        for name in testList:
            num = name.split('_')[0]
            testVec = self.img2vector(os.path.join(self.testingDir, name))
            classifier = KNN(3, trainingMat, hwLabels)
            classifyRes = classifier.knn(testVec)
            # print("the classifier came back with: {}, the real answer is: {}, is correct: {}".format(classifyRes, num, classifyRes == num))
            if (classifyRes != num):
                errorCount += 1
                print("the classifier came back with: {}, the real answer is: {}, the error file name is: {}".format(classifyRes, num, name))
        print("the total error count is: {}, the error rate is: {}".format(errorCount, errorCount / len(testList)))


def testDating():
    print('begin to test dating ...')
    datingTest = DatingTest()
    datingTest.testClassify()
    print('dating testing end ---------\n')


def testHW():
    print('begin to test hand writting digital ...')
    hwdt = HandWrittingDigitalTest()
    hwdt.testClassify()
    print('hand writting digital testing end ---------\n')


if __name__ == "__main__":
    testDating()
    testHW()
