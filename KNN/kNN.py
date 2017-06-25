#coding=utf-8
from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0] #sh数据集长度，4

    '''
        计算inX与每个数据做点的差值
        如inX = [0,0]
        tile(inX,(4,1)),行变4倍，列1倍
            产生数组4x2：[[0,0],
                    [0,0],
                    [0,0],
                    [0,0]]
        结果：
        [[-1.0,-1.1],[-1.0,-1.0],[0,0],[0,-0.1]]
    '''
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2 #求平方
    sqDistamces = sqDiffMat.sum(axis=1)#一行内的相加（横坐标^2+纵坐标^2）
    distances = sqDistamces**0.5 #平方根得出距离，1x4数组
    '''
        返回数组比较之后的索引顺序
        [[1, 0],
       [0, 1],
       [0, 1],
       [1, 0]]
    '''
    sortedDistIndicies = distances.argsort()
    classCount = {} #用来保存相似度最接近的项和频率
    for i in range(k):
        voteILabel = labels[sortedDistIndicies[i]]
        classCount[voteILabel] = classCount.get(voteILabel,0)+1

    #对相似度的项按照频率排序，取频率最高的那个
    sortedClassCount = sorted(classCount.iteritems(),
                              key = operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]

group,labels = createDataSet()
sortedClassCount = classify0([0,0],group,labels,2)
print  sortedClassCount