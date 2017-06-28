#coding=utf-8
from math import log
import operator
def createDataSet():
    dataSet = [[1,1,'yes']
               ,[1,1,'yes']
               ,[1,0,'no']
               ,[0,1,'no']
               ,[0,1,'no']]

    labels = ['离开水面是否能生存','是否有蹼脚']
    return dataSet,labels

#计算香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelsCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelsCounts.keys():
            labelsCounts[currentLabel] = 0

        labelsCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelsCounts:
        prob = float(labelsCounts[key])/numEntries
        shannonEnt -= prob* log(prob,2)
    return shannonEnt
'''
分割数据
axis：第几个特征；value:特征值
方法将根据从第axis个特征中筛选出值为value的数据,数据结果中不包含本特征
'''
def splitDataSet(dataSet,axis,value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


'''
选择数据集中 最好的特征去分离
'''
def chooseBestFeatureToSplit(dataSet):
    numberFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)#原始香农熵
    bestInfoGain = 0.0 #最好的信息增益
    bestFeature=-1 #最好的特征的索引值

    #计算每个特征值的香农熵
    for i in range(numberFeatures):
        featList = [example[i] for example in dataSet] #每个特征值数组
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        #判断该特征值的信息增益
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature