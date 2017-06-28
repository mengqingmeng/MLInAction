#coding=utf-8
from tree import Tree

dataSet,labels = Tree.createDataSet()
print(dataSet)
print('----------')

shannonEnt = Tree.calcShannonEnt(dataSet)
print(shannonEnt)
print('----------')

#分离第一个特征中，值为1的数据
retDataSet = Tree.splitDataSet(dataSet,0,1)
print(retDataSet)
print('----------')

#测试选择最好的特征值去构建树
bsetFeature = Tree.chooseBestFeatureToSplit(dataSet)
print(bsetFeature)