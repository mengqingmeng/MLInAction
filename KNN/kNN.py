#coding=utf-8
from numpy import *
import operator
from os import listdir

#创建数据集
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

#kNN分类算法
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0] #数据集长度

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
    sqDiffMat = diffMat**2 #对数组中的每个数求平方
    sqDistamces = sqDiffMat.sum(axis=1)#一行内的相加（横坐标^2+纵坐标^2）
    distances = sqDistamces**0.5 #平方根得出距离
    '''
        返回数组比较之后的索引顺序
        [[1, 0],
       [0, 1],
       [0, 1],
       [1, 0]]
    '''
    sortedDistIndicies = distances.argsort()    #从小到大排序，返回排序后的索引值
    classCount = {} #用来保存相似度最接近的项和频率
    for i in range(k):
        voteILabel = labels[sortedDistIndicies[i]]
        classCount[voteILabel] = classCount.get(voteILabel,0)+1

    #对相似度的项按照频率排序，取频率最高的那个
    sortedClassCount = sorted(dict2list(classCount),
                              key = operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]

#归一化数据
def autoNorm(dataSet):
    minValues = dataSet.min(0)#获取每一列的最小值
    maxValues = dataSet.max(0)#获取每一列的最大值
    ranges = maxValues - minValues#相减得到范围
    normDataSet = zeros(shape(dataSet))#创建归一化之后的结果集,如：1000x3
    m = dataSet.shape[0]#如：1000
    normDataSet = dataSet - tile(minValues,(m,1))#数据集的每一项减去相应列的最小值
    normDataSet = normDataSet/tile(ranges,(m,1))#上步得到的结果每一项除以相应列的范围（最大值-最小值）
    return normDataSet,ranges,minValues #返回，归一化结果集，每一列的范围，每一列的最小值

#从文件读取约会数据
def file2matrix(filename):
    fr = open(filename)#打开文件
    arrayOfLines = fr.readlines()#读取每行信息，放在list中
    numberOfLines = len(arrayOfLines)#获取信息的总长度，如：1000
    returnMat = zeros((numberOfLines,3))#创建1000x3的数组
    classLabelVector = []#存放标签的list
    index = 0
    for line in arrayOfLines:#遍历每行信息
        line = line.strip()#去除空格
        listFromLine = line.split('\t')#将每行信息，以\t分割，存在list中
        returnMat[index,:]= listFromLine[0:3]#将数据结果存放在returnMat中
        classLabelVector.append(int(listFromLine[-1]))#将标签结果放在classLabelVector
        index +=1
    return returnMat,classLabelVector

def dict2list(dic:dict):
    ''' 将字典转化为列表 '''
    keys = dic.keys()
    vals = dic.values()
    lst = [(key, val) for key, val in zip(keys, vals)]
    return lst

def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(raw_input('玩游戏所花费的时间百分比？'))
    ffMiles = float(raw_input('每年获得的飞行常客里程数？'))
    iceCream = float(raw_input('每周消费的冰淇淋公升数？'))
    returnMat, classLabelVector = file2matrix("/Users/mqm/Desktop/ml/MLInAction_sourceCode/Ch02/datingTestSet2.txt")
    normMat,ranges,minVals = autoNorm(returnMat)
    inArr = array([percentTats,ffMiles,iceCream])
    classifierResult = classify0(inArr-minVals/ranges,normMat,classLabelVector,3)
    print('结果：',resultList[classifierResult-1])

def raw_input(question):
    result = 0.0
    print(question)
    result = input()
    return result

#将32x32的矩阵转为1x1024
def img2vector(fileName):
    returnVect = zeros((1,1024))
    fr = open(fileName)
    for i in range(32):
        linStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(linStr[j])
    return returnVect

#手写输入识别
def handwritingClassTest():
    #训练集数据
    hwLables = [] #训练集，存放正确数值
    trainingFileList = listdir("/Users/mqm/Desktop/ml/MLInAction_sourceCode/Ch02/digits/trainingDigits")
    fileCount = len(trainingFileList)
    trainingMat = zeros((fileCount,1024)) #测试集数据
    for i in range(fileCount):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]#文件名称，去掉后缀
        classNumStr = int(fileStr.split('_')[0]) #文件标注的数值
        hwLables.append(classNumStr)
        trainingMat[i,:] = img2vector("/Users/mqm/Desktop/ml/MLInAction_sourceCode/Ch02/digits/trainingDigits/"+fileNameStr)

    #用测试集 测试错误率
    testFileList = listdir("/Users/mqm/Desktop/ml/MLInAction_sourceCode/Ch02/digits/testDigits")
    errorCount = 0.0
    testFileCount = len(testFileList)
    for i in range(testFileCount):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector("/Users/mqm/Desktop/ml/MLInAction_sourceCode/Ch02/digits/testDigits/"+fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLables,3)
        print("分类结果："+str(classifierResult)+"；正确结果："+str(classNumStr))
        if(classifierResult!=classNumStr): errorCount+=1

    print("total error:"+str(errorCount))
    print("total error rate:"+str(errorCount/float(testFileCount)))