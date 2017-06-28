#coding=utf-8
from knn import kNN

#测试创建数据集
group,labels = kNN.createDataSet()
sortedClassCount = kNN.classify0([0,0],group,labels,2)
# print(sortedClassCount)
# print('----------------')
returnMat,classLabelVector = kNN.file2matrix("/Users/mqm/Desktop/ml/MLInAction_sourceCode/Ch02/datingTestSet2.txt")
# print(returnMat)
# print(classLabelVector)
# print('----------------')
normDataSet,ranges,minValues = kNN.autoNorm(returnMat)
# print("normDataSet:",normDataSet)
# print("columnRange:",ranges)
# print("columnMinVal:",minValues)

'''
约会网站分析器 错误率
    前100个作为测试集，后900作为训练集，然后用分类器，产生分类结果
    产生的结果与测试集的标签作对比，计算错误率
'''
hoRatio = 0.1 #定义测试集 数量百分比
norMat,ranges,minValues = kNN.autoNorm(returnMat)#获取归一化数据
m = normDataSet.shape[0]#获取数据集第一维的长度，1000
numTestVecs = int(m*hoRatio)#测试集数量,100
errorCount = 0.0
for i in range(numTestVecs):
    classifierResult = kNN.classify0(norMat[i,:],norMat[numTestVecs:m,:],classLabelVector[numTestVecs:m],3)
    print("分类结果：",classifierResult,";实际结果：",classLabelVector[i])
    if(classifierResult!=classLabelVector[i]): errorCount+=1

print("错误率：",errorCount/float(numTestVecs))
print("------------------")
#kNN.classifyPerson()

#测试将图片的32x32矩阵转为1x1024
#img2Vector = kNN.img2vector("/Users/mqm/Desktop/ml/MLInAction_sourceCode/Ch02/digits/trainingDigits/0_0.txt")

kNN.handwritingClassTest()