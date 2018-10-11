#!/usr/bin/python
# encoding: utf-8


"""
@author : 杜函敏
@contact : duhanmin@foxmail.com
@File : SVMImplementation.py
@time : 2017/7/24 23:14
"""

from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

plt.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体
plt.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题


#在控制台上显示日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

###############################################################################
# 下载数据，如果尚未在磁盘上，并将其作为numpy数组加载

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# shape求图片矩阵的维度大小
# n_samples图数量 h,w图的大小
n_samples, h, w = lfw_people.images.shape

# data中为每个图片矩阵的特征向量（列）
X = lfw_people.data
n_features = X.shape[1]

# 提取不同人的身份标记
y = lfw_people.target

# 提取人的名字
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("*************总数据集大小**********************")
print("**图数量: %d" % n_samples)
print("**特征向量数: %d" % n_features)
print("**人数: %d" % n_classes)
print("*************总数据集大小**********************")


###############################################################################
# 分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
###############################################################################
# 特征提取/降维
n_components = 150
print("从 %d 个维度中提取到 %d 维度" % (X_train.shape[0],n_components))
#主成分分析建模
pca = PCA(n_components=n_components, whiten=True).fit(X_train)
eigenfaces = pca.components_.reshape((n_components, h, w))
print("根据主成分进行降维开始")
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("降维结束")
###############################################################################
# 训练SVM
print("训练SVM分类模型开始")
t0 = time()
#构建归类精确度5x6=30
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
#图片用rbf核函数,权重自动选取
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("SVM训练结束，结果如下：" "SVM训练用时 %0.3fs" % (time() - t0))
print(clf.best_estimator_)

# ###############################################################################
print("测试集SVM分类模型开始")

t0 = time()
y_pred = clf.predict(X_test_pca)
print("测试集用时 %0.3fs" % (time() - t0))

print("误差衡量")
# 数据中1的个数为a，预测1的次数为b，预测1命中的次数为c
# 准确率 precision = c / b
# 召回率 recall = c / a
# f1_score = 2 * precision * recall / (precision + recall)

print(classification_report(y_test, y_pred, target_names=target_names))
print("预测值和实际值对角矩阵")
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


# ###############################################################################
# 画图
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

# 绘制一部分测试集上的预测结果
def title(y_pred, y_test, target_names, i):
    ##以空格为分隔符，把y_pred分成一个list。分割的次数1。[-1]取最后一个
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return '预测值: %s\n 实际值: %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# 画人脸咯，eigenfaces主成分特征脸
eigenface_titles = ["特征脸 %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()