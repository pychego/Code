import numpy as np
import pandas as pd
import re
import sklearn
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold, train_test_split
from random import randint
from sklearn.metrics import confusion_matrix, classification_report

training_file = 'training.data'

# 使用 NumPy 的 genfromtxt 函数来加载数据
data = np.genfromtxt(training_file, delimiter=' ')


X = data[:, 0: 6] # 选取前6列作为特征 

# X1 = X[:,1]-X[:,0]  # 补充特征
# X2 = X[:,2]-X[:,1]
# X3 = X[:,4]-X[:,3]
# X4 = X[:,5]-X[:,4]
# X = np.column_stack((X,X1,X2,X3,X4))

# # 将X按列标准化
# X = sklearn.preprocessing.scale(X)
y = data[:, 6]    # 第七列是标签
# 将-1替换成0便于计算,最后再替换回来.
y[y == -1] = 0

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randint(1,100))

# # 创建和训练XGBoost分类器
# model = xgb.XGBClassifier()
# model.fit(X_train, y_train)

# # 使用训练后的模型进行预测
# y_pred = model.predict(X_test)

# # 计算准确率
# accuracy = accuracy_score(y_test, y_pred)
# print("准确率：", accuracy)

# # 创建随机森林分类器
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# # 在训练数据上拟合分类器
# rf_classifier.fit(X_train, y_train)

# # 使用分类器进行预测
# y_pred = rf_classifier.predict(X_test)

# # 计算准确率
# accuracy = accuracy_score(y_test, y_pred)
# print("准确率：", accuracy)

# 创建SVM分类器
svm_classifier = SVC(kernel='rbf', C=1.0)

# 训练SVM模型
svm_classifier.fit(X_train, y_train)

# 预测测试数据
y_pred = svm_classifier.predict(X_test)

# 计算准确度
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确度：{accuracy:.2f}")









