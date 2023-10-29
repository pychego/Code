import numpy as np
import pandas as pd
# import kmeans form sklearn
from sklearn.cluster import KMeans

"""添加新特征, 并保存文件train.npy, test.npy"""

def kmeans(X, k):
    # kmeans
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
    # get labels
    labels = kmeans.labels_
    # get centroids
    centroids = kmeans.cluster_centers_
    return labels, centroids
    
def add_difference(X):
    """添加速度的差值特征
®
    Args:
        df (_type_): _description_
    """
    label = X[:,6]
    X = X[:, :-1]
    X1 = X[:,1]-X[:,0]  # 补充特征
    X2 = X[:,2]-X[:,1]
    X3 = X[:,4]-X[:,3]
    X4 = X[:,5]-X[:,4]
    X = np.column_stack((X,X1,X2,X3,X4, label))
    return X  # 需返回X
    

data  = np.genfromtxt('training.data', delimiter=' ')
data = add_difference(data)
# 将numpy数组保存为csv文件, 保留4位小数
pd.DataFrame(data).to_csv('add_feature_training.csv', index=False, header=False, float_format='%.4f')