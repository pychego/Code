import numpy as np
np.seterr(divide='ignore', invalid='ignore')

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
# def sigmoid(x):
#     """返回对应的概率值"""
#     return 1.0 / (1.0 + np.exp(-x))
def sigmoid(x):
    y = x.copy()      # 对sigmoid函数优化，避免出现极大的数据溢出
    y[x >= 0] = 1.0 / (1 + np.exp(-x[x >= 0]))
    y[x < 0] = np.exp(x[x < 0]) / (1 + np.exp(x[x < 0]))
    return y


def softmax(x):
    """返回对应的概率值"""
    # exp_x = np.exp(x)
    # softmax_x = np.zeros(x.shape,dtype=float)
    # for i in range(len(x[0])):
    #     softmax_x[:,i] = exp_x[:,i] / (exp_x[0,i] + exp_x[1,i])
    x = x - x.max(axis=0)
    y = np.exp(x)
    y /= y.sum(axis=0, keepdims=True)
    return y 