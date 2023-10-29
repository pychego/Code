import numpy as np
from nn_predict import nn_predict

def nn_test(nn,test_x,test_y):
    nn = nn_predict(nn,test_x)
    y_output = nn.a[nn.depth-1]
    y_output = y_output.T
    label = np.argmax(y_output, axis=1) #按行找出最大元素所在下标
    expectation = np.argmax(test_y, axis=1)  # 此时test_y是one-hot编码，按行找出最大元素所在下标
    # expectation = test_y   # 此时test_y不是one-hot编码，直接赋值
    wrongs = sum(label != expectation) #求预测与期望不相等的个数
    success_ratio = 1-wrongs/test_y.shape[0]
    return wrongs, success_ratio