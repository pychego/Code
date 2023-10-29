from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from Class import *
from nn_train import nn_train
from nn_forward import nn_forward
from nn_test import nn_test
from nn_predict import nn_predict
from nn_backpropagation import nn_backpropagation
from nn_applygradient import nn_applygradient
from function import sigmoid, softmax
from tqdm import tqdm
from random import randint
from sklearn.preprocessing import OneHotEncoder
from pprint import pprint

import numpy as np
import pandas as pd


def nn_testChess():
    # with open('krkopt.data') as my_data:  # 读取兵王问题数据集
    #     lines = my_data.readlines()
    #     data = np.zeros((28056, 6), dtype=float)
    #     label = np.zeros((28056, 2), dtype=float)
    #     i = 0
    #     for line in lines:
    #         line = line.split(',')  # 以逗号分开,构成列表
    #         if i == 0:
    #             line[0] = 'a'  # 不知道为什么第一个数据乱码，用写字板打开是'a'

    #         line[0] = ord(line[0]) - 96  # 将字母转换成数字, a的ASCII码是97
    #         line[1] = float(line[1]) - 48 # 字符1的ASCII码是49
    #         line[2] = ord(line[2]) - 96
    #         line[3] = float(line[3]) - 48
    #         line[4] = ord(line[4]) - 96
    #         line[5] = float(line[5]) - 48
    #         data[i, :] = line[:-1] # 除去最后一个元素，即标签

    #         if line[6][0] == 'd':  # line[6]是字符串,可能是'draw', line[6][0]是'd'
    #             label[i] = np.array([1, 0])  # 将标签转换成one-hot编码
    #         else:
    #             label[i] = np.array([0, 1])
    #         i += 1
    #         if i == 28056:
    #             break
    train_data = np.loadtxt('../MNIST/train/train.csv', delimiter=',')
    test_data = np.loadtxt('../MNIST/test/test.csv', delimiter=',')
    
    
    # 选择小样本训练和测试
    num_train = train_data.shape[0]
    selected_rows = np.random.choice(num_train, size=int(0.1 * num_train), replace=False)
    train_data = train_data[selected_rows, :]
    
    num_test  = test_data.shape[0]
    selected_rows = np.random.choice(num_test, size=int(0.1 * num_test), replace=False)
    test_data = test_data[selected_rows, :]
    
    
    # train_data = np.random.choice(train_data, size=int(0.2 * len(train_data)), replace=False)
    # test_data = np.random.choice(test_data, size=int(0.2 * len(test_data)), replace=False)
    
    train_x, train_y = train_data[:, :-1], train_data[:, -1]
    test_x, test_y = test_data[:, :-1], test_data[:, -1]   
    
    encoder = OneHotEncoder(sparse=False)

    # 使用OneHotEncoder进行独热编码
    train_y = encoder.fit_transform(train_y.reshape(-1, 1))
    test_y = encoder.fit_transform(test_y.reshape(-1, 1))
    
    randam_state = randint(0, 100)

    # ratioTraining = 0.4
    # ratioValidation = 0.1
    # ratioTesting = 0.5   # 训练集:验证集:测试集 = 4:1:5
    # randam_state = randomint(0, 100)
    # xTraining, xTesting, yTraining, yTesting = train_test_split(train_x, train_y, test_size=0.2,
    #                                                             random_state=randam_state)  # 随机分配数据集
    # xTesting, xValidation, yTesting, yValidation = train_test_split(test_x, test_y,
    #                                                                 test_size=ratioValidation / ratioTesting,
    #                                                                 random_state=randam_state)
    
    xTraining, xTesting, yTraining, yTesting = train_test_split(train_x, train_y, test_size=0.2,
                                                                random_state=randam_state)  # 随机分配数据集
    xValidation, yValidation = test_x, test_y
    pprint(yTraining)
    pprint(yTesting)
    # 拆分成测试集和验证集
    scaler = StandardScaler(copy=False)
    scaler.fit(xTraining)
    scaler.transform(xTraining)  # 标准归一化
    scaler.transform(xTesting)
    scaler.transform(xValidation)

    # 定义模型
    nn = NN(layer=[784, 400, 169, 49, 10], active_function='sigmoid', learning_rate=0.1, batch_normalization=1,
            optimization_method='Adam',
            objective_function='Cross Entropy')

    option = Option()      # Option仅控制batch_size和iteration
    option.batch_size = 128 # 一次送进去的样本数
    option.iteration = 1 # 控制小的迭代次数?

    iteration = 0
    maxAccuracy = 0
    totalAccuracy = []
    totalCost = []
    maxIteration = 50
    for iteration in tqdm(range(maxIteration)):
        iteration = iteration + 1
        nn = nn_train(nn, option, xTraining, yTraining)
        totalCost.append(sum(nn.cost.values()) / len(nn.cost.values()))
        # plot(totalCost)
        (wrongs, accuracy) = nn_test(nn, xValidation, yValidation)
        totalAccuracy.append(accuracy)
        if accuracy > maxAccuracy:
            maxAccuracy = accuracy
            storedNN = nn

        cost = totalCost[iteration - 1]
        print('Accuracy:',accuracy)
        print('Cost:',totalCost[iteration - 1])

    fig, (ax1,ax2) = plt.subplots(nrows = 2, figsize=(5, 2.7), layout='constrained')
    ax1.plot(np.arange(len(totalCost)),totalCost, color='red')
    ax1.set_title('Average Objective Function Value on the Training Set')
    #plt.show()
    #plt.savefig('totalCost.png')

    #fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
    ax2.plot(np.arange(len(totalAccuracy)), totalAccuracy, color='red')
    ax2.set_title('Accuracy on the Validation Set')
    plt.show()
    plt.savefig('totalCostAccuracy.png')


    wrongs, accuracy = nn_test(storedNN, xTesting, yTesting)
    print('Accuracy on Testset:', accuracy)

nn_testChess()
