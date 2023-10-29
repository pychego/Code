import numpy as np
from pprint import pprint

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
from PIL import Image
import numpy as np

# 指定包含图片的文件夹路径
num = 9
folder_path = '../MNIST/train/{}'.format(num)

# 初始化一个空列表来保存图像数据
image_data = []

# 遍历文件夹中的每张图片
for filename in os.listdir(folder_path):
    image_path = os.path.join(folder_path, filename)
    image = Image.open(image_path)
    # 将image展平为一维数组
    image = image.resize((784, 1))
    image = np.array(image)[0]
    
    # 将图像转换为数组
    image_array = np.array(image)
    
    # 将图像数组添加到image_data列表中
    image_data.append(image_array)
# 转换image_data列表为NumPy数组
image_data_array = np.array(image_data)
# 在image_data_array数组最后加一列label为2
image_data_array = np.c_[image_data_array, np.ones(image_data_array.shape[0])*num]
pprint(f"数组的形状：{image_data_array.shape}")
# 将image_data_array数组保存到文件中
np.savetxt('../MNIST/train/{}.csv'.format(num), image_data_array, fmt='%d', delimiter=',')

# 可以使用image_data_array进行后续处理，如保存到文件或进行图像处理操作

