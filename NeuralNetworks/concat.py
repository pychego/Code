import os
import numpy as np

# 指定包含CSV文件的文件夹路径
folder_path = '../MNIST/test/'

import os
import pandas as pd


# 初始化一个空的DataFrame以保存合并后的数据
merged_data = pd.DataFrame()

# 遍历文件夹中的每个CSV文件
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        
        # 读取CSV文件
        data = pd.read_csv(file_path)
        
        # 合并数据，使用列名匹配
        merged_data = pd.concat([merged_data, data], axis=0)

# 将合并后的数据保存为一个新的CSV文件
merged_data.to_csv('../MNIST/test/merged_data.csv', index=False)

# 打印合并后的数据的形状
print(f"合并后的数据形状：{merged_data.shape}")
