import os
import torch
import sklearn
import numpy as np
from torch.utils.data import DataLoader

from trainf import test, evaluate
from dataset import DealDataset, TestDataset
from model import TimeSequence, FC

device = torch.device("cpu")
feature_num = 6

# 读入文件
data = np.genfromtxt('testing.data', delimiter=' ')
# feature标准化
data[:, 0: feature_num] = sklearn.preprocessing.scale(data[:, 0: feature_num])

test_dataset = TestDataset(data)
test_loader = DataLoader(
        test_dataset, batch_size=256, shuffle=True, num_workers=0,
    )

model = FC(6, 2)
ckpt = torch.load(os.path.join("ckpt", "fc-{:04d}.ckpt".format(40)), map_location="cpu")
model.load_state_dict(ckpt)
model.eval()

y_pred = None
input_map = lambda x: x.float().to(device)
for row in test_loader:
    row = row.float().to(device)
    # 将row转换为tensor
    # row = torch.from_numpy(row)
    # label = label.reshape(-1).long()
    output = model(row)
    if y_pred is None: 
        y_pred = output.cpu().argmax(dim=1).detach().numpy()
    else:
        y_pred = np.concatenate([y_pred, output.cpu().argmax(dim=1).detach().numpy()])

print(y_pred.T)
# 将y_pred中的0转换为-1
y_pred[y_pred == 0] = -1
# save y_pred as txt
np.savetxt('y_pred.txt', y_pred.T, fmt='%d')