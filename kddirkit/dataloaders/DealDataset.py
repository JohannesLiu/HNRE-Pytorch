import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import numpy as np


class DealDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self):
        # xy = np.loadtxt('./dataset/diabetes.csv.gz', delimiter=',', dtype=np.float32)  # 使用numpy读取数据
        # self.x_data = torch.from_numpy(xy[:, 0:-1])
        # self.y_data = torch.from_numpy(xy[:, [-1]])
        # self.len = xy.shape[0]
        NotImplemented

    def __getitem__(self, index):
        # return self.x_data[index], self.y_data[index]
        NotImplemented

    def __len__(self):
        # return self.len
        NotImplemented