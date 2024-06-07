# 数据读取，继承dataset类
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader,ConcatDataset
from torchvision import transforms
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader

Train_DATA_PATH =  "./datasets/archive/fashion-mnist_train.csv"
Test_DATA_PATH = "./datasets/archive/fashion-mnist_test.csv"
class FashionMNISTDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        data = pd.read_csv(csv_file)
        self.X = np.array(data.iloc[:, 1:]).reshape(-1, 1, 28, 28).astype(float)
        self.Y = np.array(data.iloc[:, 0]);
        del data;  # 结束data对数据的引用,节省空间
        self.len = len(self.X)

    def __len__(self):
        # return len(self.X)
        return self.len

    def __getitem__(self, idx):
        item = self.X[idx]
        label = self.Y[idx]
        return (item, label)

if __name__ == "__main__":
    Train_DATA_PATH =  "./datasets/archive/fashion-mnist_train.csv"
    Test_DATA_PATH = "./datasets/archive/fashion-mnist_test.csv"
    train_dataset = FashionMNISTDataset(csv_file=Train_DATA_PATH)
    test_dataset = FashionMNISTDataset(csv_file=Test_DATA_PATH)
    train_loader = DataLoader(dataset= train_dataset, batch_size= 8, shuffle= True)
    # for index,x,y in enumerate(dataloader) :
    #     print(index,x,y)
    a=iter(train_loader)
    data=next(a)
    img=data[0][0].reshape(28,28)
    data[0][0].shape,img.shape
    plt.imshow(img,cmap = plt.cm.gray)
    plt.show()