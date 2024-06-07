import torchvision
import torchvision.transforms as transforms#处理数据的模块

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

def collate_fn(batch):
    """
    batch 是一个列表，包含了每个样本的信息，每个元素是一个 tuple (image, label)，
    其中 image 是 PIL.Image.Image 类型，label 是对应的标签。
    """
    images = []
    labels = []
    for image, label in batch:
        image = np.array(image)  # 将 PIL.Image.Image 转换成 numpy 数组
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)

minist=torchvision.datasets.FashionMNIST(root="./datasets",download=True, train=True)#如果用于训练则会导入用于训练的大样本数据,transform=transforms.ToTensor())
dataset = torchvision.datasets.FashionMNIST(root="./datasets/", download=False, train=True)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
for index,(x,y) in enumerate(train_loader):
    # print("{}||{}||{}".format(index,x,y))
    # print("index : {}".format(index))
    # print("+++++++++++++++++++++++++++")
    # print("y is : {}".format(y))
    print()