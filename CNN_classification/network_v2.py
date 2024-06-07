import torch

# 构建CNN作为10-分类模型
from torch import nn
from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Linear


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU()) #16*28*28
        self.pool1 = nn.MaxPool2d(2) #16*14*14
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )#32*12*12
        self.layer3 = nn.Sequential(
            nn.Conv2d(32,64,3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )#64*10*10
        self.layer4 = nn.Sequential(
            nn.Conv2d(64,128,3),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )#128*8*8
        self.pool2 = nn.MaxPool2d(2) # 128*4*4
        self.fc = nn.Linear(128*4*4, 10) #全连接层

    def forward(self, x):
        out = self.layer1(x)
        # print(out.shape)
        out = self.pool1(out)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = self.layer4(out)
        out = self.pool2(out)
        # print(out.shape)

        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fc(out)
        return out

if __name__ == "__main__":
    cnn = Net();
    print(cnn(torch.rand(1, 1, 28, 28)))
