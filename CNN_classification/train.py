import os
import time

import torch
import torchvision.datasets

from Datesets import FashionMNISTDataset
from network import Net as net
# from network_v2 import Net as net
from torch import optim, nn
from torch.utils.data import DataLoader

# 定义超参
BATCH_SIZE = 512  # 批量训练大小
epoch_num = 200
learn_rate = 0.01

path = "./results/"
def main():
    # dataset = torchvision.datasets.FashionMNIST(root="./datasets/datasets", download=False, train=True)

    isBest = Best()#新建一个best对象
    if not(os.path.exists(path+'checkpoint/')):
        os.makedirs(path+'checkpoint/',exist_ok=True)
    if not (os.path.exists(path + 'logs/')):
        os.makedirs(path + 'logs/', exist_ok=True)
    model = net()  # 创建模型
    DEVICE = torch.device("cpu")
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    # print(DEVICE)
    model.to(DEVICE)

    # 数据加载器
    Train_DATA_PATH = "./datasets/archive/fashion-mnist_train.csv"
    Test_DATA_PATH = "./datasets/archive/fashion-mnist_test.csv"
    train_dataset = FashionMNISTDataset(csv_file=Train_DATA_PATH)
    test_dataset = FashionMNISTDataset(csv_file=Test_DATA_PATH)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    #定义损失函数
    loss_func = nn.CrossEntropyLoss() #loss函数

    #优化器
    # optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)


    start_time = time.time()
    record = open("results/logs/train_record.txt", 'w+')
    loss_record = open("results/logs/loss_record.txt","w+")
    for epoch in range(epoch_num):
        # print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))
        train(epoch= epoch, train_loader = train_loader, DEVICE= DEVICE, optimizer=optimizer,train_dataset=train_dataset,model=model,loss_func=loss_func, loss_record = loss_record)
        end_time = time.time()
        print("epoch time is {:.2f}".format(end_time - start_time))
        val(Net=model, test_loader= test_loader,epoch=epoch,DEVICE = DEVICE, isBest=isBest, record= record)
        save_checkpoint(epoch,model=model, optimizer= optimizer, isBest=isBest)
    record.close()
    loss_record.close()
def train(epoch,train_loader,DEVICE,optimizer,train_dataset,model,loss_func,loss_record):
    # 记录损失函数
    model.train()
    losses = []

    for i, (images, labels) in enumerate(train_loader):
        images = images.float().to(DEVICE)
        labels = labels.to(DEVICE)
        # 清零
        optimizer.zero_grad()
        outputs = model(images)
        # 计算损失函数
        loss = loss_func(outputs, labels).cpu()
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().data.item())
        if (i + 1) % 100 == 0:
            print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' % (
            epoch + 1, epoch_num, i + 1, len(train_dataset) // BATCH_SIZE, loss.data.item()))
            loss_record.write('Epoch : %d,   Loss: %.4f \n'  % (
            epoch + 1,   loss.data.item()))


def val(Net,test_loader,epoch,DEVICE,isBest, record):
    Net.eval()
    correct = 0
    total = 0
    torch.cuda.empty_cache() #节约显存
    for images, labels in test_loader:
        images = images.float().to(DEVICE)
        outputs = Net(images).cpu()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    acc = 100 * correct / total
    isBest.is_best(acc, epoch= epoch+1)
    print('epoch : %d 准确率: %.4f %%' % (epoch+1,acc))
    print("best acc : {:.4f}% ，best epoch : {}".format(isBest.best_acc,isBest.best_epoch))

    record.write('epoch : %d , 准确率: %.4f %% ,' % (epoch+1,acc))
    record.write("best acc : {:.4f}% ,best epoch : {} \n".format(isBest.best_acc,isBest.best_epoch))




def save_checkpoint(epoch, model, optimizer, isBest):
    # if not os.input.exists("checkpoint/"):
    #     os.makedirs("checkpoint/",exist_ok=True)
    model_out_path = path + "checkpoint/" + "model_epoch_{}.pth".format(epoch+1)
    state = {"epoch": epoch+1, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "isBest":isBest.state_dict()}

    # if not os.input.exists("checkpoint/loss/"):
    #     os.makedirs("checkpoint/loss/")
    torch.save(state, model_out_path)

class Best:
    def __init__(self):
        self.best_acc = 0
        self.best_epoch = -1

    def is_best(self, acc, epoch):
        if acc >= self.best_acc:
            self.best_acc = acc
            self.best_epoch = epoch

    def reset(self):
        self.best_acc = 0
        self.best_epoch = -1

    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items()}
        return state_dict

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

if __name__ == "__main__":
    main()