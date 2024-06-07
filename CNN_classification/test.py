import os
import time

import torch
import torchvision.datasets

from Datesets import FashionMNISTDataset
from network import Net as net
from torch import optim, nn
from torch.utils.data import DataLoader, Dataset
input = "results/outputs/lr0.001_defultnet_200epoch/model_epoch_195.pth" #input of the checkpoint
output = "result/outputs/"
BATCH_SIZE = 128
Test_DATA_PATH = "./datasets/archive/fashion-mnist_test.csv"

def test():
    DEVICE = torch.device("cpu")
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    model = net()
    model.to(DEVICE)
    model.eval()
    correct = 0
    total = 0
    torch.cuda.empty_cache()  # 节约显存
    test_dataset = FashionMNISTDataset(csv_file=Test_DATA_PATH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    #加载checkpoint
    checkpoint = torch.load(input)
    model.load_state_dict(checkpoint["model"])
    for images, labels in test_loader:
        images = images.float().to(DEVICE)
        outputs = model(images).cpu()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        # isBest.is_best(100 * correct / total, epoch=epoch)
    print('准确率: %.4f %%' % (100 * correct / total))
    print("best acc : {:.4f}% ，best epoch : {}".format(checkpoint["isBest"]["best_acc"].float(), checkpoint["isBest"]["best_epoch"]))

# checkpoint = torch.load("results/checkpoint/model_epoch_14.pth")
# print(checkpoint["isBest"])
if __name__ == "__main__":
    # checkpoint = torch.load(input)
    # print(checkpoint["isBest"]["best_acc"].double())
    test()
