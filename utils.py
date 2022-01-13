import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
# from opacus import PrivacyEngine  # pip install opacus
from data import AlzheimersData
from model import ContrastiveLoss
from torch.autograd import Variable


def loadData():
    """
    Load Alzheimer's train and test set
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    trainset = ImageFolder(
        "C:/Datasets/Alzheimer_s Dataset/train", transform=transform)
    testset = ImageFolder(
        "C:/Datasets/Alzheimer_s Dataset/test", transform=transform)
    trainData = AlzheimersData(trainset, transform=transform)
    testData = AlzheimersData(testset, transform=transform)
    trainloader = DataLoader(trainData, batch_size=50, shuffle=True)
    testloader = DataLoader(testData, batch_size=50, shuffle=True)
    return trainloader, testloader


def train(net, trainLoader, epochs, device):
    """
    Train the network on the training set.
    """
    criterion = ContrastiveLoss()

    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # privacy_engine = PrivacyEngine(net,
    #                                # batchS-size/length of dataset
    #                                sample_rate=50/len(trainLoader.dataset),
    #                                # length of data
    #                                sample_size=len(trainLoader.dataset),
    #                                noise_multiplier=1.5,  # amount of noise
    #                                max_grad_norm=2.0)
    # privacy_engine.attach(optimizer)
    for epoch in range(0, epochs):
        e_loss = 0
        count = 0
        v_loss = 0
        for i, data in enumerate(trainLoader):
            # print(data)
            count += 1
            img0, img1, label = data
            img0, img1, label = img0.to(
                device), img1.to(device), label.to(device)
            output1, output2 = net.forward(img0, img1)
            optimizer.zero_grad()
            loss_contrastive = criterion(output1, output2, label)
            e_loss += loss_contrastive.item()
            loss_contrastive.backward()
            optimizer.step()
        print("Epoch Loss:", e_loss/count)


def test(net, testloader, device):
    """Validate the network on the entire test set."""
    criterion = ContrastiveLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for i, data in enumerate(testloader):
            img0, img1, label = data
            img0, img1, label = img0.to(device), img1.to(
                device), label.to(device)
            output1, output2 = net.forward(img0, img1)
            loss += criterion(output1, output2, label)
    print("Validation Loss")
    return loss
