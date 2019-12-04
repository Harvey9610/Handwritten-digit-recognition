#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
import time
from model import Model

batch_size = 32
epochs = 20
lr = 1e-2

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])

train_datasets = MNIST(
    root='./data/',
    train=True,
    transform=transform,
    download=True)

test_datasets = MNIST(
    root='./data/',
    train=False,
    transform=transform,
    download=True)

train_loader = torch.utils.data.DataLoader(
    train_datasets,
    batch_size=batch_size,
    shuffle=True,
    **kwargs)

test_loader = torch.utils.data.DataLoader(
    test_datasets,
    batch_size=batch_size,
    shuffle=True,
    **kwargs)

def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    for _, (images, labels) in enumerate(train_loader, start=1):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        _, prediction = torch.max(outputs, 1)
        correct += (predicts == labels).sum()

        loss = criterion(outputss, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    accuracy = correct / len(train_datasets)
    return total_loss, accuracy

def test(epoch, model, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for _, (images, labels) in enumerate(test_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(data)
            _, prediction = torch.max(outputs, 1)
            correct += (prediction == labels).sum()

            loss = criterion(outputs, labels)
            total_loss += loss

    accuracy = correct / len(test_datasets)
    return total_loss, accuracy

def main():
    model = Model()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    criterion = criterion.to(device)

    train_loss = 0.
    train_acc = 0.
    test_loss = 0.
    test_acc = 0.

    for epoch in range(1, epochs+1):
        train_loss, train_acc = train(epoch, model, train_loader, criterion, optimizer)
        test_loss, test_acc = test(epoch, model, test_loader, criterion)
        print('epoch: [%d/%d]\ttrain_loss: %.4f\ttrain_acc: %.4f\ttest_loss: %.4f\t test_acc: %.4f' % (epoch, epochs, train_loss, train_acc, test_loss, test_acc))

    model.save('mnist.pkt')

if __name__ == '__main__':
    main()
