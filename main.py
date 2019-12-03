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

transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))

train_datasets = MNIST(root='./data/',
                      train=True,
                      transform=transform,
                      download=True)

test_datasets = MNIST(root='./data/',
                     train=False,
                     transform=transform,
                     download=True)

train_loader = torch.utils.data.DataLoader(train_datasets,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           **kwargs)

test_loader = torch.utils.data.DataLoader(test_datasets,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          **kwargs)

def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    t0 = time.time()
    for _, (images, labels) in enumerate(train_loader, start=1):
        print('Epochs: [%d/%d]' % (epoch, epochs))
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
    t1 = time.time()
    print('epoch: [%d/%d]\ntime: %d\ttrain_loss: %.4f\taccuracy: %.4f' % (epoch, epochs, (t1-t0), total_loss.item(), accuracy))

def test(epoch, model, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    t0 = time.time()
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
    t1 = time.time()
    print('epoch: [%d/%d]\ntime: %d\tloss: %.4f\taccuracy: %.4f' % (epoch, epochs, (t1-t0), total_loss.item(), accuracy))

def main():
    model = Model()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs):
        train(epoch, model, train_loader, criterion, optimizer)
        test(epoch, model, test_loader, criterion)

    model.save('mnist.pkt')

if __name__ = '__main__':
    main()
