'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
from prototypical import PrototypicalLoss

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.2, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--eval', '-e', action='store_true', help='run only on test')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
import ipdb
ipdb.set_trace()
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = ACDCResNet18()
net = AllConvACDC()
# net = OldAllConvACDC()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=8.8e-6)
optimizer = optim.Adam(net.parameters(), lr=0.002)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    criterion.train()
    train_loss = 0
    accuracy = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm(net.parameters(), 5.)
        optimizer.step()

        train_loss += loss.item()
        accuracy += criterion.acc
        total += 1.

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%%'
            % (train_loss/(batch_idx+1), accuracy/total ))

def test(epoch):
    global best_acc
    net.eval()
    criterion.eval()
    test_loss = 0
    accuracy = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            accuracy += criterion.acc
            total += 1.

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%%'
                % (test_loss/(batch_idx+1), accuracy/total))

    # Save checkpoint.
    acc = accuracy/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc

def sgdr_step(scheduler):
    scheduler.step()
    if scheduler.last_epoch == scheduler.T_max:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 2*scheduler.T_max)
        scheduler.step()
    return scheduler

if start_epoch > 0:
    for i in range(start_epoch):
        scheduler = sgdr_step(scheduler)

for epoch in range(start_epoch, start_epoch+500):
    scheduler = sgdr_step(scheduler)
    if not args.eval:
        train(epoch)
    else:
        print("Loaded at epoch:", epoch)
    test(epoch)
    if args.eval:
        break
