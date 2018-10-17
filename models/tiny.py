# a tiny CNN for testing 
# simple training got 94% train and 89% validation on CIFAR-10, validation loss 0.33
import torch
import torch.nn as nn
import torch.nn.functional as F


class AllConv(nn.Module):
    """Not exactly the all convolutional network, but based on it."""
    def __init__(self, num_classes=10, n_filters=96//4, Conv2d=nn.Conv2d):
        self.num_classes = 10
        super(AllConv, self).__init__()
        self.features = nn.Sequential(
            Conv2d(3, n_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
            Conv2d(n_filters, n_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),
            Conv2d(n_filters, n_filters, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace=True),

            Conv2d(n_filters, n_filters*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_filters*2),
            nn.ReLU(inplace=True),
            Conv2d(n_filters*2, n_filters*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_filters*2),
            nn.ReLU(inplace=True),
            Conv2d(n_filters*2, n_filters*2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(n_filters*2),
            nn.ReLU(inplace=True),

            Conv2d(n_filters*2, n_filters*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_filters*2),
            nn.ReLU(inplace=True),
            Conv2d(n_filters*2, n_filters*2, kernel_size=1, stride=1),
            nn.BatchNorm2d(n_filters*2),
            nn.ReLU(inplace=True),
            Conv2d(n_filters*2, n_filters*2, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(n_filters*2, 10)

    def forward(self, x):
        x = self.features(x)
        n, c, _, _ = x.size()
        x = self.pool(x).view(n, c)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    net = AllConv()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

