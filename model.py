from torch.autograd.function import InplaceFunction
import torch.nn as nn
import torch as torch


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=3):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 2nd pass in block
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # base case doesnt do anything, assuming layers match
        self.shortcut = nn.Sequential()
        self.use_shortcut = stride != 1 or in_channels != out_channels
        # if stride isnt 1, the data is downsampled, and if in_channels != out_channels
        # then you need to account for that jump/shortcut by performing
        # batch normalization
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    # forward pass, inputs: x is the input data
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # you want to make sure input and outputs match. if they do
        # then you can just add input to the output and return that
        # otherwise, you need to batch normalize as defined per the
        # ResidualBlock class above
        shortcut = self.shortcut if self.use_shortcut else x
        out_add = out + shortcut
        out = torch.relu(out_add)
        # returns the final allowed shortcut output to pass to the next block/layer
        return out


class AudioCNN(nn.Module):
    # num classes is the amount of possible predictions
    def __init__(self, num_classes=50):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = nn.ModuleList([ResidualBlock(64, 64) for i in range(3)])
        self.layer2 = nn.ModuleList(
            [ResidualBlock(64 if i == 0 else 128, 128) for i in range(4)]
        )
        self.layer3 = nn.ModuleList(
            [ResidualBlock(128 if i == 0 else 256, 256) for i in range(6)]
        )
        self.layer4 = nn.ModuleList(
            [ResidualBlock(256 if i == 0 else 512, 512) for i in range(6)]
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # dropout ensures that a model doesnt memorize certain neurons and
        # become heavily dependant on them. it drops 0.5 random neurons so
        # that the network is forced to learn accross all of them
        self.dropout = nn.Dropout(0.5)
        self.linlayer = nn.Linear(512, num_classes)

    def foward(self, x):
        out = self.conv1(x)
        for block in self.layer1:
            out = block(out)
        for block in self.layer2:
            out = block(out)
        for block in self.layer3:
            out = block(out)
        for block in self.layer4:
            out = block(out)
        out = self.avg_pool(out)
        out = out.view(x.size(0), -1)
        out = self.dropout(out)
        out = self.linlayer(out)
        return out
