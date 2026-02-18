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

        out = self.conv2(x)
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
