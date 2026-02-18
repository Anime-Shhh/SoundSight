from sys import flags
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=3):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(
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
