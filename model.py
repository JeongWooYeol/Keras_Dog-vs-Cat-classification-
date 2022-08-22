import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int,
                 out_channels: int,
                 stride: int = 1):
        super(BasicBlock, self).__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size= 3, stride= stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU(inplace = True)
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels = in_channels, out_channels = out_channels * BasicBlock.expansion, kernel_size= 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
    expansion = 1

    def forward(self, x):
        result = self.residual_function(x) + self.shortcut(x)
        result = self.relu(result)
        return result


class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(BottleNeck, self).__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, stride = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = out_channels, out_channels = out_channels * BottleNeck.expansion, kernel_size = 1, stride = 1, bias = False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion)
        )
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU(inplace = True)

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels = in_channels, out_channels = out_channels * BottleNeck.expansion, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )
    expansion = 4

    def forward(self, x):
        result = self.residual_function(x) + self.shortcut(x)
        result = self.relu(result)
        return result

class ResNet(nn.Module):
    def __init__(self, block, num_block,  init_weights = True):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
        self.conv2 = self.make_layer(block, 64, num_block[0], 1)
        self.conv3 = self.make_layer(block, 64 * 2, num_block[1], stride = 2)
        self.conv4 = self.make_layer(block, 64 * 4, num_block[2], stride = 2)
        self.conv5 = self.make_layer(block, 64 * 8, num_block[3], stride = 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(in_features = 512 * block.expansion, out_features= 500),
            nn.Linear(500, 2)
        )

        if init_weights:
            self.initialize_weights()

    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        x = self.conv3(output)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
def resnet18():
    return ResNet(BottleNeck, [2, 2, 2, 2])








