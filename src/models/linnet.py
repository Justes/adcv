import torch
import torch.nn as nn
from torchsummary import summary

class LinNet(nn.Module):

    def __init__(self, num_classes, block, layers, *args):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 32, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.layer4 = self._make_layer(block, 512, layers[3])
        #self.layer5 = self._make_layer(block, 1024, layers[3])

        """
        self.block1 = Bottleneck(64, 128)
        self.block2 = Bottleneck(128, 256)
        self.block3 = Bottleneck(256, 512)
        self.block4 = Bottleneck(512, 1024)
        """

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(1024, 512)
        self.classifer = nn.Linear(512, num_classes)

        self._init_params()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        x = self.layer3(x)
        x = self.maxpool(x)
        x = self.layer4(x)
        #x = self.maxpool(x)
        #x = self.layer5(x)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.maxpool(x)
        x = self.block3(x)
        x = self.maxpool(x)
        x = self.block4(x)
        """
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        if not self.training:
            return x

        out = self.classifer(x)

        return out, x


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes * block.expansion, stride, downsample))
        
        for i in range(1, blocks):
            layers.append(nn.Sequential(
                    nn.Conv2d(planes * block.expansion, self.inplanes, 1, 1),
                    nn.BatchNorm2d(self.inplanes),
                ))
            layers.append(block(self.inplanes, planes * block.expansion, stride, downsample))

        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)


    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inc, outc, stride, downsample):
        super().__init__()
        self.conv1 = nn.Conv2d(inc, outc, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(outc)
        self.conv2 = nn.Conv2d(outc, inc, 1)
        self.bn2 = nn.BatchNorm2d(inc)
        self.conv3 = nn.Conv2d(inc, outc, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(outc)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        residual = self.downsample(residual)

        out = residual + x 

        return self.relu(out)


def linnet16(num_classes, **kwargs):
    model = LinNet(num_classes=num_classes, block=Bottleneck, layers=[1, 1, 1, 1])
    return model

def linnet19(num_classes, **kwargs):
    model = LinNet(num_classes=num_classes, block=Bottleneck, layers=[1, 1, 1, 1, 1])
    return model

def linnet28(num_classes, **kwargs):
    model = LinNet(num_classes=num_classes, block=Bottleneck, layers=[1, 1, 2, 2])
    return model

