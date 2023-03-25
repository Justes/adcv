import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

# from torchsummary import summary

model_urls = {
    "linnet16": 'linnet16-pretrained-imagenet1k-best.pth'
}


class LinNet(nn.Module):

    def __init__(self, num_classes, block, layers, dropout_p=0, *args):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 32, 7, 1, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1])
        self.layer3 = self._make_layer(block, 256, layers[2])
        self.layer4 = self._make_layer(block, 512, layers[3])

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1d = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(dropout_p)
        self.classifer = nn.Linear(512, num_classes)

        self._init_params()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
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

        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn1d(x)
        x = self.relu(x)
        x = self.dropout(x)
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
                nn.Conv2d(planes * block.expansion, self.inplanes, 1, 1, bias=False),
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


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inc, outc, stride, downsample):
        super().__init__()
        self.conv1 = nn.Conv2d(inc, outc, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outc)
        self.conv2 = nn.Conv2d(outc, inc, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(inc)
        self.conv3 = nn.Conv2d(inc, outc, 3, 1, 1, bias=False)
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


def linnet16(num_classes, pretrained=True, dropout_p=0, **kwargs):
    model = LinNet(num_classes=num_classes, block=Bottleneck, layers=[1, 1, 1, 1], dropout_p=dropout_p)
    if pretrained and kwargs.get("pretrained_model") != "":
        print(kwargs.get("pretrained_model", model_urls["linnet16"]))
        init_pretrained_weights(model, kwargs.get("pretrained_model", model_urls["linnet16"]))
    return model


def linnet19(num_classes, pretrained=True, **kwargs):
    model = LinNet(num_classes=num_classes, block=Bottleneck, layers=[1, 1, 1, 2])
    if pretrained and kwargs.get("pretrained_model") != "":
        print(kwargs.get("pretrained_model", model_urls["linnet16"]))
        init_pretrained_weights(model, kwargs.get("pretrained_model", model_urls["linnet16"]))
    return model


def init_pretrained_weights(model, model_url):
    """
    Initialize model with pretrained weights.
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    # pretrain_dict = model_zoo.load_url(model_url)
    use_mps = torch.backends.mps.is_available()
    device = 'mps' if use_mps else 'cpu'
    pretrained = torch.load(model_url, map_location=torch.device(device))
    pretrain_dict = pretrained.state_dict()
    print('Pretrained linnet16 loaded')
    model_dict = model.state_dict()
    pretrain_dict = {
        k: v
        for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    print(f"Initialized model with pretrained weights from {model_url}")
