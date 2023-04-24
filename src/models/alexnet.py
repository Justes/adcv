import torch
import torch.nn as nn


class AlexNet(nn.Module):

    def __init__(self, num_classes, dropout_p=0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(),
        )

        self.fc = nn.Linear(4096, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        if not self.training:
            return x

        logits = self.fc(x)

        return logits, x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def alexnet(num_classes, pretrained=True,  **kwargs):
    model = AlexNet(num_classes)
    if pretrained and kwargs.get("pretrained_model") != "":
        print(kwargs.get("pretrained_model"))
        init_pretrained_weights(model, kwargs.get("pretrained_model"))
    return model


def init_pretrained_weights(model, model_url):
    """
    Initialize model with pretrained weights.
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    # pretrain_dict = model_zoo.load_url(model_url)
    use_mps = torch.backends.mps.is_available()
    device = 'mps' if use_mps else 'cpu'
    pretrain_dict = torch.load(model_url, map_location=torch.device(device))
    for k, v in pretrain_dict.items():
        print(k, v.shape)

    #pretrain_dict = pretrained.state_dict()
    print('Pretrained model loaded')
    model_dict = model.state_dict()
    for k, v in model_dict.items():
        print(k, v.shape)

    pretrain_dict = {
        k: v
        for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }

    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    print(f"Initialized model with pretrained weights from {model_url}")

if __name__ == "__main__":
    tmp = torch.randn(2, 3, 224, 224)
    net = AlexNet(576)
    out = net(tmp)
    print('alex out:', out.shape)
