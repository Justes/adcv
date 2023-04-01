import torch
import torch.nn as nn
from .vision_transformer import vit_base_patch16_224


class VitVeri(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.in_planes = 768
        self.base = vit_base_patch16_224()
        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x):
        base = self.base(x)
        feat = self.bottleneck(base)

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, base

        return base


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def vit_base_veri(num_classes):
    model = VitVeri(num_classes)
    return model


if __name__ == "__main__":
    veri = VitVeri(576)
    print(veri)