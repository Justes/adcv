# Copyright (c) EEEM071, University of Surrey

from .resnet import resnet18, resnet18_fc256, resnet18_fc512, resnet34, resnet34_fc512, resnet50, resnet50_fc512, resnet101, resnet152
from .linnet import linnet16
from .tvmodels import mobilenet_v3_small, vgg16, googlenet
from .alexnet import alexnet

__model_factory = {
    # image classification models
    "resnet18": resnet18,
    "resnet18_fc256": resnet18_fc256,
    "resnet18_fc512": resnet18_fc512,
    "resnet34": resnet34,
    "resnet34_fc512": resnet34_fc512,
    "resnet50": resnet50,
    "resnet50_fc512": resnet50_fc512,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "mobilenet_v3_small": mobilenet_v3_small,
    "vgg16": vgg16,
    "alexnet": alexnet,
    "googlenet": googlenet,
    "linnet16": linnet16
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError(f"Unknown model: {name}")
    return __model_factory[name](*args, **kwargs)
