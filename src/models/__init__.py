# Copyright (c) EEEM071, University of Surrey

from .resnet import resnet18, resnet18_fc256, resnet34, resnet50, resnet50_fc512, resnet101, resnet152
from .linnet import linnet16, linnet19

__model_factory = {
    # image classification models
    "resnet18": resnet18,
    "resnet18_fc256": resnet18_fc256,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet50_fc512": resnet50_fc512,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "linnet16": linnet16,
    "linnet19": linnet19
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError(f"Unknown model: {name}")
    return __model_factory[name](*args, **kwargs)
