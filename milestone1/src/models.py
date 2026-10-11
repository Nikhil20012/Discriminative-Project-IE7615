import torch.nn as nn
from torchvision import models

MODEL_NAMES = ["custom", "mobilenet", "resnet50"]


class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 256))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(name, num_classes, pretrained=True):
    if name == "custom":
        return CustomCNN(num_classes)
    if name == "mobilenet":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise ValueError(f"unknown model: {name}")


def head(model, name):
    return model.fc if name == "resnet50" else model.classifier


def stages(model, name):
    if name == "resnet50":
        return [model.layer1, model.layer2, model.layer3, model.layer4]
    return list(model.features)


def freeze_backbone(model, name):
    for p in model.parameters():
        p.requires_grad = False
    for p in head(model, name).parameters():
        p.requires_grad = True


def unfreeze_top(model, name, n_stages):
    for stage in stages(model, name)[-n_stages:]:
        for p in stage.parameters():
            p.requires_grad = True


def frozen_bn_to_eval(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


def count_params(model):
    return sum(p.numel() for p in model.parameters())