import torch.nn as nn
from torchvision import models


class PaddyResNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
