import torch
import torch.nn as nn
from torchvision import models


class PaddyResNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        for param in self.base_model.parameters():
            param.requires_grad = False

        for param in self.base_model.layer4.parameters():
            param.requires_grad = True

        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)
