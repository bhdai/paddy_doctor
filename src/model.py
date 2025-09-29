import torch.nn as nn
import timm


class TimmPaddyNet(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()

        # setting num_classes = 0 to remove the original classifier head
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )

        in_features = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
