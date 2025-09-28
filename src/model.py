import torch
import torch.nn as nn
from torchvision import models


class PaddyMultimodalNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_varieties: int,
        embedding_dim: int = 32,
        age_mlp_dim: int = 16,
    ):
        super().__init__()

        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        backbone_out_feature = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # remove the final layer

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.variety_embedding = nn.Embedding(
            num_embeddings=num_varieties, embedding_dim=embedding_dim
        )

        self.age_processor = nn.Sequential(
            nn.Linear(1, age_mlp_dim), nn.ReLU(), nn.BatchNorm1d(age_mlp_dim)
        )

        combined_feature_dim = backbone_out_feature + embedding_dim + age_mlp_dim

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(combined_feature_dim),
            nn.Linear(combined_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(
        self,
        image: torch.Tensor,
        variety_idx: torch.Tensor,
        age: torch.Tensor,
    ) -> torch.Tensor:
        image_features = self.backbone(image)  # (B, backbone_out_feature)

        variety_features = self.variety_embedding(variety_idx)
        # age needs to be unsqueezed to have a feature dimension of 1 for the linear layer
        age_features = self.age_processor(age.unsqueeze(1).float())
        combined_features = torch.cat(
            [image_features, variety_features, age_features], dim=1
        )

        output = self.classifier(combined_features)

        return output
