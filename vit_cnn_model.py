# model/vit_cnn_model.py
"""
Hybrid ResNet50 + ViT model.

- Uses torchvision's ResNet50 and ViT-B/16 backbones.
- Removes classifier heads from both backbones, projects each to a common embedding size,
  concatenates them, and runs a small MLP classifier that outputs raw logits.

Output:
    Tensor of shape (B, num_outputs) with raw logits (use BCEWithLogitsLoss for binary).
"""
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet18, ResNet50_Weights, ResNet18_Weights
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights


class HybridModel(nn.Module):
    def __init__(
        self,
        cnn_backbone: str = 'resnet50',  # 'resnet50' or 'resnet18' (lighter)
        cnn_pretrained: bool = True,
        vit_pretrained: bool = True,
        cnn_proj_dim: int = 512,
        vit_proj_dim: int = 512,
        classifier_hidden: int = 256,
        num_outputs: int = 1,
        freeze_backbones: bool = False,
        dropout: float = 0.1,
        init_weights: bool = True,
    ):
        """
        Args:
            cnn_pretrained: use pretrained weights for ResNet50 if True.
            vit_pretrained: use pretrained weights for ViT-B/16 if True.
            cnn_proj_dim: output dim of CNN projection head.
            vit_proj_dim: output dim of ViT projection head.
            classifier_hidden: hidden size in final classifier MLP.
            num_outputs: number of output units (1 for binary logits).
            freeze_backbones: if True, freeze backbone parameters (heads/projection layers remain trainable).
            dropout: dropout probability used in projection/classifier heads.
        """
        super().__init__()

        # --- CNN backbone (ResNet50 or ResNet18) ---
        if cnn_backbone == 'resnet18':
            cnn_weights = ResNet18_Weights.DEFAULT if cnn_pretrained else None
            cnn_model = resnet18(weights=cnn_weights)
            in_features = 512
        else:
            cnn_weights = ResNet50_Weights.DEFAULT if cnn_pretrained else None
            cnn_model = resnet50(weights=cnn_weights)
            in_features = 2048

        # remove the classifier head and use a small projection head
        cnn_model.fc = nn.Identity()
        self.cnn = cnn_model
        self.cnn_head = nn.Sequential(
            nn.Linear(in_features, cnn_proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # --- ViT backbone (ViT-B/16) ---
        vit_weights = ViT_B_16_Weights.DEFAULT if vit_pretrained else None
        vit = vit_b_16(weights=vit_weights)
        # Save the input dim of the head, then remove it
        vit_in_features = vit.heads.head.in_features
        vit.heads.head = nn.Identity()
        self.vit = vit
        self.vit_head = nn.Sequential(
            nn.Linear(vit_in_features, vit_proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # --- Final classifier ---
        joint_dim = cnn_proj_dim + vit_proj_dim
        self.classifier = nn.Sequential(
            nn.Linear(joint_dim, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_outputs),
        )

        # optional: initialize heads for faster convergence when training from scratch
        if init_weights:
            self._init_weights()

        if freeze_backbones:
            # Freeze all backbone params (cnn and vit). Projection heads and classifier remain trainable.
            for param in self.cnn.parameters():
                param.requires_grad = False
            for param in self.vit.parameters():
                param.requires_grad = False

    def _init_weights(self):
        # Xavier init for linear layers in heads and classifier
        def init_linear(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        self.cnn_head.apply(init_linear)
        self.vit_head.apply(init_linear)
        self.classifier.apply(init_linear)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input images tensor of shape (B, 3, H, W) expected to be 224x224 for pretrained weights.
        Returns:
            logits: tensor of shape (B, num_outputs)
        """
        # CNN branch
        cnn_feats = self.cnn(x)              # (B, 2048)
        cnn_proj = self.cnn_head(cnn_feats)  # (B, cnn_proj_dim)

        # ViT branch
        vit_feats = self.vit(x)              # (B, vit_in_features)
        vit_proj = self.vit_head(vit_feats)  # (B, vit_proj_dim)

        # Concatenate and classify
        joint = torch.cat([cnn_proj, vit_proj], dim=1)  # (B, joint_dim)
        logits = self.classifier(joint)                 # (B, num_outputs)
        return logits


if __name__ == "__main__":
    # Quick smoke test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridModel(cnn_pretrained=False, vit_pretrained=False).to(device).eval()
    dummy = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(dummy)
    print("Output shape:", out.shape)  # expect (2, 1) by default
