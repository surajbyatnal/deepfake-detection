# model/vit_cnn_model.py
"""
Hybrid CNN + ViT models with multiple fusion strategies.

- Uses torchvision's ResNet50/ResNet18 and ViT-B/16 backbones.
- Multiple fusion strategies: concatenation, attention-based, cross-modal attention, bilinear.
- Removes classifier heads from both backbones, projects features, fuses them, and runs MLP classifier.

Output:
    Tensor of shape (B, num_outputs) with raw logits (use BCEWithLogitsLoss for binary).
"""
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import resnet50, resnet18, ResNet50_Weights, ResNet18_Weights
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights


class AttentionFusion(nn.Module):
    """Learns to weight CNN and ViT features using attention."""
    def __init__(self, dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, cnn_feat: torch.Tensor, vit_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cnn_feat: (B, dim)
            vit_feat: (B, dim)
        Returns:
            fused: (B, dim)
        """
        combined = torch.cat([cnn_feat, vit_feat], dim=1)  # (B, dim*2)
        weights = self.attention(combined)  # (B, 2)
        
        # Apply learned weights
        fused = weights[:, 0:1] * cnn_feat + weights[:, 1:2] * vit_feat  # (B, dim)
        return fused


class CrossModalAttention(nn.Module):
    """Cross-modal attention fusion between CNN and ViT features."""
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        # Projection layers
        self.cnn_q = nn.Linear(dim, dim)
        self.vit_k = nn.Linear(dim, dim)
        self.vit_v = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        
    def forward(self, cnn_feat: torch.Tensor, vit_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cnn_feat: (B, dim)
            vit_feat: (B, dim)
        Returns:
            fused: (B, dim)
        """
        B = cnn_feat.shape[0]
        
        # Project
        q = self.cnn_q(cnn_feat)  # (B, dim)
        k = self.vit_k(vit_feat)  # (B, dim)
        v = self.vit_v(vit_feat)  # (B, dim)
        
        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, self.head_dim)  # (B, num_heads, head_dim)
        k = k.view(B, self.num_heads, self.head_dim)
        v = v.view(B, self.num_heads, self.head_dim)
        
        # Attention
        scores = torch.einsum('bnd,bmd->bnm', q, k) / (self.head_dim ** 0.5)  # (B, num_heads, 1, 1)
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum('bnm,bmd->bnd', attn, v)  # (B, num_heads, head_dim)
        out = out.reshape(B, self.dim)
        
        # Final projection
        out = self.output(out)
        return out


class BilinearFusion(nn.Module):
    """Bilinear fusion between CNN and ViT features."""
    def __init__(self, dim: int, out_dim: Optional[int] = None):
        super().__init__()
        out_dim = out_dim or dim
        self.dim = dim
        self.bilinear = nn.Bilinear(dim, dim, out_dim)
    
    def forward(self, cnn_feat: torch.Tensor, vit_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cnn_feat: (B, dim)
            vit_feat: (B, dim)
        Returns:
            fused: (B, out_dim)
        """
        return self.bilinear(cnn_feat, vit_feat)


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
        fusion_strategy: str = 'concat',  # 'concat', 'attention', 'cross_modal', 'bilinear', 'add', 'mul'
    ):
        """
        Args:
            cnn_backbone: 'resnet50' or 'resnet18'.
            cnn_pretrained: use pretrained weights for CNN if True.
            vit_pretrained: use pretrained weights for ViT-B/16 if True.
            cnn_proj_dim: output dim of CNN projection head.
            vit_proj_dim: output dim of ViT projection head.
            classifier_hidden: hidden size in final classifier MLP.
            num_outputs: number of output units (1 for binary logits).
            freeze_backbones: if True, freeze backbone parameters.
            dropout: dropout probability used in projection/classifier heads.
            fusion_strategy: how to fuse CNN and ViT features.
                - 'concat': concatenate features (original)
                - 'attention': learned weighted sum
                - 'cross_modal': cross-modal attention
                - 'bilinear': bilinear fusion
                - 'add': element-wise addition
                - 'mul': element-wise multiplication
        """
        super().__init__()
        
        self.fusion_strategy = fusion_strategy

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
        # Compute the dimension after fusion
        if fusion_strategy == 'concat':
            joint_dim = cnn_proj_dim + vit_proj_dim
        elif fusion_strategy in ['add', 'mul']:
            # Element-wise operations require same dimension
            assert cnn_proj_dim == vit_proj_dim, f"For {fusion_strategy} fusion, cnn_proj_dim must equal vit_proj_dim"
            joint_dim = cnn_proj_dim
        elif fusion_strategy == 'attention':
            assert cnn_proj_dim == vit_proj_dim, "For attention fusion, cnn_proj_dim must equal vit_proj_dim"
            self.fusion_module = AttentionFusion(cnn_proj_dim)
            joint_dim = cnn_proj_dim
        elif fusion_strategy == 'cross_modal':
            assert cnn_proj_dim == vit_proj_dim, "For cross_modal fusion, cnn_proj_dim must equal vit_proj_dim"
            self.fusion_module = CrossModalAttention(cnn_proj_dim, num_heads=4)
            joint_dim = cnn_proj_dim
        elif fusion_strategy == 'bilinear':
            self.fusion_module = BilinearFusion(cnn_proj_dim, vit_proj_dim)
            joint_dim = vit_proj_dim
        else:
            raise ValueError(f"Unknown fusion_strategy: {fusion_strategy}")
        
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
        
        # Initialize fusion module if it exists
        if hasattr(self, 'fusion_module'):
            self.fusion_module.apply(init_linear)

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

        # Fusion strategy
        if self.fusion_strategy == 'concat':
            joint = torch.cat([cnn_proj, vit_proj], dim=1)  # (B, joint_dim)
        elif self.fusion_strategy == 'add':
            joint = cnn_proj + vit_proj  # (B, dim)
        elif self.fusion_strategy == 'mul':
            joint = cnn_proj * vit_proj  # (B, dim)
        elif self.fusion_strategy == 'attention':
            joint = self.fusion_module(cnn_proj, vit_proj)  # (B, dim)
        elif self.fusion_strategy == 'cross_modal':
            joint = self.fusion_module(cnn_proj, vit_proj)  # (B, dim)
        elif self.fusion_strategy == 'bilinear':
            joint = self.fusion_module(cnn_proj, vit_proj)  # (B, out_dim)
        else:
            raise ValueError(f"Unknown fusion_strategy: {self.fusion_strategy}")

        # Classify
        logits = self.classifier(joint)  # (B, num_outputs)
        return logits


if __name__ == "__main__":
    # Quick smoke test with different fusion strategies
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy = torch.randn(2, 3, 224, 224, device=device)
    
    fusion_strategies = ['concat', 'add', 'mul', 'attention', 'cross_modal', 'bilinear']
    
    print("Testing HybridModel with different fusion strategies:")
    for strategy in fusion_strategies:
        try:
            # For add/mul, ensure equal projection dims
            proj_dim = 512 if strategy in ['add', 'mul'] else 512
            model = HybridModel(
                cnn_pretrained=False, 
                vit_pretrained=False,
                cnn_proj_dim=proj_dim,
                vit_proj_dim=proj_dim,
                fusion_strategy=strategy
            ).to(device).eval()
            
            with torch.no_grad():
                out = model(dummy)
            print(f"  {strategy:15s} -> Output shape: {out.shape}")
        except Exception as e:
            print(f"  {strategy:15s} -> Error: {e}")
