"""
models.py
---------
All model components for the embryo development classification pipeline.

Components:
  - EmbryoEncoder       : ResNet or ViT backbone
  - ProjectionHead      : MLP head for contrastive learning
  - MAEDecoder          : Transformer decoder for masked reconstruction
  - FocalFusionModule   : Fuse N focal-plane embeddings (mean/max/attention)
  - TemporalHead        : GRU or Transformer over time
  - TemporalGapHead     : Predict time-gap bin (temporal SSL)
  - ExpertHead          : Single classification head (one expert in MoE)
  - RouterNetwork       : Soft gating over experts
  - SoftMoEClassifier   : Mixture-of-Experts classifier
  - EmbryoPretrainModel : Full pretraining model
  - EmbryoFinetuneModel : Full fine-tuning model
"""

import math
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models


# =============================================================================
# Backbone: ResNet
# =============================================================================

class ResNetEncoder(nn.Module):
    """ResNet backbone with configurable depth. Returns feature embedding."""

    def __init__(self, arch: str = "resnet50", pretrained: bool = False,
                 embed_dim: int = 2048):
        super().__init__()
        weights = "IMAGENET1K_V1" if pretrained else None
        factory = {
            "resnet18": tv_models.resnet18,
            "resnet34": tv_models.resnet34,
            "resnet50": tv_models.resnet50,
        }
        if arch not in factory:
            raise ValueError(f"Unknown ResNet arch: {arch}. Choose from {list(factory)}")

        net = factory[arch](weights=weights)
        out_features = net.fc.in_features

        # Remove final FC layer
        self.features = nn.Sequential(*list(net.children())[:-1])  # → [B, C, 1, 1]
        self.out_dim = out_features

        # Optional projection to embed_dim
        if embed_dim != out_features:
            self.proj = nn.Linear(out_features, embed_dim)
            self.out_dim = embed_dim
        else:
            self.proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] → [B, embed_dim]"""
        feat = self.features(x)
        feat = feat.flatten(1)
        if self.proj is not None:
            feat = self.proj(feat)
        return feat


# =============================================================================
# Backbone: ViT (lightweight implementation for MAE compatibility)
# =============================================================================

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # [B, N, D]


class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, dim)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.drop_path(attn_out)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x * random_tensor / keep_prob


class ViTEncoder(nn.Module):
    """Lightweight Vision Transformer encoder."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.out_dim = embed_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_ratio, dpr[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor, return_tokens: bool = False):
        """
        x: [B, C, H, W]
        Returns: [B, embed_dim] (cls token) or ([B, embed_dim], [B, N, embed_dim])
        """
        B = x.shape[0]
        x = self.patch_embed(x)  # [B, N, D]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if return_tokens:
            return x[:, 0], x[:, 1:]  # cls, patch_tokens
        return x[:, 0]


def build_encoder(cfg: dict) -> nn.Module:
    """Factory to build backbone from config."""
    model_cfg = cfg["model"]
    arch = model_cfg["backbone"]
    embed_dim = model_cfg["embed_dim"]
    in_chans = cfg["data"]["image_channels"]
    img_size = cfg["data"]["image_size"][0]

    if arch.startswith("vit"):
        vc = model_cfg["vit"]
        return ViTEncoder(
            img_size=img_size,
            patch_size=vc["patch_size"],
            in_chans=in_chans,
            embed_dim=vc["embed_dim"],
            depth=vc["depth"],
            num_heads=vc["num_heads"],
            mlp_ratio=vc["mlp_ratio"],
            drop_path_rate=vc["drop_path_rate"],
        )
    else:
        rc = model_cfg["resnet"]
        return ResNetEncoder(
            arch=arch,
            pretrained=rc["pretrained"],
            embed_dim=embed_dim,
        )


# =============================================================================
# Projection Head
# =============================================================================

class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning (SimCLR/MoCo style)."""

    def __init__(self, in_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 3, use_bn: bool = True):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=not use_bn))
            if i < len(dims) - 2:  # no BN/ReLU on last layer
                if use_bn:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# =============================================================================
# MAE Decoder
# =============================================================================

class MAEDecoder(nn.Module):
    """
    Transformer decoder for masked image modeling (MAE-style).

    Reconstructs pixel patches from visible token embeddings.
    """

    def __init__(
        self,
        num_patches: int,
        patch_size: int,
        in_chans: int,
        encoder_embed_dim: int,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.in_chans = in_chans
        patch_dim = patch_size * patch_size * in_chans

        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim)
        )

        self.decoder_blocks = nn.ModuleList([
            ViTBlock(decoder_embed_dim, decoder_num_heads)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_dim)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

    def forward(
        self, x: torch.Tensor, ids_restore: torch.Tensor
    ) -> torch.Tensor:
        """
        x: visible tokens [B, N_vis, D]
        ids_restore: [B, N] indices to restore token order
        Returns: [B, N_patches, patch_dim]
        """
        x = self.decoder_embed(x)

        # Append mask tokens
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1] + 1, 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        x = torch.cat([x[:, :1, :], x_], dim=1)  # prepend cls token

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x[:, 1:, :])  # remove cls token → [B, N, patch_dim]
        return x


def random_masking(
    x: torch.Tensor, mask_ratio: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomly mask patches.

    Returns:
        x_masked: [B, N_vis, D] visible tokens (with cls)
        mask: [B, N] binary mask (1=masked)
        ids_restore: [B, N] indices to restore original order
    """
    B, N, D = x.shape
    len_keep = int(N * (1 - mask_ratio))

    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

    mask = torch.ones(B, N, device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore


# =============================================================================
# Multi-Focal Fusion Module
# =============================================================================

class FocalFusionModule(nn.Module):
    """
    Fuse N focal-plane embeddings into a single embedding.

    Methods:
        'mean': simple average
        'max': element-wise max
        'attention': learned query-based attention pooling
        'cross_attention': cross-attention between focal planes
    """

    def __init__(self, embed_dim: int, num_focal: int, method: str = "attention",
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.method = method
        self.embed_dim = embed_dim
        self.num_focal = num_focal

        if method == "attention":
            self.attn_query = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                               dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(embed_dim)
        elif method == "cross_attention":
            self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                               dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(embed_dim)
            self.cls = nn.Parameter(torch.randn(1, 1, embed_dim))
        elif method in ("mean", "max"):
            pass
        else:
            raise ValueError(f"Unknown fusion method: {method}")

    def forward(self, focal_embeds: torch.Tensor) -> torch.Tensor:
        """
        focal_embeds: [B, N_focal, embed_dim]
        Returns: [B, embed_dim]
        """
        if self.method == "mean":
            return focal_embeds.mean(dim=1)

        elif self.method == "max":
            return focal_embeds.max(dim=1).values

        elif self.method == "attention":
            B = focal_embeds.shape[0]
            query = self.attn_query.expand(B, -1, -1)  # [B, 1, D]
            out, _ = self.attn(query, focal_embeds, focal_embeds)
            return self.norm(out.squeeze(1))  # [B, D]

        elif self.method == "cross_attention":
            B = focal_embeds.shape[0]
            cls = self.cls.expand(B, -1, -1)
            x = torch.cat([cls, focal_embeds], dim=1)  # [B, 1+N, D]
            out, _ = self.attn(x, x, x)
            return self.norm(out[:, 0])  # cls token output


# =============================================================================
# Temporal Head
# =============================================================================

class TemporalHead(nn.Module):
    """
    Process a sequence of frame embeddings over time.
    Returns sequence-level embeddings (same length as input).
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2,
                 temporal_type: str = "gru", dropout: float = 0.1,
                 transformer_heads: int = 8, transformer_depth: int = 4):
        super().__init__()
        self.type = temporal_type

        if temporal_type == "gru":
            self.rnn = nn.GRU(
                input_dim, hidden_dim, num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True, bidirectional=False
            )
            self.out_dim = hidden_dim
            self.norm = nn.LayerNorm(hidden_dim)

        elif temporal_type == "transformer":
            self.pos_embed = nn.Parameter(torch.randn(1, 1000, input_dim) * 0.02)
            self.blocks = nn.ModuleList([
                ViTBlock(input_dim, transformer_heads)
                for _ in range(transformer_depth)
            ])
            self.norm = nn.LayerNorm(input_dim)
            self.out_dim = input_dim
        else:
            raise ValueError(f"Unknown temporal type: {temporal_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        Returns: [B, T, out_dim]
        """
        if self.type == "gru":
            out, _ = self.rnn(x)  # [B, T, hidden_dim]
            return self.norm(out)

        elif self.type == "transformer":
            T = x.shape[1]
            x = x + self.pos_embed[:, :T, :]
            for blk in self.blocks:
                x = blk(x)
            return self.norm(x)


# =============================================================================
# Temporal Gap Prediction Head (pretraining)
# =============================================================================

class TemporalGapHead(nn.Module):
    """Predict the time-gap bin between two timepoints."""

    def __init__(self, in_dim: int, num_bins: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_bins),
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """z1, z2: [B, D] → [B, num_bins]"""
        return self.net(torch.cat([z1, z2], dim=-1))


# =============================================================================
# MoE Classifier Components
# =============================================================================

class ExpertHead(nn.Module):
    """Single expert classification head for a stage group."""

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 512,
                 dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RouterNetwork(nn.Module):
    """Soft router that assigns weights to each expert."""

    def __init__(self, in_dim: int, num_experts: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns soft weights [B, num_experts]."""
        return F.softmax(self.net(x), dim=-1)


class SoftMoEClassifier(nn.Module):
    """
    Soft Mixture-of-Experts classifier.

    - Router assigns soft weights to each expert.
    - Each expert classifies the full 16-stage space.
    - Final prediction = weighted sum of expert logits.
    - Auxiliary: router_weights returned for diversity loss.
    """

    def __init__(self, in_dim: int, num_classes: int, num_experts: int = 3,
                 router_hidden: int = 256, head_hidden: int = 512,
                 dropout: float = 0.2):
        super().__init__()
        self.num_experts = num_experts

        self.router = RouterNetwork(in_dim, num_experts, router_hidden)
        self.experts = nn.ModuleList([
            ExpertHead(in_dim, num_classes, head_hidden, dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, D]
        Returns: logits [B, num_classes], router_weights [B, num_experts]
        """
        weights = self.router(x)  # [B, num_experts]

        expert_logits = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )  # [B, num_experts, num_classes]

        # Weighted sum
        logits = (weights.unsqueeze(-1) * expert_logits).sum(dim=1)  # [B, num_classes]
        return logits, weights


# =============================================================================
# Auxiliary heads for fine-tuning
# =============================================================================

class CellCountHead(nn.Module):
    """Auxiliary head: predict cell count (1/2/4/8) for early stages."""

    def __init__(self, in_dim: int, num_counts: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_counts)
        )

    def forward(self, x):
        return self.net(x)


class BlastocystScoreHead(nn.Module):
    """Auxiliary head: predict blastocyst expansion score (0-4) for late stages."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# =============================================================================
# Full Pretraining Model
# =============================================================================

class EmbryoPretrainModel(nn.Module):
    """
    Full model for self-supervised pretraining.

    Forward modes (controlled by which losses are active):
      - contrastive: encode two views, project, compute InfoNCE
      - mae: mask patches, reconstruct
      - temporal: encode two timepoints, predict gap bin
    """

    def __init__(self, cfg: dict):
        super().__init__()
        model_cfg = cfg["model"]
        pretrain_cfg = cfg["pretrain"]
        embed_dim = model_cfg["embed_dim"]
        in_chans = cfg["data"]["image_channels"]

        # Shared encoder
        self.encoder = build_encoder(cfg)
        encoder_out_dim = self.encoder.out_dim

        # Projection head (contrastive)
        ph = model_cfg["projection_head"]
        self.projection_head = ProjectionHead(
            in_dim=encoder_out_dim,
            hidden_dim=ph["hidden_dim"],
            output_dim=ph["output_dim"],
            num_layers=ph["num_layers"],
            use_bn=ph["use_bn"],
        )

        # MAE decoder (only for ViT backbone)
        self.is_vit = cfg["model"]["backbone"].startswith("vit")
        if self.is_vit:
            vc = model_cfg["vit"]
            img_size = cfg["data"]["image_size"][0]
            patch_size = vc["patch_size"]
            num_patches = (img_size // patch_size) ** 2
            mae_cfg = model_cfg["mae_decoder"]
            self.mae_decoder = MAEDecoder(
                num_patches=num_patches,
                patch_size=patch_size,
                in_chans=in_chans,
                encoder_embed_dim=encoder_out_dim,
                decoder_embed_dim=mae_cfg["decoder_embed_dim"],
                decoder_depth=mae_cfg["decoder_depth"],
                decoder_num_heads=mae_cfg["decoder_num_heads"],
            )
            self.mask_ratio = pretrain_cfg["mae"]["mask_ratio"]
        else:
            self.mae_decoder = None

        # Temporal gap prediction head
        tg_cfg = model_cfg["temporal_gap_head"]
        num_gap_bins = len(pretrain_cfg["temporal_ssl"]["gap_bins"]) + 1
        self.temporal_gap_head = TemporalGapHead(
            in_dim=encoder_out_dim,
            num_bins=num_gap_bins,
            hidden_dim=tg_cfg["hidden_dim"],
        )

        # Focal fusion (used before projection in contrastive mode)
        fc = model_cfg["fusion"]
        self.fusion = FocalFusionModule(
            embed_dim=encoder_out_dim,
            num_focal=cfg["data"]["num_focal_planes"],
            method=fc["method"],
            num_heads=fc["num_heads"],
            dropout=fc["dropout"],
        )

    def encode_focal(self, focal_imgs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        focal_imgs: [B, N_focal, C, H, W]
        Returns:
            focal_embeds: [B, N_focal, D]  (per-plane embeddings)
            fused:        [B, D]           (fused embedding)
        """
        B, N, C, H, W = focal_imgs.shape
        imgs_flat = focal_imgs.view(B * N, C, H, W)
        embeds_flat = self.encoder(imgs_flat)  # [B*N, D]
        focal_embeds = embeds_flat.view(B, N, -1)
        fused = self.fusion(focal_embeds)
        return focal_embeds, fused

    def forward_contrastive(
        self, view1: torch.Tensor, view2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns normalized projections z1, z2 for InfoNCE."""
        _, fused1 = self.encode_focal(view1)
        _, fused2 = self.encode_focal(view2)
        z1 = self.projection_head(fused1)
        z2 = self.projection_head(fused2)
        return z1, z2

    def forward_cross_focal_contrastive(
        self, focal_imgs: torch.Tensor
    ) -> torch.Tensor:
        """
        Per-plane contrastive: each focal plane embedding is projected.
        Returns: [B * N_focal, proj_dim] projections with positives = same (B, t).
        """
        B, N, C, H, W = focal_imgs.shape
        flat = focal_imgs.view(B * N, C, H, W)
        embeds = self.encoder(flat)  # [B*N, D]
        projs = self.projection_head(embeds)  # [B*N, proj_dim]
        return projs  # arrange: (f0_b0, f0_b1, ..., f1_b0, ...) by caller

    def forward_mae(
        self, focal_imgs: torch.Tensor, target_focal_idx: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MAE forward pass (ViT only).

        focal_imgs: [B, N_focal, C, H, W]
        target_focal_idx: [B] index of best-focus plane per sample (or None → same plane)

        Returns: (pred_patches, target_patches, mask)
        """
        if not self.is_vit:
            raise RuntimeError("MAE requires ViT backbone")

        B, N, C, H, W = focal_imgs.shape
        # Use first focal plane as input by default
        input_img = focal_imgs[:, 0]  # [B, C, H, W]

        # Get patch tokens from encoder
        _, patch_tokens = self.encoder(input_img, return_tokens=True)  # [B, N_p, D]
        patch_tokens_masked, mask, ids_restore = random_masking(patch_tokens, self.mask_ratio)

        # Prepend cls token (just zeros for decoder input)
        cls = torch.zeros(B, 1, patch_tokens_masked.shape[-1], device=focal_imgs.device)
        visible_with_cls = torch.cat([cls, patch_tokens_masked], dim=1)

        pred_patches = self.mae_decoder(visible_with_cls, ids_restore)  # [B, N_p, patch_dim]

        # Build reconstruction target
        if target_focal_idx is not None:
            # Best-focus plane per sample
            target_imgs = focal_imgs[
                torch.arange(B), target_focal_idx
            ]  # [B, C, H, W]
        else:
            target_imgs = input_img

        # Patchify target
        p = self.encoder.patch_embed.patch_size
        target_patches = self._patchify(target_imgs, p)  # [B, N_p, p*p*C]

        return pred_patches, target_patches, mask

    @staticmethod
    def _patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
        """imgs: [B, C, H, W] → [B, N_patches, patch_size*patch_size*C]"""
        B, C, H, W = imgs.shape
        p = patch_size
        assert H % p == 0 and W % p == 0
        h, w = H // p, W // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, p * p * C)
        return x

    def forward_temporal(
        self, view_t1: torch.Tensor, view_t2: torch.Tensor
    ) -> torch.Tensor:
        """
        Temporal gap prediction.
        Returns: [B, num_bins] logits
        """
        _, fused1 = self.encode_focal(view_t1)
        _, fused2 = self.encode_focal(view_t2)
        return self.temporal_gap_head(fused1, fused2)


# =============================================================================
# Full Fine-tuning Model
# =============================================================================

class EmbryoFinetuneModel(nn.Module):
    """
    Full model for supervised fine-tuning.

    Pipeline:
        [N_focal images] → Encoder (per-plane) → FocalFusion
                        → TemporalGRU (optional, over sequence)
                        → SoftMoEClassifier → 16-class output
                        + Auxiliary heads
    """

    def __init__(self, cfg: dict):
        super().__init__()
        model_cfg = cfg["model"]
        ft_cfg = cfg["finetune"]

        # Shared encoder (will be loaded from pretrain checkpoint)
        self.encoder = build_encoder(cfg)
        encoder_out_dim = self.encoder.out_dim

        # Multi-focal fusion
        fc = model_cfg["fusion"]
        self.fusion = FocalFusionModule(
            embed_dim=encoder_out_dim,
            num_focal=cfg["data"]["num_focal_planes"],
            method=fc["method"],
            num_heads=fc["num_heads"],
            dropout=fc["dropout"],
        )

        # Optional temporal GRU
        cls_cfg = model_cfg["classifier"]
        self.use_temporal = cls_cfg.get("use_temporal_gru", True)
        if self.use_temporal:
            tc = model_cfg["temporal"]
            self.temporal_head = TemporalHead(
                input_dim=encoder_out_dim,
                hidden_dim=tc["hidden_dim"],
                num_layers=tc["num_layers"],
                temporal_type=tc["type"],
                dropout=tc["dropout"],
                transformer_heads=tc.get("transformer_heads", 8),
                transformer_depth=tc.get("transformer_depth", 4),
            )
            classifier_in_dim = self.temporal_head.out_dim
        else:
            self.temporal_head = None
            classifier_in_dim = encoder_out_dim

        # Main classifier (MoE)
        num_classes = cfg["data"]["num_stages"]
        if cls_cfg.get("use_moe", True):
            self.classifier = SoftMoEClassifier(
                in_dim=classifier_in_dim,
                num_classes=num_classes,
                num_experts=cls_cfg["num_experts"],
                router_hidden=cls_cfg["router_hidden_dim"],
                head_hidden=cls_cfg["head_hidden_dim"],
                dropout=cls_cfg["dropout"],
            )
        else:
            self.classifier = ExpertHead(
                in_dim=classifier_in_dim,
                num_classes=num_classes,
                hidden_dim=cls_cfg["head_hidden_dim"],
                dropout=cls_cfg["dropout"],
            )
            self.use_moe = False

        self.use_moe = cls_cfg.get("use_moe", True)

        # Auxiliary heads
        stage_groups = cfg["data"]["stage_groups"]
        self.cell_count_head = CellCountHead(in_dim=classifier_in_dim, num_counts=4)
        self.blastocyst_head = BlastocystScoreHead(in_dim=classifier_in_dim)
        self.early_stage_ids = [s - 1 for s in stage_groups["early"]]
        self.late_stage_ids = [s - 1 for s in stage_groups["late"]]

    def encode_focal(self, focal_imgs: torch.Tensor) -> torch.Tensor:
        """
        focal_imgs: [B, N_focal, C, H, W]
        Returns: [B, D]
        """
        B, N, C, H, W = focal_imgs.shape
        flat = focal_imgs.view(B * N, C, H, W)
        embeds = self.encoder(flat).view(B, N, -1)
        return self.fusion(embeds)

    def forward(
        self,
        focal_imgs: Optional[torch.Tensor] = None,
        focal_seq: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Two input modes:
          - Single frame:   focal_imgs [B, N_focal, C, H, W]
          - Sequence:       focal_seq  [B, T, N_focal, C, H, W]

        Returns dict with:
          logits:         [B, num_classes] or [B, T, num_classes]
          router_weights: [B, num_experts] (if MoE) or None
          cell_count_logits: [B, 4]
          blastocyst_score:  [B]
        """
        if focal_seq is not None:
            return self._forward_sequence(focal_seq)
        else:
            return self._forward_single(focal_imgs)

    def _forward_single(self, focal_imgs: torch.Tensor) -> dict:
        """Forward pass for single-frame input."""
        fused = self.encode_focal(focal_imgs)  # [B, D]

        if self.use_temporal and self.temporal_head is not None:
            # Treat single frame as sequence of length 1
            fused = self.temporal_head(fused.unsqueeze(1)).squeeze(1)  # [B, D]

        if self.use_moe:
            logits, router_weights = self.classifier(fused)
        else:
            logits = self.classifier(fused)
            router_weights = None

        cell_count = self.cell_count_head(fused)
        blast_score = self.blastocyst_head(fused)

        return {
            "logits": logits,
            "router_weights": router_weights,
            "cell_count_logits": cell_count,
            "blastocyst_score": blast_score,
            "embeddings": fused,
        }

    def _forward_sequence(self, focal_seq: torch.Tensor) -> dict:
        """Forward pass for temporal sequence input."""
        B, T, N, C, H, W = focal_seq.shape

        # Encode each frame
        fused_seq = []
        for t in range(T):
            fused_seq.append(self.encode_focal(focal_seq[:, t]))  # [B, D]
        fused_seq = torch.stack(fused_seq, dim=1)  # [B, T, D]

        if self.use_temporal and self.temporal_head is not None:
            fused_seq = self.temporal_head(fused_seq)  # [B, T, out_dim]

        # Classify each timestep
        B_T = B * T
        fused_flat = fused_seq.view(B_T, -1)

        if self.use_moe:
            logits, router_weights = self.classifier(fused_flat)
            logits = logits.view(B, T, -1)
            router_weights = router_weights.view(B, T, -1)
        else:
            logits = self.classifier(fused_flat).view(B, T, -1)
            router_weights = None

        cell_count = self.cell_count_head(fused_flat).view(B, T, -1)
        blast_score = self.blastocyst_head(fused_flat).view(B, T)

        return {
            "logits": logits,
            "router_weights": router_weights,
            "cell_count_logits": cell_count,
            "blastocyst_score": blast_score,
            "embeddings": fused_seq,
        }

    def get_encoder_parameter_groups(
        self, base_lr: float, encoder_lr_scale: float, layer_lr_decay: float
    ) -> List[Dict]:
        """
        Return parameter groups with layer-wise LR decay for optimizer.
        Encoder layers get lower LR; heads get full LR.
        """
        head_params = []
        encoder_params_by_depth = {}

        for name, param in self.named_parameters():
            if name.startswith("encoder."):
                # Extract depth from name (heuristic)
                parts = name.split(".")
                if "blocks" in parts:
                    depth = int(parts[parts.index("blocks") + 1])
                else:
                    depth = 0
                if depth not in encoder_params_by_depth:
                    encoder_params_by_depth[depth] = []
                encoder_params_by_depth[depth].append(param)
            else:
                head_params.append(param)

        param_groups = [{"params": head_params, "lr": base_lr}]

        max_depth = max(encoder_params_by_depth.keys()) if encoder_params_by_depth else 0
        for depth, params in encoder_params_by_depth.items():
            decay = layer_lr_decay ** (max_depth - depth)
            lr = base_lr * encoder_lr_scale * decay
            param_groups.append({"params": params, "lr": lr})

        return param_groups

    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder_last_n_blocks(self, n: int):
        """Unfreeze last N transformer/residual blocks of the encoder."""
        if hasattr(self.encoder, "blocks"):
            blocks = self.encoder.blocks
            for blk in blocks[-n:]:
                for param in blk.parameters():
                    param.requires_grad = True
        elif hasattr(self.encoder, "features"):
            # ResNet: unfreeze last N children
            children = list(self.encoder.features.children())
            for child in children[-n:]:
                for param in child.parameters():
                    param.requires_grad = True

    def unfreeze_encoder(self):
        """Unfreeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True


def load_pretrained_encoder(model: EmbryoFinetuneModel, checkpoint_path: str,
                             strict: bool = False) -> None:
    """Load encoder weights from a pretraining checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

    # Filter for encoder keys only
    encoder_state = {
        k.replace("encoder.", ""): v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }
    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=strict)
    print(f"[load_pretrained_encoder] Loaded from {checkpoint_path}")
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
