"""
Student architecture builders for MASt3R knowledge distillation.

Each builder returns an AsymmetricMASt3R instance whose encoder has been
replaced by a lightweight backbone (MobileNetV3, ViT-Small/DINOv2, ViT-Tiny,
or DINOv3).  The decoder and heads are standard MASt3R components, so any
MASt3R loss function can be applied directly.

To define a custom student, follow the pattern of any builder here:
  1. Call _build_mast3r_shell() to get the decoder + head scaffold.
  2. Attach your backbone as student.tiny_enc or student.tiny_vit.
  3. Bind a custom _encode_image method that returns (tokens, pos, None).
"""

import types
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from mast3r.model import AsymmetricMASt3R


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _build_mast3r_shell(device, dec_embed_dim: int, dec_depth: int,
                        dec_num_heads: int, mlp_ratio: float) -> AsymmetricMASt3R:
    """Build an AsymmetricMASt3R model that acts as decoder + head scaffold.

    The enc_depth=0 setting disables the built-in encoder so that a custom
    lightweight backbone can be substituted via _encode_image.
    """
    return AsymmetricMASt3R(
        pos_embed='RoPE100',
        patch_embed_cls='PatchEmbedDust3R',
        img_size=(512, 512),
        head_type='catmlp+dpt',
        output_mode='pts3d+desc24',
        depth_mode=('exp', -float('inf'), float('inf')),
        conf_mode=('exp', 1, float('inf')),
        enc_embed_dim=dec_embed_dim,
        enc_depth=0,
        enc_num_heads=dec_num_heads,
        dec_embed_dim=dec_embed_dim,
        dec_depth=dec_depth,
        dec_num_heads=dec_num_heads,
        mlp_ratio=mlp_ratio,
        two_confs=True,
        desc_conf_mode=('exp', 0, float('inf')),
        freeze='none',
        landscape_only=False,
    ).to(device)


def _ceil_div(a, b):
    return (a + b - 1) // b


def _get_patch_size(model) -> int:
    """Infer patch size from a model's patch_embed attribute."""
    if hasattr(model, 'patch_embed'):
        pe = model.patch_embed
        if hasattr(pe, 'patch_size'):
            v = pe.patch_size
            return int(v[0] if isinstance(v, (tuple, list)) else v)
        if hasattr(pe, 'proj') and hasattr(pe.proj, 'stride'):
            return int(pe.proj.stride[0])
    return 16


def _make_encode_image(patch_size: int, feat_attr: str = 'tiny_enc'):
    """Create an _encode_image method for a backbone that outputs [B, D, H, W] feature maps.

    Args:
        patch_size: spatial stride of the backbone's feature map.
        feat_attr: name of the attribute on the student that holds the backbone.
    """
    def _encode_image(self, image: torch.Tensor, true_shape=None):
        feat = getattr(self, feat_attr)(image)  # [B, D, Hf, Wf]
        ps = patch_size
        if true_shape is None:
            H_in = torch.tensor([image.shape[-2]], device=feat.device)
            W_in = torch.tensor([image.shape[-1]], device=feat.device)
        else:
            ts = true_shape if torch.is_tensor(true_shape) else \
                torch.tensor(true_shape, device=feat.device)
            H_in, W_in = (ts[0].reshape(1), ts[1].reshape(1)) if ts.ndim == 1 \
                else (ts[..., 0], ts[..., 1])
        Ht = int(_ceil_div(H_in, ps).max().item())
        Wt = int(_ceil_div(W_in, ps).max().item())
        if feat.shape[-2:] != (Ht, Wt):
            feat = F.interpolate(feat, size=(Ht, Wt), mode='bilinear', align_corners=False)
        B, D, H, W = feat.shape
        tokens = feat.view(B, D, H * W).transpose(1, 2).contiguous()
        ys = torch.arange(H, device=feat.device, dtype=torch.long)
        xs = torch.arange(W, device=feat.device, dtype=torch.long)
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        pos = torch.stack([gx, gy], dim=-1).view(1, H * W, 2).expand(B, -1, -1).contiguous()
        return tokens, pos, None
    return _encode_image


# ---------------------------------------------------------------------------
# MobileNetV3 backbone
# ---------------------------------------------------------------------------

class MobileNetFeatureMap(nn.Module):
    """MobileNetV3-Large feature extractor at stride 16.

    Args:
        out_dim_target: output channel dimension (should match dec_embed_dim).
        name: timm model name.
        pretrained: load ImageNet-pretrained weights.
    """

    def __init__(self, out_dim_target: int = 512, name: str = 'mobilenetv3_large_100',
                 pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            name, pretrained=pretrained, features_only=True, out_indices=(2, 3, 4))
        for mod in self.backbone.modules():
            if hasattr(mod, 'inplace'):
                mod.inplace = False  # prevent AMP + skip-connection version errors
        c_in = self.backbone.feature_info.channels()[1]  # stride-16 channels
        self.adapter = nn.Sequential(
            nn.Conv2d(c_in, out_dim_target, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_dim_target),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, f3, _ = self.backbone(x)  # f3 = stride-16 feature map
        feat = self.adapter(f3)
        # Cache for optional feature alignment loss
        if hasattr(self, '_parent_ref') and self._parent_ref is not None:
            parent = self._parent_ref()
            if parent is not None:
                parent.last_feat_encoder = feat
        return feat


def build_mobilenet_student(device: str = 'cuda', dec_embed_dim: int = 512,
                             dec_depth: int = 6, dec_num_heads: int = 4,
                             mlp_ratio: float = 1.0,
                             backbone_name: str = 'mobilenetv3_large_100',
                             pretrained_backbone: bool = True) -> AsymmetricMASt3R:
    """Build a student with MobileNetV3-Large encoder (stride-16, ~5.4 M params).

    Args:
        device: target device.
        dec_embed_dim: decoder hidden dimension.
        dec_depth: number of decoder cross-attention blocks.
        dec_num_heads: number of attention heads in decoder.
        mlp_ratio: MLP expansion ratio in decoder blocks.
        backbone_name: timm model name for MobileNet.
        pretrained_backbone: load ImageNet-pretrained weights.

    Returns:
        AsymmetricMASt3R student model.
    """
    student = _build_mast3r_shell(device, dec_embed_dim, dec_depth, dec_num_heads, mlp_ratio)
    student.tiny_enc = MobileNetFeatureMap(
        out_dim_target=dec_embed_dim, name=backbone_name,
        pretrained=pretrained_backbone).to(device)
    student.tiny_enc._parent_ref = weakref.ref(student)
    student._encode_image = types.MethodType(
        _make_encode_image(patch_size=16, feat_attr='tiny_enc'), student)
    return student


# ---------------------------------------------------------------------------
# ViT-Small / DINOv2 backbone
# ---------------------------------------------------------------------------

def build_vit_student(device: str = 'cuda', dec_embed_dim: int = 512,
                      dec_depth: int = 6, dec_num_heads: int = 4,
                      mlp_ratio: float = 1.0, prefer_dinov2: bool = True,
                      freeze_backbone: bool = False,
                      backbone_type: str = 'dinov2',
                      pretrained_backbone: bool = True) -> AsymmetricMASt3R:
    """Build a student with ViT-Small / DINOv2 encoder (~22 M params).

    Args:
        device: target device.
        dec_embed_dim: decoder hidden dimension.
        dec_depth: number of decoder cross-attention blocks.
        dec_num_heads: number of attention heads in decoder.
        mlp_ratio: MLP expansion ratio.
        prefer_dinov2: prefer DINOv2 weights over supervised ViT-S.
        freeze_backbone: freeze ViT weights; only adapter + decoder train.
        backbone_type: 'dinov2' or 'dune'.
        pretrained_backbone: load pretrained weights.

    Returns:
        AsymmetricMASt3R student model.
    """
    from mast3r.tiny_mast3r_model import (
        ViTFeatureMap,
        _create_tiny_encoder_method as _create_vit_encoder_method,
    )
    student = _build_mast3r_shell(device, dec_embed_dim, dec_depth, dec_num_heads, mlp_ratio)
    student.tiny_vit = ViTFeatureMap(
        out_dim_target=dec_embed_dim,
        prefer_dinov2=prefer_dinov2,
        backbone_type=backbone_type,
        pretrained_backbone=pretrained_backbone,
    ).to(device)
    patch_size = student.tiny_vit.patch
    student._encode_image = types.MethodType(
        _create_vit_encoder_method(patch_size), student)

    if freeze_backbone:
        for p in student.tiny_vit.vit.parameters():
            p.requires_grad_(False)
        n_frozen = sum(p.numel() for p in student.tiny_vit.vit.parameters())
        print(f"[ViT student] Froze backbone ({n_frozen:,} params). "
              "Only adapter + decoder + heads are trainable.")
    return student


# ---------------------------------------------------------------------------
# ViT-Tiny backbone (timm)
# ---------------------------------------------------------------------------

class ViTTinyFeatureMap(nn.Module):
    """ViT-Tiny/patch16 feature extractor (~5.7 M params, embed_dim=192).

    Args:
        out_dim_target: output channel dimension.
        model_name: timm model name.
        freeze_backbone: freeze ViT weights.
        img_size: input image resolution (used to pre-configure patch_embed).
    """

    def __init__(self, out_dim_target: int = 512,
                 model_name: str = 'vit_tiny_patch16_224',
                 freeze_backbone: bool = False, img_size: int = 512):
        super().__init__()
        self.patch = 16
        self.vit = timm.create_model(
            model_name, pretrained=True, num_classes=0,
            img_size=img_size, dynamic_img_size=True)
        if hasattr(self.vit, 'patch_embed'):
            self.vit.patch_embed.strict_img_size = False
        self.emb_dim = self.vit.embed_dim  # 192

        self.adapter = None
        if self.emb_dim != out_dim_target:
            self.adapter = nn.Sequential(
                nn.Conv2d(self.emb_dim, out_dim_target, kernel_size=1, bias=False),
                nn.GroupNorm(1, out_dim_target),
            )

        if freeze_backbone:
            for p in self.vit.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        ph = (self.patch - H % self.patch) % self.patch
        pw = (self.patch - W % self.patch) % self.patch
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode='replicate')
        _, _, Hp, Wp = x.shape
        Ht, Wt = Hp // self.patch, Wp // self.patch

        if hasattr(self.vit, 'patch_embed') and \
                getattr(self.vit.patch_embed, 'strict_img_size', False):
            self.vit.patch_embed.strict_img_size = False

        tokens = self.vit.forward_features(x)  # [B, 1+N, C]
        if tokens.shape[1] == Ht * Wt + 1:
            tokens = tokens[:, 1:]  # drop CLS
        feat = tokens.transpose(1, 2).contiguous().view(B, self.emb_dim, Ht, Wt)
        if self.adapter is not None:
            feat = self.adapter(feat)
        return feat


def build_vit_tiny_student(device: str = 'cuda', dec_embed_dim: int = 512,
                            dec_depth: int = 6, dec_num_heads: int = 4,
                            mlp_ratio: float = 1.0,
                            model_name: str = 'vit_tiny_patch16_224',
                            freeze_backbone: bool = False,
                            img_size: int = 512) -> AsymmetricMASt3R:
    """Build a student with ViT-Tiny encoder (~5.7 M params, embed_dim=192).

    Args:
        device: target device.
        dec_embed_dim: decoder hidden dimension.
        dec_depth: number of decoder cross-attention blocks.
        dec_num_heads: number of attention heads.
        mlp_ratio: MLP expansion ratio.
        model_name: timm model name.
        freeze_backbone: freeze ViT-Tiny weights.
        img_size: input image resolution.

    Returns:
        AsymmetricMASt3R student model.
    """
    from mast3r.tiny_mast3r_model import _create_tiny_encoder_method as _create_vit_encoder_method
    student = _build_mast3r_shell(device, dec_embed_dim, dec_depth, dec_num_heads, mlp_ratio)
    student.tiny_vit = ViTTinyFeatureMap(
        out_dim_target=dec_embed_dim, model_name=model_name,
        freeze_backbone=freeze_backbone, img_size=img_size,
    ).to(device)
    student._encode_image = types.MethodType(
        _create_vit_encoder_method(student.tiny_vit.patch), student)
    return student


# ---------------------------------------------------------------------------
# DINOv3 backbone (HuggingFace transformers)
# ---------------------------------------------------------------------------

class DINOv3FeatureMap(nn.Module):
    """DINOv3 feature extractor (ConvNeXt or ViT) from HuggingFace.

    Supported model names:
        ConvNeXt: facebook/dinov3-convnext-{tiny,small,base,large}-pretrain-lvd1689m
        ViT:      facebook/dinov3-vit{s,splus,b,l,hplus}16-pretrain-lvd1689m

    Args:
        out_dim_target: output channel dimension.
        model_name: HuggingFace model identifier.
        freeze_backbone: freeze backbone weights.
    """

    _CONVNEXT_DIMS = {'tiny': 768, 'small': 768, 'base': 1024, 'large': 1536}
    _VIT_DIMS = {'vits': 384, 'vitsplus': 384, 'vitb': 768, 'vitl': 1024, 'vithplus': 1280}

    def __init__(self, out_dim_target: int = 512,
                 model_name: str = 'facebook/dinov3-convnext-small-pretrain-lvd1689m',
                 freeze_backbone: bool = True):
        super().__init__()
        from transformers import AutoModel

        self.model_name = model_name
        self.is_convnext = 'convnext' in model_name.lower()
        self.patch = 16

        if self.is_convnext:
            self.emb_dim = next(
                (d for k, d in self._CONVNEXT_DIMS.items() if k in model_name.lower()), 768)
        else:
            self.emb_dim = next(
                (d for k, d in self._VIT_DIMS.items() if k in model_name.lower()), 384)

        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        self.adapter = None
        if self.emb_dim != out_dim_target:
            self.adapter = nn.Sequential(
                nn.Conv2d(self.emb_dim, out_dim_target, kernel_size=1, bias=False),
                nn.GroupNorm(1, out_dim_target),
            )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        outputs = self.backbone(pixel_values=x, output_hidden_states=True)

        if self.is_convnext:
            feat = outputs.hidden_states[-1]
            target_H, target_W = H // self.patch, W // self.patch
            if feat.shape[-2:] != (target_H, target_W):
                feat = F.interpolate(feat, size=(target_H, target_W),
                                     mode='bilinear', align_corners=False)
        else:
            tokens = outputs.last_hidden_state  # [B, 1+reg+N, C]
            Ht, Wt = H // self.patch, W // self.patch
            patch_tokens = tokens[:, 5:5 + Ht * Wt, :]  # skip CLS + 4 registers
            feat = patch_tokens.transpose(1, 2).contiguous().view(B, self.emb_dim, Ht, Wt)

        if self.adapter is not None:
            feat = self.adapter(feat)
        return feat


def build_dinov3_student(device: str = 'cuda', dec_embed_dim: int = 512,
                          dec_depth: int = 6, dec_num_heads: int = 4,
                          mlp_ratio: float = 1.0,
                          model_name: str = 'facebook/dinov3-convnext-small-pretrain-lvd1689m',
                          freeze_backbone: bool = True) -> AsymmetricMASt3R:
    """Build a student with a DINOv3 backbone (ConvNeXt or ViT).

    Args:
        device: target device.
        dec_embed_dim: decoder hidden dimension.
        dec_depth: number of decoder cross-attention blocks.
        dec_num_heads: number of attention heads.
        mlp_ratio: MLP expansion ratio.
        model_name: HuggingFace DINOv3 model identifier.
        freeze_backbone: freeze backbone weights (adapter still trains).

    Returns:
        AsymmetricMASt3R student model.
    """
    student = _build_mast3r_shell(device, dec_embed_dim, dec_depth, dec_num_heads, mlp_ratio)
    student.tiny_vit = DINOv3FeatureMap(
        out_dim_target=dec_embed_dim, model_name=model_name,
        freeze_backbone=freeze_backbone,
    ).to(device)
    student._encode_image = types.MethodType(
        _make_encode_image(patch_size=student.tiny_vit.patch, feat_attr='tiny_vit'), student)
    return student
