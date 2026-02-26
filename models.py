"""
models.py – Model factory for transfer learning on Caltech-256.

Provides a unified interface to load pretrained CNNs and Vision Transformers,
replace the classification head for 257 classes, and optionally freeze the
backbone for feature-extraction mode.

Supported models
----------------
CNNs:       resnet50, resnet101, vgg16, efficientnet_b0, mobilenet_v2, inception_v3
Transformers: vit_b_16, vit_l_16, swin_t, deit_base_patch16_224
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models

try:
    import timm
    _TIMM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TIMM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_CLASSES = 257  # 256 Caltech categories + 1 clutter class

# Model display names used in comparison tables / plots
MODEL_NAMES: dict[str, str] = {
    "resnet50": "ResNet-50",
    "resnet101": "ResNet-101",
    "vgg16": "VGG-16",
    "efficientnet_b0": "EfficientNet-B0",
    "mobilenet_v2": "MobileNet-V2",
    "inception_v3": "InceptionV3",
    "vit_b_16": "ViT-B/16",
    "vit_l_16": "ViT-L/16",
    "swin_t": "Swin-T",
    "deit_base_patch16_224": "DeiT-Base",
}

# Expected input sizes (height/width)
INPUT_SIZES: dict[str, int] = {
    "resnet50": 224,
    "resnet101": 224,
    "vgg16": 224,
    "efficientnet_b0": 224,
    "mobilenet_v2": 224,
    "inception_v3": 299,
    "vit_b_16": 224,
    "vit_l_16": 224,
    "swin_t": 224,
    "deit_base_patch16_224": 224,
}

# ImageNet normalisation statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model: nn.Module) -> int:
    """Return the total number of parameters (trainable + frozen)."""
    return sum(p.numel() for p in model.parameters())


def freeze_backbone(model: nn.Module, model_key: str) -> None:
    """
    Freeze all parameters except the final classification head.

    The classification head varies by architecture and is *not* frozen so
    that it can be trained from scratch on the target dataset.
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the head
    head = _get_head(model, model_key)
    if head is not None:
        for param in head.parameters():
            param.requires_grad = True


def unfreeze_last_n_layers(model: nn.Module, model_key: str, n: int) -> None:
    """
    Unfreeze the last *n* layers/blocks of the backbone plus the head.

    Useful for gradual fine-tuning after initial feature-extraction training.
    If *n* == 0, only the head remains trainable.
    If *n* < 0 or very large, all parameters are unfrozen.
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Build a flat ordered list of all named children
    children = list(model.named_children())
    if n > 0:
        for _, layer in children[-n:]:
            for param in layer.parameters():
                param.requires_grad = True
    elif n < 0:
        for param in model.parameters():
            param.requires_grad = True

    # Always unfreeze the head
    head = _get_head(model, model_key)
    if head is not None:
        for param in head.parameters():
            param.requires_grad = True


def _get_head(model: nn.Module, model_key: str) -> nn.Module | None:
    """Return the classification head module for the given architecture."""
    if model_key in ("resnet50", "resnet101"):
        return model.fc
    if model_key == "vgg16":
        return model.classifier
    if model_key == "efficientnet_b0":
        return model.classifier
    if model_key == "mobilenet_v2":
        return model.classifier
    if model_key == "inception_v3":
        return model.fc
    if model_key in ("vit_b_16", "vit_l_16"):
        return model.heads
    if model_key == "swin_t":
        return model.head
    if model_key == "deit_base_patch16_224" and _TIMM_AVAILABLE:
        return model.head
    return None  # pragma: no cover


# ---------------------------------------------------------------------------
# Individual model builders
# ---------------------------------------------------------------------------


def _build_resnet50(pretrained: bool = True) -> nn.Module:
    weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = tv_models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def _build_resnet101(pretrained: bool = True) -> nn.Module:
    weights = tv_models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
    model = tv_models.resnet101(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def _build_vgg16(pretrained: bool = True) -> nn.Module:
    weights = tv_models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_models.vgg16(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    return model


def _build_efficientnet_b0(pretrained: bool = True) -> nn.Module:
    weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_models.efficientnet_b0(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    return model


def _build_mobilenet_v2(pretrained: bool = True) -> nn.Module:
    weights = tv_models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    model = tv_models.mobilenet_v2(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    return model


def _build_inception_v3(pretrained: bool = True) -> nn.Module:
    weights = tv_models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_models.inception_v3(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    # Auxiliary classifier
    if model.AuxLogits is not None:
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, NUM_CLASSES)
    return model


def _build_vit_b_16(pretrained: bool = True) -> nn.Module:
    weights = tv_models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_models.vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, NUM_CLASSES)
    return model


def _build_vit_l_16(pretrained: bool = True) -> nn.Module:
    weights = tv_models.ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_models.vit_l_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, NUM_CLASSES)
    return model


def _build_swin_t(pretrained: bool = True) -> nn.Module:
    weights = tv_models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_models.swin_t(weights=weights)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, NUM_CLASSES)
    return model


def _build_deit(pretrained: bool = True) -> nn.Module:
    if not _TIMM_AVAILABLE:
        raise ImportError(
            "The 'timm' library is required for DeiT. Install it with: pip install timm"
        )
    model = timm.create_model(
        "deit_base_patch16_224",
        pretrained=pretrained,
        num_classes=NUM_CLASSES,
    )
    return model


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------

_BUILDERS = {
    "resnet50": _build_resnet50,
    "resnet101": _build_resnet101,
    "vgg16": _build_vgg16,
    "efficientnet_b0": _build_efficientnet_b0,
    "mobilenet_v2": _build_mobilenet_v2,
    "inception_v3": _build_inception_v3,
    "vit_b_16": _build_vit_b_16,
    "vit_l_16": _build_vit_l_16,
    "swin_t": _build_swin_t,
    "deit_base_patch16_224": _build_deit,
}


def get_model(
    model_key: str,
    pretrained: bool = True,
    freeze: bool = True,
    device: torch.device | None = None,
) -> nn.Module:
    """
    Load a pretrained model, replace its classification head for Caltech-256,
    and optionally freeze the backbone.

    Parameters
    ----------
    model_key : str
        One of the keys in ``MODEL_NAMES``.
    pretrained : bool
        Whether to load ImageNet pretrained weights.
    freeze : bool
        If *True* (default), freeze backbone and keep only the head trainable.
        Set to *False* to fine-tune all parameters from the start.
    device : torch.device, optional
        Device to move the model to. Defaults to CUDA if available, else CPU.

    Returns
    -------
    nn.Module
        Ready-to-use model on the requested device.
    """
    if model_key not in _BUILDERS:
        raise ValueError(
            f"Unknown model '{model_key}'. "
            f"Valid options: {list(_BUILDERS.keys())}"
        )

    model = _BUILDERS[model_key](pretrained=pretrained)

    if freeze:
        freeze_backbone(model, model_key)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return model.to(device)


def list_models() -> list[str]:
    """Return a sorted list of all supported model keys."""
    return sorted(_BUILDERS.keys())
