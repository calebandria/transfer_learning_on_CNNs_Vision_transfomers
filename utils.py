"""
utils.py - Utility functions for the transfer-learning comparison framework.

Covers:
  - Data loading and augmentation for Caltech-256
  - Training loop with early stopping & checkpointing
  - Evaluation (top-1 / top-5 accuracy, inference time, per-class metrics)
  - Visualization helpers (loss/accuracy curves, comparison plots, confusion matrix)
  - CSV summary export
"""

from __future__ import annotations

import os
import time
import copy
import random
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; switch to "TkAgg" if needed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

# ImageNet statistics (used for all models)
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def get_transforms(
    input_size: int = 224,
    augment: bool = True,
) -> dict[str, transforms.Compose]:
    """
    Build torchvision transform pipelines.

    Parameters
    ----------
    input_size : int
        Spatial size expected by the model (224 for most; 299 for InceptionV3).
    augment : bool
        Whether to apply training augmentation.

    Returns
    -------
    dict with keys ``"train"``, ``"val"``, and ``"test"``.
    """
    train_tf: list = []
    if augment:
        train_tf = [
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
        ]
    else:
        train_tf = [
            transforms.Resize(int(input_size * 1.14)),
            transforms.CenterCrop(input_size),
        ]

    eval_tf = [
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
    ]

    common = [
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]

    return {
        "train": transforms.Compose(train_tf + common),
        "val": transforms.Compose(eval_tf + common),
        "test": transforms.Compose(eval_tf + common),
    }


def load_caltech256(
    data_dir: str | Path,
    input_size: int = 224,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    augment: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """
    Load Caltech-256 from *data_dir* and return train / val / test DataLoaders.

    Caltech-256 should be organised as::

        data_dir/
            001.ak47/
            002.american-flag/
            ...
            257.clutter/

    Parameters
    ----------
    data_dir : str or Path
        Root directory of the dataset (parent of the class subdirectories).
    input_size : int
        Spatial size for resizing / cropping.
    train_ratio, val_ratio : float
        Fractions for train and validation splits; test gets the remainder.
    batch_size : int
        Mini-batch size for all loaders.
    num_workers : int
        Number of DataLoader worker processes.
    seed : int
        Random seed for reproducible splits.
    augment : bool
        Whether to apply data augmentation on the training split.

    Returns
    -------
    (train_loader, val_loader, test_loader, class_names)
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            "Please update DATA_DIR in the configuration cell."
        )

    tf = get_transforms(input_size=input_size, augment=augment)

    # Load full dataset with train transform to get class names; we'll swap
    # transforms per split via Subset + a wrapper approach below.
    full_dataset = datasets.ImageFolder(root=str(data_dir))
    class_names = full_dataset.classes
    n = len(full_dataset)

    # Reproducible split indices
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n).tolist()

    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    logger.info(
        "Dataset split – train: %d  val: %d  test: %d  total: %d",
        len(train_idx), len(val_idx), len(test_idx), n,
    )

    # Create per-split datasets with appropriate transforms
    train_ds = _TransformedSubset(full_dataset, train_idx, tf["train"])
    val_ds = _TransformedSubset(full_dataset, val_idx, tf["val"])
    test_ds = _TransformedSubset(full_dataset, test_idx, tf["test"])

    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    return train_loader, val_loader, test_loader, class_names


class _TransformedSubset(torch.utils.data.Dataset):
    """A subset of an ImageFolder dataset with a custom transform."""

    def __init__(
        self,
        base_dataset: datasets.ImageFolder,
        indices: list[int],
        transform: Callable,
    ) -> None:
        self.base = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        path, label = self.base.samples[self.indices[idx]]
        img = self.base.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 7, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: float = float("inf")
        self.should_stop: bool = False

    def step(self, val_loss: float) -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    model_key: str,
    num_epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 7,
    checkpoint_dir: str | Path = "checkpoints",
    device: torch.device | None = None,
    resume_checkpoint: str | Path | None = None,
) -> dict:
    """
    Train *model* with the given loaders and return a history dictionary.

    Returns
    -------
    dict with keys:
        ``train_loss``, ``val_loss``, ``train_acc``, ``val_acc``,
        ``epoch_times``, ``best_val_acc``, ``best_epoch``,
        ``total_time``, ``checkpoint_path``.
    """
    if device is None:
        device = next(model.parameters()).device

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"{model_key}_best.pth"

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True,
    )
    early_stop = EarlyStopping(patience=patience)

    start_epoch = 0
    best_val_acc = 0.0
    best_epoch = 1
    best_weights = copy.deepcopy(model.state_dict())

    # Optionally resume from checkpoint
    if resume_checkpoint is not None and Path(resume_checkpoint).exists():
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        logger.info("Resumed from %s (epoch %d)", resume_checkpoint, start_epoch)

    history: dict = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "epoch_times": [],
    }

    is_inception = (model_key == "inception_v3")
    total_start = time.time()

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        # ---- Training phase ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"[{model_key}] Epoch {epoch+1}/{num_epochs} train", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            if is_inception:
                outputs, aux_outputs = model(inputs)
                loss = criterion(outputs, labels) + 0.4 * criterion(aux_outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # ---- Validation phase ----
        val_loss, val_acc = evaluate_loss_acc(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["epoch_times"].append(epoch_time)

        logger.info(
            "[%s] Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  "
            "train_acc=%.4f  val_acc=%.4f  time=%.1fs",
            model_key, epoch + 1, num_epochs,
            train_loss, val_loss, train_acc, val_acc, epoch_time,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": best_weights,
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                },
                ckpt_path,
            )
            logger.info("  ✓ New best val_acc=%.4f – checkpoint saved.", best_val_acc)

        early_stop.step(val_loss)
        if early_stop.should_stop:
            logger.info("Early stopping triggered at epoch %d.", epoch + 1)
            break

    total_time = time.time() - total_start
    model.load_state_dict(best_weights)

    history["best_val_acc"] = best_val_acc
    history["best_epoch"] = best_epoch
    history["total_time"] = total_time
    history["checkpoint_path"] = str(ckpt_path)

    return history


def evaluate_loss_acc(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Return (loss, top-1 accuracy) on *loader* without computing gradients."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def top_k_accuracy(outputs: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    """Compute top-k accuracy for a batch."""
    with torch.no_grad():
        _, top_k = outputs.topk(k, dim=1)
        correct = top_k.eq(labels.view(-1, 1).expand_as(top_k))
        return correct.any(dim=1).float().mean().item()


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: list[str],
    device: torch.device | None = None,
) -> dict:
    """
    Full evaluation on the test set.

    Returns
    -------
    dict with keys:
        ``top1_acc``, ``top5_acc``, ``avg_inference_ms``,
        ``all_preds``, ``all_labels``, ``report``, ``confusion_matrix``.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    top1_correct = 0
    top5_correct = 0
    total = 0
    inference_times: list[float] = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            t0 = time.perf_counter()
            outputs = model(inputs)
            t1 = time.perf_counter()

            batch_ms = (t1 - t0) * 1000 / inputs.size(0)
            inference_times.append(batch_ms)

            _, preds = torch.max(outputs, 1)
            top1_correct += (preds == labels).sum().item()
            top5_correct += int(
                top_k_accuracy(outputs, labels, k=min(5, outputs.size(1)))
                * labels.size(0)
            )
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    top1 = top1_correct / total
    top5 = top5_correct / total
    avg_ms = float(np.mean(inference_times))

    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "top1_acc": top1,
        "top5_acc": top5,
        "avg_inference_ms": avg_ms,
        "all_preds": all_preds,
        "all_labels": all_labels,
        "report": report,
        "confusion_matrix": cm,
    }


def measure_inference_time(
    model: nn.Module,
    input_size: int = 224,
    n_runs: int = 100,
    batch_size: int = 1,
    device: torch.device | None = None,
) -> float:
    """
    Benchmark single-image inference time in milliseconds.

    Parameters
    ----------
    n_runs : int
        Number of forward passes to average over (first 10 are warm-up).
    """
    if device is None:
        device = next(model.parameters()).device

    dummy = torch.randn(batch_size, 3, input_size, input_size, device=device)
    model.eval()
    times = []

    with torch.no_grad():
        for i in range(n_runs + 10):
            t0 = time.perf_counter()
            _ = model(dummy)
            t1 = time.perf_counter()
            if i >= 10:  # skip warm-up
                times.append((t1 - t0) * 1000 / batch_size)

    return float(np.mean(times))


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def build_summary_df(results: dict[str, dict], model_params: dict[str, int]) -> pd.DataFrame:
    """
    Assemble all per-model metrics into a single DataFrame.

    Parameters
    ----------
    results : dict
        Mapping of model_key → evaluate_model() output.
    model_params : dict
        Mapping of model_key → total parameter count.
    """
    from models import MODEL_NAMES  # local import to avoid circular deps

    rows = []
    for key, res in results.items():
        rows.append({
            "Model": MODEL_NAMES.get(key, key),
            "Top-1 Acc (%)": round(res["top1_acc"] * 100, 2),
            "Top-5 Acc (%)": round(res["top5_acc"] * 100, 2),
            "Inf. Time (ms)": round(res["avg_inference_ms"], 2),
            "Params (M)": round(model_params.get(key, 0) / 1e6, 2),
            "Total Train Time (s)": round(res.get("total_time", 0), 1),
            "Best Val Acc (%)": round(res.get("best_val_acc", 0) * 100, 2),
        })
    df = pd.DataFrame(rows).sort_values("Top-1 Acc (%)", ascending=False).reset_index(drop=True)
    return df


def save_summary_csv(df: pd.DataFrame, path: str | Path = "results/summary.csv") -> None:
    """Save summary DataFrame to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Summary saved to %s", path)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_training_curves(
    history: dict,
    model_key: str,
    save_dir: str | Path = "results/plots",
) -> None:
    """Save loss and accuracy training curves for one model."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title(f"{model_key} – Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Val")
    axes[1].set_title(f"{model_key} – Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.suptitle(f"Training curves – {model_key}", fontsize=14)
    fig.tight_layout()
    out = save_dir / f"{model_key}_curves.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("Saved training curves → %s", out)


def plot_accuracy_comparison(
    summary_df: pd.DataFrame,
    save_dir: str | Path = "results/plots",
) -> None:
    """Bar chart comparing Top-1 accuracy across models."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["steelblue" if "ViT" in m or "Swin" in m or "DeiT" in m
              else "salmon" for m in summary_df["Model"]]
    ax.bar(summary_df["Model"], summary_df["Top-1 Acc (%)"], color=colors)
    ax.set_xlabel("Model")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Top-1 Accuracy Comparison (blue=Transformer, red=CNN)")
    ax.set_xticklabels(summary_df["Model"], rotation=35, ha="right")
    fig.tight_layout()
    out = save_dir / "accuracy_comparison.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("Saved accuracy comparison → %s", out)


def plot_inference_time_comparison(
    summary_df: pd.DataFrame,
    save_dir: str | Path = "results/plots",
) -> None:
    """Bar chart comparing inference time across models."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(summary_df["Model"], summary_df["Inf. Time (ms)"], color="mediumseagreen")
    ax.set_xlabel("Model")
    ax.set_ylabel("Avg Inference Time (ms / image)")
    ax.set_title("Inference Time Comparison")
    ax.set_xticklabels(summary_df["Model"], rotation=35, ha="right")
    fig.tight_layout()
    out = save_dir / "inference_time_comparison.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("Saved inference time comparison → %s", out)


def plot_size_vs_accuracy(
    summary_df: pd.DataFrame,
    save_dir: str | Path = "results/plots",
) -> None:
    """Scatter plot of model size (params) vs Top-1 accuracy."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(summary_df["Params (M)"], summary_df["Top-1 Acc (%)"], s=80)
    for _, row in summary_df.iterrows():
        ax.annotate(row["Model"], (row["Params (M)"], row["Top-1 Acc (%)"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Model Size vs Accuracy Trade-off")
    fig.tight_layout()
    out = save_dir / "size_vs_accuracy.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info("Saved size-vs-accuracy plot → %s", out)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    model_key: str,
    save_dir: str | Path = "results/plots",
    max_classes: int = 30,
) -> None:
    """
    Plot and save a confusion matrix.

    For readability, when there are more than *max_classes* classes, only the
    *max_classes* most frequently confused classes are shown.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if len(class_names) > max_classes:
        # Show the classes with most off-diagonal confusion
        off_diag = cm.copy()
        np.fill_diagonal(off_diag, 0)
        top_idx = np.argsort(off_diag.sum(axis=1))[-max_classes:]
        top_idx = np.sort(top_idx)
        cm = cm[np.ix_(top_idx, top_idx)]
        class_names = [class_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(max(10, len(class_names) // 2),
                                    max(8, len(class_names) // 2)))
    sns.heatmap(cm, annot=len(class_names) <= 20, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix – {model_key} (top-{len(class_names)} classes)")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    fig.tight_layout()
    out = save_dir / f"{model_key}_confusion_matrix.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    logger.info("Saved confusion matrix → %s", out)
