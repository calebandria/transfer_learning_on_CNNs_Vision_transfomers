"""
transformer_trainer.py – Robust, memory-safe, reproducible transformer training loop.

Provides a unified interface for fine-tuning multiple HuggingFace Transformers
(BERT, RoBERTa, DeBERTa, etc.) with:

- Error handling:   try/except per model so the loop continues on failure.
- tqdm progress:    outer bar over models, inner bar over training batches.
- best_state check: warns and skips if no improvement was recorded.
- Model caching:    ``cache_dir`` prevents repeated HuggingFace downloads.
- OOM fallback:     ``force_cpu_on_oom`` retries on CPU when CUDA runs out of memory.
- Skip-if-trained:  ``skip_trained`` skips models whose checkpoint already exists.
- Random seeds:     ``set_seed`` is called before every model's training run.
- Explicit cleanup: ``del`` large objects + ``gc.collect`` + ``cuda.empty_cache``.
- Training summary: prints success / failure counts at the end.

Typical usage
-------------
>>> from transformer_trainer import run_transformer_comparison, CONFIG, NUM_CLASSES
>>> histories, models, tokenizers, test_loaders = run_transformer_comparison(
...     train_df, val_df, test_df,
...     num_classes=NUM_CLASSES,
...     config=CONFIG,
... )
"""

from __future__ import annotations

import copy
import gc
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    _TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TRANSFORMERS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

#: Default training hyper-parameters.  Override by passing your own ``config``
#: dict to :func:`run_transformer_comparison`.
CONFIG: dict = {
    "batch_size": 16,
    "max_length": 128,
    "num_epochs": 3,
    "lr": 2e-5,
    "seed": 42,
}

#: Default number of output classes.
NUM_CLASSES: int = 2

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

#: Mapping of human-readable display names → HuggingFace model identifiers.
#: Extend or replace this dict when calling :func:`run_transformer_comparison`.
MODEL_REGISTRY: Dict[str, str] = {
    "BERT-base": "google-bert/bert-base-uncased",
    "RoBERTa-base": "roberta-base",
    "DeBERTa-v3-base": "microsoft/deberta-v3-base",
}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for Python, NumPy, and PyTorch (CPU + all GPUs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TextClassificationDataset(Dataset):
    """Minimal :class:`torch.utils.data.Dataset` for tokenised text classification."""

    def __init__(self, encodings: dict, labels: List[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def build_dataloaders(
    train_df,
    val_df,
    test_df,
    tokenizer,
    *,
    text_col: str = "text",
    label_col: str = "labels",
    batch_size: int = 16,
    max_length: int = 128,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Tokenise DataFrames and return ``(train_loader, val_loader, test_loader)``.

    Parameters
    ----------
    train_df, val_df, test_df : pandas.DataFrame
        DataFrames containing at least *text_col* and *label_col* columns.
    tokenizer
        A HuggingFace tokenizer (already loaded via ``AutoTokenizer.from_pretrained``).
    text_col : str
        Column name for raw input text.
    label_col : str
        Column name for integer class labels.
    batch_size : int
        Mini-batch size used for all three loaders.
    max_length : int
        Maximum token-sequence length; longer sequences are truncated.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        ``(train_loader, val_loader, test_loader)``
    """

    def _encode(df) -> dict:
        return tokenizer(
            df[text_col].tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    def _make_loader(df, shuffle: bool) -> DataLoader:
        enc = _encode(df)
        ds = TextClassificationDataset(enc, df[label_col].tolist())
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return (
        _make_loader(train_df, shuffle=True),
        _make_loader(val_df, shuffle=False),
        _make_loader(test_df, shuffle=False),
    )


# ---------------------------------------------------------------------------
# Single-model training
# ---------------------------------------------------------------------------


def train_transformer(
    *,
    model_name: str,
    model_display: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    config: dict,
    device: torch.device,
    cache_dir: Optional[str] = None,
) -> dict:
    """Fine-tune a HuggingFace sequence-classification model.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g. ``"google-bert/bert-base-uncased"``).
    model_display : str
        Human-readable label used in log messages and tqdm descriptions.
    train_loader, val_loader : DataLoader
        Prepared :class:`torch.utils.data.DataLoader` instances.
    num_classes : int
        Number of output classes.
    config : dict
        Training configuration.  Recognised keys: ``num_epochs`` (int),
        ``lr`` (float).
    device : torch.device
        Device on which to run training.
    cache_dir : str, optional
        Local directory for HuggingFace model cache.  Avoids repeated downloads
        across runs or across models that share a base checkpoint.

    Returns
    -------
    dict
        History with keys ``train_loss``, ``val_loss``, ``val_acc``,
        ``best_val_acc``, ``best_epoch``, and ``best_state``
        (the ``state_dict`` of the best checkpoint, or *None* if val accuracy
        never improved above 0).
    """
    if not _TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "The 'transformers' library is required.  "
            "Install it with: pip install transformers"
        )

    num_epochs: int = config.get("num_epochs", 3)
    lr: float = config.get("lr", 2e-5)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        cache_dir=cache_dir,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history: dict = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "best_val_acc": 0.0,
        "best_epoch": 1,
        "best_state": None,
    }

    for epoch in range(1, num_epochs + 1):
        # ---- Training phase ----
        model.train()
        running_loss = 0.0
        n_train = 0

        for batch in tqdm(
            train_loader,
            desc=f"[{model_display}] Epoch {epoch}/{num_epochs}",
            leave=False,
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            n_train += labels.size(0)

        train_loss = running_loss / max(n_train, 1)

        # ---- Validation phase ----
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                val_loss_sum += loss.item() * labels.size(0)
                preds = outputs.logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        logger.info(
            "[%s] Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f",
            model_display,
            epoch,
            num_epochs,
            train_loss,
            val_loss,
            val_acc,
        )

        if val_acc > history["best_val_acc"]:
            history["best_val_acc"] = val_acc
            history["best_epoch"] = epoch
            history["best_state"] = copy.deepcopy(model.state_dict())

    return history


# ---------------------------------------------------------------------------
# Main comparison loop
# ---------------------------------------------------------------------------


def run_transformer_comparison(
    train_df,
    val_df,
    test_df,
    num_classes: int,
    config: dict,
    *,
    model_registry: Optional[Dict[str, str]] = None,
    device: Optional[torch.device] = None,
    cache_dir: Optional[str] = None,
    skip_trained: bool = False,
    trained_cache_dir: str = "checkpoints/transformers",
    seed: int = 42,
    force_cpu_on_oom: bool = True,
) -> Tuple[
    Dict[str, dict],
    Dict[str, nn.Module],
    Dict[str, object],
    Dict[str, DataLoader],
]:
    """Train and evaluate multiple HuggingFace transformer models.

    Implements a robust, memory-safe loop with error handling, progress
    tracking, model caching, OOM fallback, and a final summary.

    Parameters
    ----------
    train_df, val_df, test_df : pandas.DataFrame
        Split dataframes containing ``text`` and ``labels`` columns (or the
        column names configured in ``config`` via ``text_col``/``label_col``).
    num_classes : int
        Number of target classes for sequence classification.
    config : dict
        Training hyper-parameters.  Recognised keys:

        - ``batch_size`` (int, default 16)
        - ``max_length`` (int, default 128)
        - ``num_epochs`` (int, default 3)
        - ``lr`` (float, default 2e-5)
        - ``text_col`` (str, default ``"text"``)
        - ``label_col`` (str, default ``"labels"``)

    model_registry : dict, optional
        Mapping of ``display_name → hf_model_id``.  Defaults to
        :data:`MODEL_REGISTRY`.
    device : torch.device, optional
        Target device.  Defaults to CUDA if available, otherwise CPU.
    cache_dir : str, optional
        Local directory used as the HuggingFace model/tokenizer cache.
        Prevents re-downloading weights on every run or across models that
        share the same base checkpoint.
    skip_trained : bool
        When *True*, skip training for any model whose checkpoint file already
        exists in *trained_cache_dir*.  The saved weights are loaded instead.
    trained_cache_dir : str
        Directory where per-model ``.pt`` checkpoint files are stored.
    seed : int
        Random seed applied (via :func:`set_seed`) before each model's
        training run for reproducibility.
    force_cpu_on_oom : bool
        When *True*, if CUDA raises an out-of-memory error the model is
        automatically retrained on CPU.

    Returns
    -------
    tuple
        ``(all_histories, all_models, all_tokenizers, all_test_loaders)``

        - ``all_histories``  – ``{display_name: history_dict}``
        - ``all_models``     – ``{display_name: best_model}``
        - ``all_tokenizers`` – ``{display_name: tokenizer}``
        - ``all_test_loaders`` – ``{display_name: test_loader}``

        Only successfully trained models appear in the returned dicts.
    """
    if model_registry is None:
        model_registry = MODEL_REGISTRY

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trained_cache_path = Path(trained_cache_dir)
    trained_cache_path.mkdir(parents=True, exist_ok=True)

    all_histories: Dict[str, dict] = {}
    all_models: Dict[str, nn.Module] = {}
    all_tokenizers: Dict[str, object] = {}
    all_test_loaders: Dict[str, DataLoader] = {}
    failed_models: List[str] = []

    text_col: str = config.get("text_col", "text")
    label_col: str = config.get("label_col", "labels")

    model_items = list(model_registry.items())

    for display_name, hf_name in tqdm(
        model_items, desc="Training transformers", unit="model"
    ):
        ckpt_file = trained_cache_path / f"{display_name.replace('/', '_')}.pt"

        # ------------------------------------------------------------------ #
        # Optional: skip already-trained models                               #
        # ------------------------------------------------------------------ #
        if skip_trained and ckpt_file.exists():
            logger.info("[%s] Checkpoint found – skipping retraining.", display_name)
            try:
                saved = torch.load(ckpt_file, map_location="cpu", weights_only=True)
                best_state = saved.get("best_state")
                if best_state is None:
                    raise ValueError("Checkpoint contains no best_state.")

                tokenizer = AutoTokenizer.from_pretrained(hf_name, cache_dir=cache_dir)
                best_model = AutoModelForSequenceClassification.from_pretrained(
                    hf_name,
                    num_labels=num_classes,
                    ignore_mismatched_sizes=True,
                    cache_dir=cache_dir,
                )
                best_model.load_state_dict(best_state)
                best_model.to(device)

                _, _, test_loader = build_dataloaders(
                    train_df,
                    val_df,
                    test_df,
                    tokenizer,
                    text_col=text_col,
                    label_col=label_col,
                    batch_size=config.get("batch_size", 16),
                    max_length=config.get("max_length", 128),
                )

                all_histories[display_name] = saved["history"]
                all_models[display_name] = best_model
                all_tokenizers[display_name] = tokenizer
                all_test_loaders[display_name] = test_loader
                continue

            except Exception as skip_exc:
                logger.warning(
                    "[%s] Failed to load cached checkpoint (%s) – retraining.",
                    display_name,
                    skip_exc,
                )

        # ------------------------------------------------------------------ #
        # Training                                                             #
        # ------------------------------------------------------------------ #
        print(f'\n{"=" * 60}')
        print(f"  Training: {display_name} ({hf_name})")
        print(f'{"=" * 60}')

        # Consistent seed before each model
        set_seed(seed)

        run_device = device
        tokenizer = None
        train_loader = val_loader = test_loader = None
        best_model = None

        try:
            # ---- Tokenizer (pre-loaded and re-used for all dataloaders) ----
            tokenizer = AutoTokenizer.from_pretrained(hf_name, cache_dir=cache_dir)

            # ---- DataLoaders ----
            train_loader, val_loader, test_loader = build_dataloaders(
                train_df,
                val_df,
                test_df,
                tokenizer,
                text_col=text_col,
                label_col=label_col,
                batch_size=config.get("batch_size", 16),
                max_length=config.get("max_length", 128),
            )

            # ---- Train (with optional CPU fallback on OOM) ----
            try:
                history = train_transformer(
                    model_name=hf_name,
                    model_display=display_name,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    num_classes=num_classes,
                    config=config,
                    device=run_device,
                    cache_dir=cache_dir,
                )
            except RuntimeError as oom_err:
                if force_cpu_on_oom and "out of memory" in str(oom_err).lower():
                    logger.warning(
                        "[%s] CUDA OOM – retrying on CPU.", display_name
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    run_device = torch.device("cpu")
                    history = train_transformer(
                        model_name=hf_name,
                        model_display=display_name,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        num_classes=num_classes,
                        config=config,
                        device=run_device,
                        cache_dir=cache_dir,
                    )
                else:
                    raise

            # ---- Validate best_state ----
            if history.get("best_state") is None:
                logger.warning(
                    "[%s] best_state is None – no improvement was recorded "
                    "during training.  Skipping this model.",
                    display_name,
                )
                failed_models.append(display_name)
                continue

            # ---- Reload best weights (uses cached weights – no extra download) ----
            best_model = AutoModelForSequenceClassification.from_pretrained(
                hf_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
                cache_dir=cache_dir,
            )
            best_model.load_state_dict(history["best_state"])
            best_model.to(run_device)

            # ---- Persist checkpoint ----
            torch.save(
                {
                    "history": {k: v for k, v in history.items() if k != "best_state"},
                    "best_state": history["best_state"],
                },
                ckpt_file,
            )
            logger.info("[%s] Checkpoint saved → %s", display_name, ckpt_file)

            all_histories[display_name] = history
            all_models[display_name] = best_model
            all_tokenizers[display_name] = tokenizer
            all_test_loaders[display_name] = test_loader

        except Exception as exc:
            logger.error(
                "[%s] Training failed: %s", display_name, exc, exc_info=True
            )
            failed_models.append(display_name)

        finally:
            # ---- Explicit cleanup: free GPU memory between models ----
            del train_loader, val_loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---------------------------------------------------------------------- #
    # Summary                                                                 #
    # ---------------------------------------------------------------------- #
    print(f'\n{"=" * 60}')
    print("  Transformer Training Summary")
    print(f'{"=" * 60}')
    succeeded = list(all_histories.keys())
    print(f"  Succeeded ({len(succeeded)}): {', '.join(succeeded) or 'none'}")
    print(f"  Failed    ({len(failed_models)}): {', '.join(failed_models) or 'none'}")
    print(f'{"=" * 60}\n')

    return all_histories, all_models, all_tokenizers, all_test_loaders
