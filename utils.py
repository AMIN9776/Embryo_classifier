"""
utils.py
--------
Shared utilities: config loading, logging (TensorBoard + CSV), checkpoint
management, metrics computation, and seed setting.
"""

import os
import csv
import json
import random
import shutil
import logging
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime

import numpy as np
import torch
import yaml


# =============================================================================
# Config
# =============================================================================

def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_config(cfg: dict, output_dir: str):
    """Save config to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int, deterministic: bool = False):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def setup_device(cfg: dict) -> torch.device:
    """Set up compute device."""
    rt = cfg["runtime"]
    if rt["device"] == "cuda" and torch.cuda.is_available():
        gpu_id = rt.get("gpu_id", 0)
        device = torch.device(f"cuda:{gpu_id}")
        print(f"[setup_device] Using GPU: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device("cpu")
        print("[setup_device] Using CPU")
    return device


# =============================================================================
# Logger
# =============================================================================

class Logger:
    """
    Unified logger supporting TensorBoard and/or CSV backends.

    Usage:
        logger = Logger(cfg, run_name="pretrain")
        logger.log({"loss": 0.5, "acc": 0.8}, step=100, phase="train")
        logger.close()
    """

    def __init__(self, cfg: dict, run_name: str = "run"):
        log_cfg = cfg["logging"]
        self.backend = log_cfg["backend"]  # 'tensorboard', 'csv', 'both'
        log_dir = Path(log_cfg["log_dir"]) / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir

        # Set up Python logger
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_dir / "training.log"),
            ],
        )
        self.py_logger = logging.getLogger(run_name)

        # TensorBoard
        self.tb_writer = None
        if self.backend in ("tensorboard", "both"):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))
                self.py_logger.info(f"TensorBoard logging to {log_dir / 'tensorboard'}")
            except ImportError:
                self.py_logger.warning("TensorBoard not available. Install with: pip install tensorboard")

        # CSV
        self.csv_files: Dict[str, Any] = {}
        if self.backend in ("csv", "both"):
            self.csv_dir = log_dir / "csv"
            self.csv_dir.mkdir(exist_ok=True)

        self.py_logger.info(f"Logger initialised | run={run_name} | backend={self.backend}")

    def log(self, metrics: Dict[str, float], step: int, phase: str = "train"):
        """Log a dict of metrics."""
        # TensorBoard
        if self.tb_writer is not None:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(f"{phase}/{k}", v, step)

        # CSV
        if self.backend in ("csv", "both"):
            csv_path = self.csv_dir / f"{phase}.csv"
            write_header = not csv_path.exists()
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["step"] + sorted(metrics.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow({"step": step, **metrics})

        # Console (info level)
        msg = f"[{phase}] step={step} | " + " | ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )
        self.py_logger.info(msg)

    def info(self, msg: str):
        self.py_logger.info(msg)

    def close(self):
        if self.tb_writer is not None:
            self.tb_writer.close()


# =============================================================================
# Checkpoint Manager
# =============================================================================

class CheckpointManager:
    """
    Manages saving and loading of model checkpoints.

    Keeps track of top-K checkpoints based on a metric.
    """

    def __init__(self, output_dir: str, metric_name: str = "accuracy_top1",
                 save_top_k: int = 3, save_last: bool = True,
                 higher_is_better: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.higher_is_better = higher_is_better

        self._checkpoints: List[Dict] = []  # [{metric, path}, ...]
        self._best_metric = float("-inf") if higher_is_better else float("inf")
        self._best_path: Optional[Path] = None

    def save(
        self,
        state: dict,
        epoch: int,
        metrics: Dict[str, float],
    ) -> bool:
        """
        Save a checkpoint. Returns True if this is the new best.

        state should contain: model_state_dict, optimizer_state_dict,
                              scaler_state_dict, epoch, cfg
        """
        metric_val = metrics.get(self.metric_name, 0.0)
        state["epoch"] = epoch
        state["metrics"] = metrics

        # Save last
        if self.save_last:
            last_path = self.output_dir / "last_checkpoint.pth"
            torch.save(state, last_path)

        # Epoch checkpoint
        epoch_path = self.output_dir / f"epoch_{epoch:04d}.pth"
        torch.save(state, epoch_path)

        # Track and maintain top-K
        self._checkpoints.append({
            "metric": metric_val,
            "path": epoch_path,
            "epoch": epoch,
        })

        if self.higher_is_better:
            self._checkpoints.sort(key=lambda x: -x["metric"])
        else:
            self._checkpoints.sort(key=lambda x: x["metric"])

        # Remove checkpoints beyond top-K (but keep epoch ones that are in top-K)
        while len(self._checkpoints) > self.save_top_k:
            to_remove = self._checkpoints.pop()
            if to_remove["path"].exists():
                to_remove["path"].unlink()

        # Check if this is new best
        current_val = metrics.get(self.metric_name, float("-inf"))
        is_best = (
            (self.higher_is_better and current_val > self._best_metric) or
            (not self.higher_is_better and current_val < self._best_metric)
        )
        if is_best:
            self._best_metric = current_val
            best_path = self.output_dir / "best_model.pth"
            shutil.copy2(epoch_path, best_path)
            self._best_path = best_path

        return is_best

    def load(self, path: Optional[str] = None) -> dict:
        """Load a checkpoint. If path is None, loads the best checkpoint."""
        if path is None:
            path = self.output_dir / "best_model.pth"
        ckpt = torch.load(path, map_location="cpu")
        return ckpt

    def load_last(self) -> Optional[dict]:
        """Load last checkpoint for resuming training."""
        last_path = self.output_dir / "last_checkpoint.pth"
        if last_path.exists():
            return torch.load(last_path, map_location="cpu")
        return None


# =============================================================================
# Metrics
# =============================================================================

class MetricTracker:
    """Accumulates and computes classification metrics."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.total = 0
        self.correct_top1 = 0
        self.correct_top3 = 0
        self.loss_sum = 0.0
        self.n_batches = 0
        # Per-class tracking
        self.per_class_correct = torch.zeros(self.num_classes)
        self.per_class_total = torch.zeros(self.num_classes)
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes)

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss: float = 0.0,
    ):
        """
        logits: [B, C] or [B, T, C]
        labels: [B] or [B, T]
        """
        if logits.dim() == 3:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            labels = labels.view(B * T)

        labels = labels.cpu()
        logits = logits.cpu()

        B = labels.shape[0]
        self.total += B
        self.loss_sum += loss * B
        self.n_batches += 1

        # Top-1
        preds = logits.argmax(dim=1)
        self.correct_top1 += (preds == labels).sum().item()

        # Top-3
        k = min(3, logits.shape[1])
        top3_preds = logits.topk(k, dim=1).indices
        for i in range(B):
            if labels[i] in top3_preds[i]:
                self.correct_top3 += 1

        # Per-class
        for cls in range(self.num_classes):
            mask = labels == cls
            self.per_class_total[cls] += mask.sum().item()
            self.per_class_correct[cls] += (preds[mask] == cls).sum().item()

        # Confusion matrix
        for pred, true in zip(preds, labels):
            if 0 <= true < self.num_classes and 0 <= pred < self.num_classes:
                self.confusion_matrix[true, pred] += 1

    def compute(self) -> Dict[str, float]:
        eps = 1e-8
        metrics = {
            "accuracy_top1": self.correct_top1 / (self.total + eps),
            "accuracy_top3": self.correct_top3 / (self.total + eps),
            "avg_loss": self.loss_sum / (self.total + eps),
        }

        # Per-class accuracy
        per_class_acc = self.per_class_correct / (self.per_class_total + eps)
        metrics["mean_per_class_accuracy"] = per_class_acc.mean().item()

        # Macro F1 (approximation from per-class stats)
        cm = self.confusion_matrix
        f1_scores = []
        for c in range(self.num_classes):
            tp = cm[c, c].item()
            fp = cm[:, c].sum().item() - tp
            fn = cm[c, :].sum().item() - tp
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            f1_scores.append(f1)
        metrics["f1_macro"] = float(np.mean(f1_scores))

        # Add per-stage accuracy
        for i, acc in enumerate(per_class_acc.tolist()):
            metrics[f"stage_{i+1}_acc"] = acc

        return metrics


# =============================================================================
# Learning Rate Scheduler
# =============================================================================

def build_scheduler(optimizer, cfg_section: dict, num_steps_per_epoch: int):
    """Build LR scheduler from config."""
    scheduler_type = cfg_section.get("scheduler", "cosine")
    total_epochs = cfg_section.get("epochs", 100)
    warmup_epochs = cfg_section.get("warmup_epochs", 10)
    min_lr = cfg_section.get("min_lr", 1e-6)

    if scheduler_type == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup = LinearLR(
            optimizer,
            start_factor=1e-4,
            end_factor=1.0,
            total_iters=warmup_epochs * num_steps_per_epoch,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=(total_epochs - warmup_epochs) * num_steps_per_epoch,
            eta_min=min_lr,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs * num_steps_per_epoch],
        )
    elif scheduler_type == "step":
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=30 * num_steps_per_epoch, gamma=0.1)
    elif scheduler_type == "plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")

    return scheduler


# =============================================================================
# AMP scaler helper
# =============================================================================

def build_scaler(cfg_section: dict):
    """Build GradScaler for AMP training."""
    if cfg_section.get("use_amp", True):
        dtype = cfg_section.get("amp_dtype", "float16")
        if dtype == "bfloat16":
            return torch.cuda.amp.GradScaler(enabled=False)  # bfloat16 doesn't need scaling
        return torch.cuda.amp.GradScaler()
    return None


def get_amp_dtype(cfg_section: dict):
    """Get AMP dtype from config."""
    if not cfg_section.get("use_amp", True):
        return None
    dtype_str = cfg_section.get("amp_dtype", "float16")
    return torch.float16 if dtype_str == "float16" else torch.bfloat16


# =============================================================================
# Misc helpers
# =============================================================================

def count_parameters(model: torch.nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total:,} | Trainable: {trainable:,}"


def print_banner(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)
