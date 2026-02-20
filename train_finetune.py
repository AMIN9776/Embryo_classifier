"""
train_finetune.py
-----------------
Supervised fine-tuning loop for 16-stage embryo development classification.

Features:
  - 3-phase progressive encoder unfreezing
  - Layer-wise LR decay
  - Soft-gated MoE classifier
  - Temporal GRU for sequence-level inference
  - Monotonicity loss
  - Auxiliary heads (cell count, blastocyst score)
  - Cross-validation support
  - AMP, gradient clipping, checkpointing

Run:
    python train_finetune.py --config config.yaml
    python train_finetune.py --config config.yaml --fold 0  # specific CV fold
    python train_finetune.py --config config.yaml --resume
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from utils import (
    load_config, save_config, set_seed, setup_device,
    Logger, CheckpointManager, MetricTracker,
    build_scheduler, build_scaler, get_amp_dtype,
    count_parameters, print_banner,
)
from dataset import EmbryoDataModule
from models import EmbryoFinetuneModel, load_pretrained_encoder
from losses import CombinedFinetuneLoss


# =============================================================================
# Training phase management
# =============================================================================

def set_training_phase(model: EmbryoFinetuneModel, epoch: int, cfg: dict,
                        logger) -> str:
    """
    Apply 3-phase progressive unfreezing based on current epoch.
    Returns current phase name.
    """
    ft_cfg = cfg["finetune"]
    p1 = ft_cfg["phase1_epochs"]
    p2 = p1 + ft_cfg["phase2_epochs"]

    if epoch < p1:
        # Phase 1: freeze encoder
        model.freeze_encoder()
        return "phase1_frozen"
    elif epoch < p2:
        # Phase 2: unfreeze last N blocks
        n_blocks = ft_cfg["unfreeze_blocks"]
        model.freeze_encoder()
        model.unfreeze_encoder_last_n_blocks(n_blocks)
        return f"phase2_partial_{n_blocks}blocks"
    else:
        # Phase 3: full encoder
        model.unfreeze_encoder()
        return "phase3_full"


def build_optimizer(model: EmbryoFinetuneModel, cfg: dict, epoch: int):
    """Build optimizer with layer-wise LR decay for current phase."""
    ft_cfg = cfg["finetune"]
    base_lr = ft_cfg["lr"]
    encoder_lr_scale = ft_cfg.get("encoder_lr_scale", 0.1)
    layer_lr_decay = ft_cfg.get("layer_lr_decay", 0.75)

    param_groups = model.get_encoder_parameter_groups(
        base_lr, encoder_lr_scale, layer_lr_decay
    )

    opt_name = ft_cfg["optimizer"]
    if opt_name == "adamw":
        return torch.optim.AdamW(
            param_groups,
            weight_decay=ft_cfg["weight_decay"],
            betas=(ft_cfg["beta1"], ft_cfg["beta2"]),
        )
    elif opt_name == "adam":
        return torch.optim.Adam(param_groups, weight_decay=ft_cfg["weight_decay"])
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


# =============================================================================
# Single epoch functions
# =============================================================================

def train_epoch(
    model: EmbryoFinetuneModel,
    loader,
    loss_fn: CombinedFinetuneLoss,
    optimizer,
    scaler,
    scheduler,
    device: torch.device,
    cfg: dict,
    amp_dtype,
    metric_tracker: MetricTracker,
    logger,
    global_step: int,
    epoch: int,
) -> dict:
    """Run one training epoch. Returns updated global_step and loss dict."""
    model.train()
    ft_cfg = cfg["finetune"]
    log_every = cfg["logging"]["log_every"]
    clip_norm = ft_cfg["clip_grad_norm"]
    use_amp = ft_cfg.get("use_amp", True)
    is_sequence = cfg["model"]["classifier"].get("use_temporal_gru", False)

    epoch_losses = {}
    epoch_steps = 0

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)

        # Determine input format (sequence or single-frame)
        if is_sequence and "focal_seq" in batch:
            inputs = batch["focal_seq"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
        else:
            inputs = batch["focal_imgs"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

        with autocast(enabled=use_amp, dtype=amp_dtype):
            if is_sequence:
                outputs = model(focal_seq=inputs)
            else:
                outputs = model(focal_imgs=inputs)

            logits = outputs["logits"]
            router_weights = outputs.get("router_weights")
            cell_count_logits = outputs.get("cell_count_logits")
            blast_scores = outputs.get("blastocyst_score")

            total_loss, loss_dict = loss_fn(
                logits=logits,
                labels=labels,
                cell_count_logits=cell_count_logits,
                blastocyst_scores=blast_scores,
                router_weights=router_weights,
                is_sequence=is_sequence,
            )

        # Backward
        if scaler:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        if cfg["finetune"]["scheduler"] != "plateau":
            scheduler.step()

        # Metrics
        with torch.no_grad():
            metric_tracker.update(logits.detach(), labels.detach(), loss_dict["total"])

        for k, v in loss_dict.items():
            epoch_losses[k] = epoch_losses.get(k, 0.0) + v
        epoch_steps += 1
        global_step += 1

        if global_step % log_every == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            log_m = dict(loss_dict)
            log_m["lr"] = lr_now
            logger.log(log_m, step=global_step, phase="finetune_train_step")

    epoch_avg = {k: v / epoch_steps for k, v in epoch_losses.items()}
    return global_step, epoch_avg


@torch.no_grad()
def evaluate(
    model: EmbryoFinetuneModel,
    loader,
    loss_fn: CombinedFinetuneLoss,
    device: torch.device,
    cfg: dict,
    amp_dtype,
    metric_tracker: MetricTracker,
) -> dict:
    """Evaluate on val/test loader. Returns metric dict."""
    model.eval()
    ft_cfg = cfg["finetune"]
    use_amp = ft_cfg.get("use_amp", True)
    is_sequence = cfg["model"]["classifier"].get("use_temporal_gru", False)

    for batch in loader:
        if is_sequence and "focal_seq" in batch:
            inputs = batch["focal_seq"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
        else:
            inputs = batch["focal_imgs"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

        with autocast(enabled=use_amp, dtype=amp_dtype):
            if is_sequence:
                outputs = model(focal_seq=inputs)
            else:
                outputs = model(focal_imgs=inputs)

            logits = outputs["logits"]
            _, loss_dict = loss_fn(logits=logits, labels=labels,
                                    is_sequence=is_sequence)

        metric_tracker.update(logits, labels, loss_dict["total"])

    return metric_tracker.compute()


# =============================================================================
# Main fine-tuning loop
# =============================================================================

def train_finetune(cfg: dict, resume: bool = False, fold_idx: Optional[int] = None):
    print_banner("Embryo Supervised Fine-Tuning")

    set_seed(cfg["runtime"]["seed"], cfg["runtime"]["deterministic"])
    device = setup_device(cfg)
    amp_dtype = get_amp_dtype(cfg["finetune"])

    fold_str = f"_fold{fold_idx}" if fold_idx is not None else ""
    output_dir = cfg["finetune"]["output_dir"] + fold_str
    os.makedirs(output_dir, exist_ok=True)
    save_config(cfg, output_dir)

    # Logging
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logger = Logger(cfg, run_name=f"finetune{fold_str}_{timestamp}")
    ckpt_manager = CheckpointManager(
        output_dir=output_dir,
        metric_name=cfg["logging"]["save_best_metric"],
        save_top_k=cfg["logging"]["save_top_k"],
        save_last=cfg["logging"]["save_last"],
        higher_is_better=True,
    )

    # Data
    logger.info("Setting up data...")
    dm = EmbryoDataModule(cfg)
    dm.setup()

    use_cv = cfg["data"].get("use_cross_validation", False)
    current_fold = fold_idx if (use_cv and fold_idx is not None) else None
    train_loader, val_loader, test_loader = dm.get_finetune_loaders(fold_idx=current_fold)

    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Class weights
    class_weights = None
    if cfg["finetune"].get("class_weights") is None:
        class_weights = dm.compute_class_weights().to(device)
        logger.info(f"Computed class weights: {class_weights.cpu().numpy().round(3)}")

    # Loss
    loss_fn = CombinedFinetuneLoss(cfg, class_weights=class_weights)

    # Model
    logger.info("Building model...")
    model = EmbryoFinetuneModel(cfg).to(device)
    logger.info(f"Parameters: {count_parameters(model)}")

    # Load pretrained encoder
    pretrain_ckpt = cfg["finetune"].get("pretrain_checkpoint")
    if pretrain_ckpt and os.path.exists(pretrain_ckpt):
        load_pretrained_encoder(model, pretrain_ckpt)
        logger.info(f"Loaded pretrained encoder from {pretrain_ckpt}")
    else:
        logger.info("No pretrained encoder found. Training from scratch.")

    if cfg["runtime"].get("compile", False):
        try:
            model = torch.compile(model)
        except Exception as e:
            logger.info(f"torch.compile() skipped: {e}")

    # Training state
    ft_cfg = cfg["finetune"]
    total_epochs = ft_cfg["epochs"]
    eval_every = ft_cfg["eval_every"]

    start_epoch = 0
    global_step = 0
    best_val_metric = float("-inf")
    current_optimizer = None
    current_scheduler = None
    scaler = build_scaler(ft_cfg)

    # Resume
    if resume:
        ckpt = ckpt_manager.load_last()
        if ckpt is not None:
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            global_step = ckpt.get("global_step", 0)
            best_val_metric = ckpt.get("best_val_metric", float("-inf"))
            if scaler and "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            logger.info(f"Resumed from epoch {ckpt['epoch']}")

    # ─── Epoch loop ───────────────────────────────────────────────────────
    logger.info(f"Starting fine-tuning for {total_epochs} epochs")
    prev_phase = None

    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()

        # ── Phase management ──────────────────────────────────────────────
        phase = set_training_phase(model, epoch, cfg, logger)
        if phase != prev_phase:
            logger.info(f"  ↳ Training phase: {phase}")
            # Rebuild optimizer when phase changes (different param groups)
            current_optimizer = build_optimizer(model, cfg, epoch)
            current_scheduler = build_scheduler(
                current_optimizer, ft_cfg, len(train_loader)
            )
            # If resuming and we have saved optimizer state, only restore for phase 1
            if resume and epoch == start_epoch and prev_phase is None:
                ckpt = ckpt_manager.load_last()
                if ckpt and "optimizer_state_dict" in ckpt:
                    try:
                        current_optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    except Exception:
                        pass
            prev_phase = phase

        # ── Train ─────────────────────────────────────────────────────────
        train_tracker = MetricTracker(cfg["data"]["num_stages"])
        global_step, train_losses = train_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=current_optimizer,
            scaler=scaler,
            scheduler=current_scheduler,
            device=device,
            cfg=cfg,
            amp_dtype=amp_dtype,
            metric_tracker=train_tracker,
            logger=logger,
            global_step=global_step,
            epoch=epoch,
        )
        train_metrics = train_tracker.compute()
        train_metrics.update(train_losses)
        train_metrics["phase"] = hash(phase) % 10000  # just for logging
        train_metrics["lr"] = current_optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start
        train_metrics["epoch_time_s"] = epoch_time
        logger.log(train_metrics, step=epoch, phase="finetune_train_epoch")

        # ── Validate ──────────────────────────────────────────────────────
        if (epoch + 1) % eval_every == 0 or epoch == total_epochs - 1:
            val_tracker = MetricTracker(cfg["data"]["num_stages"])
            val_metrics = evaluate(
                model, val_loader, loss_fn, device, cfg, amp_dtype, val_tracker
            )
            logger.log(val_metrics, step=epoch, phase="finetune_val_epoch")

            # LR plateau scheduler
            if ft_cfg["scheduler"] == "plateau":
                current_scheduler.step(val_metrics["accuracy_top1"])

            # Checkpoint
            save_metric = val_metrics.get(cfg["logging"]["save_best_metric"], 0.0)
            state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": current_optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "epoch": epoch,
                "global_step": global_step,
                "best_val_metric": best_val_metric,
                "cfg": cfg,
                "fold": fold_idx,
            }
            is_best = ckpt_manager.save(state, epoch, val_metrics)
            if is_best:
                best_val_metric = save_metric
                logger.info(
                    f"  ★ New best at epoch {epoch}: "
                    f"{cfg['logging']['save_best_metric']}={save_metric:.4f}"
                )

            logger.info(
                f"Epoch {epoch}/{total_epochs-1} | "
                f"train_acc={train_metrics['accuracy_top1']:.3f} | "
                f"val_acc={val_metrics['accuracy_top1']:.3f} | "
                f"val_f1={val_metrics['f1_macro']:.3f} | "
                f"time={epoch_time:.1f}s"
            )
        else:
            logger.info(
                f"Epoch {epoch}/{total_epochs-1} | "
                f"train_acc={train_metrics['accuracy_top1']:.3f} | "
                f"time={epoch_time:.1f}s"
            )

    # ─── Final test evaluation ──────────────────────────────────────────
    logger.info("\n=== Final Test Evaluation ===")
    best_ckpt = ckpt_manager.load()
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_tracker = MetricTracker(cfg["data"]["num_stages"])
    test_metrics = evaluate(model, test_loader, loss_fn, device, cfg, amp_dtype, test_tracker)
    logger.log(test_metrics, step=total_epochs, phase="test_final")

    logger.info("Test Results:")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")

    logger.info("Fine-tuning complete!")
    logger.close()
    return test_metrics


# =============================================================================
# Cross-validation runner
# =============================================================================

def run_cross_validation(cfg: dict):
    """Run K-fold cross-validation."""
    print_banner("Cross-Validation Run")
    n_folds = cfg["data"]["cv_folds"]
    all_metrics = []

    for fold in range(n_folds):
        print(f"\n{'='*40}\n  Fold {fold+1}/{n_folds}\n{'='*40}")
        test_metrics = train_finetune(cfg, fold_idx=fold)
        all_metrics.append(test_metrics)

    # Aggregate
    print_banner("Cross-Validation Summary")
    aggregated = {}
    for key in all_metrics[0].keys():
        vals = [m[key] for m in all_metrics if isinstance(m.get(key), float)]
        if vals:
            import numpy as np
            aggregated[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    for k, v in aggregated.items():
        if "acc" in k or "f1" in k:
            print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")

    return aggregated


# =============================================================================
# Entry point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Embryo Fine-Tuning")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fold", type=int, default=None,
                        help="Specific CV fold (overrides config)")
    parser.add_argument("--cv", action="store_true",
                        help="Run full cross-validation")
    parser.add_argument("--pretrain_checkpoint", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    if args.pretrain_checkpoint:
        cfg["finetune"]["pretrain_checkpoint"] = args.pretrain_checkpoint
    if args.batch_size:
        cfg["finetune"]["batch_size"] = args.batch_size
    if args.epochs:
        cfg["finetune"]["epochs"] = args.epochs
    if args.output_dir:
        cfg["finetune"]["output_dir"] = args.output_dir
    if args.fold is not None:
        cfg["data"]["cv_fold_index"] = args.fold

    if args.cv or cfg["data"].get("use_cross_validation", False):
        run_cross_validation(cfg)
    else:
        fold = args.fold if args.fold is not None else None
        train_finetune(cfg, resume=args.resume, fold_idx=fold)
