"""
train_pretrain.py
-----------------
Self-supervised pretraining loop.

Objectives (weights configurable in config.yaml):
  1. Cross-focal contrastive (InfoNCE)
  2. MAE masked reconstruction (same-focal or best-focus target)
  3. Temporal gap prediction

Run:
    python train_pretrain.py --config config.yaml
    python train_pretrain.py --config config.yaml --resume   # auto-resumes from last ckpt
"""

import argparse
import os
import time
from pathlib import Path

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
from models import EmbryoPretrainModel
from losses import CombinedPretrainLoss


# =============================================================================
# Training step
# =============================================================================

def pretrain_step(
    batch: dict,
    model: EmbryoPretrainModel,
    loss_fn: CombinedPretrainLoss,
    device: torch.device,
    cfg: dict,
    amp_dtype,
) -> dict:
    """
    Run a single pretraining forward + loss computation.
    Returns dict of losses (no backward yet).
    """
    pt_cfg = cfg["pretrain"]
    mae_target_mode = pt_cfg["mae"]["reconstruction_target"]
    use_amp = pt_cfg.get("use_amp", True)

    # Move batch to device
    view1 = batch["focal_view1"].to(device, non_blocking=True)   # [B, N, C, H, W]
    view2 = batch["focal_view2"].to(device, non_blocking=True)
    raw = batch["focal_raw"].to(device, non_blocking=True)
    temporal = batch["temporal_view"].to(device, non_blocking=True)
    gap_bins = batch["gap_bin"].to(device, non_blocking=True)
    best_focus_idx = batch["best_focus_idx"].to(device, non_blocking=True)

    B = view1.shape[0]
    N = view1.shape[1]

    with autocast(enabled=use_amp, dtype=amp_dtype):
        # ── Contrastive: two global views ─────────────────────────────────
        z1, z2 = model.forward_contrastive(view1, view2)

        # ── Cross-focal contrastive: treat focal planes as positives ──────
        cross_focal_projs = model.forward_cross_focal_contrastive(raw)

        # ── MAE (ViT only) ─────────────────────────────────────────────────
        mae_pred = mae_target = mae_mask = None
        if model.is_vit and pt_cfg["loss_weights"].get("mae", 0) > 0:
            if mae_target_mode == "best_focus":
                target_idx = best_focus_idx
            else:
                target_idx = None
            mae_pred, mae_target, mae_mask = model.forward_mae(raw, target_idx)

        # ── Temporal gap prediction ────────────────────────────────────────
        temporal_logits = None
        if pt_cfg["loss_weights"].get("temporal", 0) > 0:
            temporal_logits = model.forward_temporal(raw, temporal)

        # ── Compute combined loss ──────────────────────────────────────────
        total_loss, loss_dict = loss_fn(
            z1=z1, z2=z2,
            cross_focal_projs=cross_focal_projs,
            B=B, N=N,
            mae_pred=mae_pred,
            mae_target=mae_target,
            mae_mask=mae_mask,
            temporal_logits=temporal_logits,
            gap_bins=gap_bins,
        )

    return total_loss, loss_dict


# =============================================================================
# Main training loop
# =============================================================================

def train_pretrain(cfg: dict, resume: bool = False):
    print_banner("Embryo Self-Supervised Pretraining")

    # Setup
    set_seed(cfg["runtime"]["seed"], cfg["runtime"]["deterministic"])
    device = setup_device(cfg)
    amp_dtype = get_amp_dtype(cfg["pretrain"])

    output_dir = cfg["pretrain"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    save_config(cfg, output_dir)

    # Logging
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logger = Logger(cfg, run_name=f"pretrain_{timestamp}")
    ckpt_manager = CheckpointManager(
        output_dir=output_dir,
        metric_name="avg_loss",
        save_top_k=cfg["logging"]["save_top_k"],
        save_last=cfg["logging"]["save_last"],
        higher_is_better=False,
    )

    # Data
    logger.info("Setting up data...")
    dm = EmbryoDataModule(cfg)
    dm.setup()
    train_loader = dm.get_pretrain_loader()
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Steps per epoch: {len(train_loader)}")

    # Model
    logger.info("Building model...")
    model = EmbryoPretrainModel(cfg).to(device)
    logger.info(f"Parameters: {count_parameters(model)}")

    if cfg["pretrain"].get("gradient_checkpointing", False):
        if hasattr(model.encoder, "blocks"):
            for blk in model.encoder.blocks:
                blk.use_checkpoint = True
            logger.info("Gradient checkpointing enabled")

    if cfg["runtime"].get("compile", False):
        try:
            model = torch.compile(model)
            logger.info("Model compiled with torch.compile()")
        except Exception as e:
            logger.info(f"torch.compile() failed: {e}")

    # Loss
    loss_fn = CombinedPretrainLoss(cfg)

    # Optimizer
    pt_cfg = cfg["pretrain"]
    lr = pt_cfg["lr"]
    if pt_cfg.get("lr_scale", True):
        lr = lr * pt_cfg["batch_size"] / 256
        logger.info(f"Scaled LR: {lr:.2e}")

    opt_cfg = pt_cfg["optimizer"]
    if opt_cfg == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=pt_cfg["weight_decay"],
            betas=(pt_cfg["beta1"], pt_cfg["beta2"]),
        )
    elif opt_cfg == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=pt_cfg["weight_decay"]
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_cfg}")

    scaler = build_scaler(pt_cfg)
    scheduler = build_scheduler(optimizer, pt_cfg, len(train_loader))

    # Resume
    start_epoch = 0
    global_step = 0
    if resume:
        ckpt = ckpt_manager.load_last()
        if ckpt is not None:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if scaler and "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            global_step = ckpt.get("global_step", 0)
            logger.info(f"Resumed from epoch {ckpt['epoch']}")

    # Training loop
    total_epochs = pt_cfg["epochs"]
    log_every = cfg["logging"]["log_every"]
    eval_every = pt_cfg["eval_every"]
    clip_norm = pt_cfg["clip_grad_norm"]

    logger.info(f"Starting pretraining for {total_epochs} epochs")

    for epoch in range(start_epoch, total_epochs):
        model.train()
        epoch_losses = {}
        epoch_steps = 0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)

            total_loss, loss_dict = pretrain_step(
                batch, model, loss_fn, device, cfg, amp_dtype
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

            # Step scheduler (per-step)
            if cfg["pretrain"]["scheduler"] != "plateau":
                scheduler.step()

            # Accumulate losses for epoch average
            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            epoch_steps += 1
            global_step += 1

            # Log per-iteration
            if global_step % log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                log_metrics = {k: v for k, v in loss_dict.items()}
                log_metrics["lr"] = lr_now
                logger.log(log_metrics, step=global_step, phase="train_step")

        # Epoch summary
        epoch_avg = {k: v / epoch_steps for k, v in epoch_losses.items()}
        epoch_time = time.time() - epoch_start
        epoch_avg["epoch_time_s"] = epoch_time
        epoch_avg["lr"] = optimizer.param_groups[0]["lr"]
        logger.log(epoch_avg, step=epoch, phase="train_epoch")
        logger.info(f"Epoch {epoch}/{total_epochs-1} done in {epoch_time:.1f}s")

        # Checkpoint
        if (epoch + 1) % eval_every == 0 or epoch == total_epochs - 1:
            state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "epoch": epoch,
                "global_step": global_step,
                "cfg": cfg,
            }
            is_best = ckpt_manager.save(state, epoch, {"avg_loss": epoch_avg["total"]})
            if is_best:
                logger.info(f"  ★ New best checkpoint at epoch {epoch}")

    logger.info("Pretraining complete!")
    logger.close()


# =============================================================================
# Entry point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Embryo Pretraining")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    # Optional overrides
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)

    # Apply CLI overrides
    if args.batch_size:
        cfg["pretrain"]["batch_size"] = args.batch_size
    if args.epochs:
        cfg["pretrain"]["epochs"] = args.epochs
    if args.lr:
        cfg["pretrain"]["lr"] = args.lr
    if args.output_dir:
        cfg["pretrain"]["output_dir"] = args.output_dir

    train_pretrain(cfg, resume=args.resume)
