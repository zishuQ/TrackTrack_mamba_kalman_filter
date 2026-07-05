from __future__ import annotations

import csv
import json
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agentguard.models.iwg import IWG
from agentguard.training.checkpointing import load_checkpoint, save_checkpoint
from agentguard.training.logging_utils import log_metrics, save_training_summary, setup_logging
from agentguard.training.metrics import MetricsTracker
from agentguard.training.optim import build_optimizer, build_scheduler


def train_iwg(
    model: IWG,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    """Train an IWG model.

    Parameters
    ----------
    model : IWG
        The IWG model to train.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    config : dict
        Training configuration.  Defaults are applied for missing keys:

        ======================  ======  =====================================
        Key                     Default  Description
        ======================  ======  =====================================
        ``epochs``              30      Number of training epochs.
        ``batch_size``          256     Batch size (for reference).
        ``optimizer``           AdamW   Optimiser type.
        ``learning_rate``       3e-4    Peak learning rate.
        ``weight_decay``        1e-4    Weight decay.
        ``warmup``              1       Warmup duration in **epochs**.
        ``scheduler``           cosine  LR scheduler type.
        ``gradient_clip_norm``  5.0     Max gradient norm for clipping.
        ``amp``                 True    Enable automatic mixed precision.
        ``seed``                42      Random seed.
        ``early_stop_patience`` 5       Epochs without val improvement to
                                        stop training.
        ``reid_dim``            (req)   ReID feature dimensionality.
        ``scalar_dim``          63      Scalar feature dimensionality.
        ``event_dim``           128     Event embedding dimensionality.
        ======================  ======  =====================================

    output_dir : str
        Directory where outputs (checkpoints, logs, metrics) are saved.

    Returns
    -------
    dict
        Training summary containing best epoch, final metrics, and paths.
    """
    # ------------------------------------------------------------------
    # Resolve configuration defaults.
    # ------------------------------------------------------------------
    cfg = {
        "epochs": 30,
        "batch_size": 256,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "warmup": 1,
        "gradient_clip_norm": 5.0,
        "amp": True,
        "seed": 42,
        "early_stop_patience": 5,
        "reid_dim": model.encoder.reid_dim,
        "scalar_dim": 63,
        "event_dim": 128,
    }
    cfg.update(config)

    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)
    logger.info("=" * 60)
    logger.info("IWG Training — start")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Config: {json.dumps(cfg, indent=2, default=str)}")

    # Reproducibility.
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])

    device = next(model.parameters()).device
    logger.info(f"Device: {device}")

    # ------------------------------------------------------------------
    # Optimizer & scheduler.
    # ------------------------------------------------------------------
    optimizer = build_optimizer(model, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = max(len(train_loader), 1)
    num_training_steps = steps_per_epoch * cfg["epochs"]
    warmup_steps = steps_per_epoch * cfg["warmup"]
    scheduler = build_scheduler(optimizer, num_training_steps, warmup_steps)

    # ------------------------------------------------------------------
    # AMP scaler.
    # ------------------------------------------------------------------
    use_amp = cfg["amp"] and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        logger.info("Automatic Mixed Precision (AMP) enabled.")

    # ------------------------------------------------------------------
    # Training state.
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    train_metrics_history: list[dict[str, Any]] = []
    val_metrics_history: list[dict[str, Any]] = []
    epoch_times: list[float] = []
    global_step = 0

    metrics_file = os.path.join(output_dir, "metrics.jsonl")
    epoch_csv_path = os.path.join(output_dir, "epoch_metrics.csv")

    # Open metrics file for streaming.
    metrics_fh = open(metrics_file, "w")

    # CSV header.
    with open(epoch_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "val_loss", "gate_accuracy",
            "motion_mae", "appearance_mae", "policy_kl",
            "learning_rate", "epoch_time_s",
        ])

    train_tracker = MetricsTracker()
    val_tracker = MetricsTracker()

    # ------------------------------------------------------------------
    # Training loop.
    # ------------------------------------------------------------------
    for epoch in range(cfg["epochs"]):
        epoch_start = time.time()
        logger.info(f"--- Epoch {epoch + 1}/{cfg['epochs']} ---")

        # ---- Training ----
        model.train()
        train_tracker.reset()

        for batch_idx, batch in enumerate(train_loader):
            # Move to device.
            track_feats = batch["track_feats"].to(device, non_blocking=True)
            det_feats = batch["det_feats"].to(device, non_blocking=True)
            scalar_feats = batch["scalar_feats"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in batch["targets"].items()}

            optimizer.zero_grad(set_to_none=True)

            # Forward with optional AMP.
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(track_feats, det_feats, scalar_feats, mask)
                loss, loss_components = _compute_iwg_loss(outputs, targets)

            # Backward.
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["gradient_clip_norm"]
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["gradient_clip_norm"]
                )
                optimizer.step()

            scheduler.step()
            global_step += 1

            # Track metrics.
            train_tracker.update(
                loss_dict={**loss_components, "grad_norm": grad_norm},
                preds={"gate": outputs["gate"], "policy_probs": outputs["policy_probs"]},
                targets=targets,
            )

            # Step-level logging.
            if batch_idx % max(1, len(train_loader) // 5) == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                log_metrics(
                    {
                        "loss": loss_components["loss"],
                        "gate_loss": loss_components["gate_loss"],
                        "lr": current_lr,
                    },
                    step=global_step,
                    epoch=epoch + 1,
                )

        # ---- Validation ----
        model.eval()
        val_tracker.reset()

        with torch.no_grad():
            for batch in val_loader:
                track_feats = batch["track_feats"].to(device, non_blocking=True)
                det_feats = batch["det_feats"].to(device, non_blocking=True)
                scalar_feats = batch["scalar_feats"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                targets = {k: v.to(device, non_blocking=True) for k, v in batch["targets"].items()}

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(track_feats, det_feats, scalar_feats, mask)
                    _, loss_components = _compute_iwg_loss(outputs, targets)

                val_tracker.update(
                    loss_dict=loss_components,
                    preds={"gate": outputs["gate"], "policy_probs": outputs["policy_probs"]},
                    targets=targets,
                )

        # ---- Epoch summary ----
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        train_metrics = train_tracker.compute()
        val_metrics = val_tracker.compute()

        train_metrics["epoch"] = epoch + 1
        val_metrics["epoch"] = epoch + 1

        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)

        current_lr = optimizer.param_groups[0]["lr"]

        # Log to console.
        logger.info(
            f"Epoch {epoch + 1:2d} — "
            f"train_loss={train_metrics.get('loss', 0):.4f}  "
            f"val_loss={val_metrics.get('loss', 0):.4f}  "
            f"gate_acc={val_metrics.get('gate_accuracy', 0):.3f}  "
            f"motion_mae={val_metrics.get('motion_mae', 0):.4f}  "
            f"app_mae={val_metrics.get('appearance_mae', 0):.4f}  "
            f"lr={current_lr:.2e}  "
            f"time={epoch_time:.1f}s"
        )

        # Write metrics.jsonl.
        metrics_line = {
            "epoch": epoch + 1,
            "train": {k: round(v, 6) if isinstance(v, float) else v for k, v in train_metrics.items()},
            "val": {k: round(v, 6) if isinstance(v, float) else v for k, v in val_metrics.items()},
            "learning_rate": current_lr,
            "epoch_time_s": round(epoch_time, 2),
        }
        metrics_fh.write(json.dumps(metrics_line) + "\n")
        metrics_fh.flush()

        # Write epoch_metrics.csv.
        with open(epoch_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f"{train_metrics.get('loss', 0):.6f}",
                f"{val_metrics.get('loss', 0):.6f}",
                f"{val_metrics.get('gate_accuracy', 0):.4f}",
                f"{val_metrics.get('motion_mae', 0):.6f}",
                f"{val_metrics.get('appearance_mae', 0):.6f}",
                f"{val_metrics.get('policy_kl', 0):.6f}",
                f"{current_lr:.8f}",
                f"{epoch_time:.2f}",
            ])

        # ---- Checkpointing & early stopping ----
        val_loss = val_metrics.get("loss", float("inf"))

        # Save last checkpoint.
        last_ckpt_path = os.path.join(output_dir, "iwg_last.pt")
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            metadata={
                "config": cfg,
                "val_metrics": val_metrics,
                "train_metrics": train_metrics,
                "reid_dim": cfg["reid_dim"],
                "scalar_dim": cfg["scalar_dim"],
                "event_dim": cfg["event_dim"],
            },
            path=last_ckpt_path,
        )

        # Save best checkpoint.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            best_ckpt_path = os.path.join(output_dir, "iwg_best.pt")
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                metadata={
                    "config": cfg,
                    "val_metrics": val_metrics,
                    "train_metrics": train_metrics,
                    "reid_dim": cfg["reid_dim"],
                    "scalar_dim": cfg["scalar_dim"],
                    "event_dim": cfg["event_dim"],
                    "is_best": True,
                },
                path=best_ckpt_path,
            )
            logger.info(f"New best model (val_loss={val_loss:.6f}) — saved to {best_ckpt_path}")
        else:
            epochs_no_improve += 1
            logger.info(
                f"No improvement for {epochs_no_improve} epoch(s) "
                f"(best val_loss={best_val_loss:.6f} @ epoch {best_epoch})"
            )

        # Early stopping.
        if epochs_no_improve >= cfg["early_stop_patience"]:
            logger.info(
                f"Early stopping triggered after {epoch + 1} epochs. "
                f"Best epoch: {best_epoch} (val_loss={best_val_loss:.6f})"
            )
            break

    # ------------------------------------------------------------------
    # Cleanup and summary.
    # ------------------------------------------------------------------
    metrics_fh.close()

    # Load best model for summary.
    best_ckpt_path = os.path.join(output_dir, "iwg_best.pt")
    if os.path.isfile(best_ckpt_path):
        best_metadata = load_checkpoint(best_ckpt_path, model)
        logger.info(f"Restored best model from epoch {best_metadata.get('epoch', '?')}")
    else:
        best_metadata = {"epoch": best_epoch}

    # Build training summary.
    training_summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_epochs": len(train_metrics_history),
        "total_steps": global_step,
        "avg_epoch_time_s": float(np.mean(epoch_times)) if epoch_times else 0.0,
        "config": cfg,
        "best_metrics": val_metrics_history[best_epoch - 1] if best_epoch > 0 and best_epoch <= len(val_metrics_history) else {},
        "output_dir": output_dir,
        "checkpoints": {
            "best": best_ckpt_path,
            "last": last_ckpt_path,
        },
    }

    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(training_summary, f, indent=2, default=str)
    logger.info(f"Training summary saved to {summary_path}")

    # Save full metrics history separately.
    history_path = os.path.join(output_dir, "metrics_history.json")
    save_training_summary(
        [{"epoch": e, "split": "train", **m} for e, m in zip(range(1, len(train_metrics_history) + 1), train_metrics_history)]
        + [{"epoch": e, "split": "val", **m} for e, m in zip(range(1, len(val_metrics_history) + 1), val_metrics_history)],
        history_path,
    )

    logger.info("IWG training complete.")
    logger.info("=" * 60)
    return training_summary


# ------------------------------------------------------------------
# Loss computation
# ------------------------------------------------------------------

def _compute_iwg_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the IWG training loss.

    Loss components:

    * ``gate_loss``: binary cross-entropy on gating predictions.
    * ``policy_loss``: KL divergence between target policy distribution and
      predicted policy probabilities.
    * ``brier_loss``: Brier score (MSE) between predicted and target gates.

    The total loss is:

    ``L_total = L_gate + 0.1 * L_policy + 0.1 * L_brier``
    """
    pred_gate = outputs["gate"]                     # (B, 2)
    pred_policy = outputs["policy_probs"]            # (B, 5)

    # Build target gate tensor from motion and appearance targets.
    target_gate = torch.stack(
        [targets["motion_target"], targets["appearance_target"]], dim=-1
    )                                               # (B, 2)
    target_policy = targets["policy_soft_target"]    # (B, 5)
    valid_mask = torch.stack(
        [targets["valid_motion"], targets["valid_appearance"]], dim=-1
    )                                               # (B, 2)

    # Gate BCE loss (only on valid entries).
    pred_gate_clipped = torch.clamp(pred_gate, 1e-6, 1.0 - 1e-6)
    gate_bce = F.binary_cross_entropy(
        pred_gate_clipped, target_gate, reduction="none"
    )                                               # (B, 2)
    gate_loss = (gate_bce * valid_mask.float()).sum() / valid_mask.float().sum().clamp(min=1)

    # Policy KL divergence.
    policy_kl = F.kl_div(
        torch.clamp(pred_policy, min=1e-8).log(),
        torch.clamp(target_policy, min=1e-8),
        reduction="batchmean",
        log_target=False,
    )

    # Brier score (MSE on gates).
    brier_loss = F.mse_loss(pred_gate, target_gate, reduction="mean")

    # Weighted total.
    sample_weight = targets.get("sample_weight", torch.ones_like(target_gate[:, 0]))
    loss = gate_loss + 0.1 * policy_kl + 0.1 * brier_loss
    loss = (loss * sample_weight.detach()).mean()

    return loss, {
        "loss": loss.detach(),
        "gate_loss": gate_loss.detach(),
        "policy_loss": policy_kl.detach(),
        "brier_loss": brier_loss.detach(),
    }
