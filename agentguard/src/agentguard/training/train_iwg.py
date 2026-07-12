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
    norm_mean: Optional[np.ndarray] = None,
    norm_std: Optional[np.ndarray] = None,
    full_val_loader: Optional[torch.utils.data.DataLoader] = None,
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
        ``full_val_every``      0       Evaluate full validation loader every
                                        N epochs when provided. 0 disables it.
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
        "full_val_every": 0,
        "resume_from": "",
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
    best_metric_source = "val"
    best_metrics: dict[str, Any] = {}
    epochs_no_improve = 0
    train_metrics_history: list[dict[str, Any]] = []
    val_metrics_history: list[dict[str, Any]] = []
    epoch_times: list[float] = []
    global_step = 0

    metrics_file = os.path.join(output_dir, "metrics.jsonl")
    epoch_csv_path = os.path.join(output_dir, "epoch_metrics.csv")
    last_ckpt_path = os.path.join(output_dir, "iwg_last.pt")

    start_epoch = 0
    resume_from = str(cfg.get("resume_from") or "")
    if resume_from:
        resume_metadata = load_checkpoint(resume_from, model, optimizer, scheduler)
        start_epoch = int(resume_metadata.get("epoch", -1)) + 1
        global_step = start_epoch * steps_per_epoch
        logger.info(
            f"Resumed IWG from {resume_from} at epoch {start_epoch} "
            f"(checkpoint epoch={resume_metadata.get('epoch', '?')})."
        )
        best_ckpt_path = os.path.join(output_dir, "iwg_best.pt")
        if os.path.isfile(best_ckpt_path):
            best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            best_meta = best_ckpt.get("metadata", {})
            best_selection = best_meta.get("best_selection_metrics") or {}
            best_val_loss = float(
                best_selection.get(
                    "loss",
                    best_meta.get("full_val_metrics", best_meta.get("val_metrics", {})).get(
                        "loss", float("inf")
                    ),
                )
            )
            best_epoch = int(best_ckpt.get("epoch", -1)) + 1
            best_metric_source = str(best_meta.get("best_metric_source", "val"))
            best_metrics = dict(best_selection or best_meta.get("val_metrics", {}))
            logger.info(
                f"Existing best checkpoint: epoch {best_epoch} "
                f"({best_metric_source}_loss={best_val_loss:.6f})."
            )

    resume_mode = start_epoch > 0

    # Open metrics file for streaming.
    metrics_fh = open(metrics_file, "a" if resume_mode else "w")

    # CSV header.
    if not resume_mode or not os.path.isfile(epoch_csv_path):
        with open(epoch_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "val_loss", "gate_accuracy",
                "motion_mae", "appearance_mae", "policy_kl",
                "learning_rate", "epoch_time_s",
            ])

    train_tracker = MetricsTracker()

    # ------------------------------------------------------------------
    # Training loop.
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, cfg["epochs"]):
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
        val_metrics = _evaluate_iwg_loader(model, val_loader, device, use_amp)
        full_val_metrics = None
        if (
            full_val_loader is not None
            and int(cfg.get("full_val_every", 0)) > 0
            and (epoch + 1) % int(cfg["full_val_every"]) == 0
        ):
            full_val_metrics = _evaluate_iwg_loader(model, full_val_loader, device, use_amp)

        # ---- Epoch summary ----
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        train_metrics = train_tracker.compute()

        train_metrics["epoch"] = epoch + 1
        val_metrics["epoch"] = epoch + 1
        if full_val_metrics is not None:
            full_val_metrics["epoch"] = epoch + 1

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
        if full_val_metrics is not None:
            logger.info(
                f"Epoch {epoch + 1:2d} full-val — "
                f"loss={full_val_metrics.get('loss', 0):.4f}  "
                f"gate_acc={full_val_metrics.get('gate_accuracy', 0):.3f}  "
                f"motion_mae={full_val_metrics.get('motion_mae', 0):.4f}  "
                f"app_mae={full_val_metrics.get('appearance_mae', 0):.4f}"
            )

        # Write metrics.jsonl.
        metrics_line = {
            "epoch": epoch + 1,
            "train": {k: round(v, 6) if isinstance(v, float) else v for k, v in train_metrics.items()},
            "val": {k: round(v, 6) if isinstance(v, float) else v for k, v in val_metrics.items()},
            "learning_rate": current_lr,
            "epoch_time_s": round(epoch_time, 2),
        }
        if full_val_metrics is not None:
            metrics_line["full_val"] = {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in full_val_metrics.items()
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
        if full_val_loader is not None and int(cfg.get("full_val_every", 0)) > 0:
            selection_metrics = full_val_metrics
            selection_source = "full_val"
        else:
            selection_metrics = val_metrics
            selection_source = "val"
        selection_loss = (
            selection_metrics.get("loss", float("inf"))
            if selection_metrics is not None
            else None
        )

        # Save last checkpoint.
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
            norm_mean=norm_mean, norm_std=norm_std,
        )

        # Save best checkpoint.
        if selection_loss is not None and selection_loss < best_val_loss:
            best_val_loss = selection_loss
            best_epoch = epoch + 1
            best_metric_source = selection_source
            best_metrics = dict(selection_metrics)
            epochs_no_improve = 0
            best_ckpt_path = os.path.join(output_dir, "iwg_best.pt")
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                metadata={
                    "config": cfg,
                    "val_metrics": val_metrics,
                    "full_val_metrics": full_val_metrics,
                    "best_metric_source": best_metric_source,
                    "best_selection_metrics": best_metrics,
                    "train_metrics": train_metrics,
                    "reid_dim": cfg["reid_dim"],
                    "scalar_dim": cfg["scalar_dim"],
                    "event_dim": cfg["event_dim"],
                    "is_best": True,
                },
                path=best_ckpt_path,
                norm_mean=norm_mean, norm_std=norm_std,
            )
            logger.info(
                f"New best model ({selection_source}_loss={selection_loss:.6f}) "
                f"— saved to {best_ckpt_path}"
            )
        elif selection_loss is None:
            logger.info(
                "Best model selection skipped this epoch; waiting for full-val "
                f"every {int(cfg.get('full_val_every', 0))} epoch(s)."
            )
        else:
            epochs_no_improve += 1
            logger.info(
                f"No improvement for {epochs_no_improve} epoch(s) "
                f"(best {best_metric_source}_loss={best_val_loss:.6f} @ epoch {best_epoch})"
            )

        # Early stopping.
        if epochs_no_improve >= cfg["early_stop_patience"]:
            logger.info(
                f"Early stopping triggered after {epoch + 1} epochs. "
                f"Best epoch: {best_epoch} ({best_metric_source}_loss={best_val_loss:.6f})"
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
        "best_metric_source": best_metric_source,
        "total_epochs": len(train_metrics_history),
        "total_steps": global_step,
        "avg_epoch_time_s": float(np.mean(epoch_times)) if epoch_times else 0.0,
        "config": cfg,
        "best_metrics": best_metrics,
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


def _evaluate_iwg_loader(
    model: IWG,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, Any]:
    model.eval()
    tracker = MetricsTracker()
    tracker.reset()
    with torch.no_grad():
        for batch in loader:
            track_feats = batch["track_feats"].to(device, non_blocking=True)
            det_feats = batch["det_feats"].to(device, non_blocking=True)
            scalar_feats = batch["scalar_feats"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in batch["targets"].items()}

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(track_feats, det_feats, scalar_feats, mask)
                _, loss_components = _compute_iwg_loss(outputs, targets)

            tracker.update(
                loss_dict=loss_components,
                preds={"gate": outputs["gate"], "policy_probs": outputs["policy_probs"]},
                targets=targets,
            )
    return tracker.compute()


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

    valid_motion = targets.get("valid_motion", torch.ones_like(targets["motion_target"], dtype=torch.bool))
    valid_appearance = targets.get("valid_appearance", torch.ones_like(targets["appearance_target"], dtype=torch.bool))
    gate_valid = torch.stack([valid_motion, valid_appearance], dim=-1).float()
    gate_bce = F.binary_cross_entropy(
        pred_gate.clamp(1e-6, 1 - 1e-6), target_gate, reduction="none"
    )
    gate_loss_per_sample = (
        (gate_bce * gate_valid).sum(dim=-1)
        / gate_valid.sum(dim=-1).clamp(min=1.0)
    )
    valid_any = gate_valid.sum(dim=-1) > 0

    pred_log_policy = torch.clamp(pred_policy, min=1e-8).log()
    target_policy_dist = torch.clamp(target_policy, min=1e-8)
    policy_kl_per_sample = F.kl_div(
        pred_log_policy, target_policy_dist, reduction='none'
    ).sum(dim=-1)  # (B,)
    policy_valid = (valid_motion & valid_appearance).float()

    brier_per_sample = (
        ((pred_gate - target_gate) ** 2 * gate_valid).sum(dim=-1)
        / gate_valid.sum(dim=-1).clamp(min=1.0)
    )

    cue_target = torch.stack(
        [
            targets.get("motion_label_confidence", torch.zeros_like(targets["motion_target"])),
            targets.get("appearance_label_confidence", torch.zeros_like(targets["appearance_target"])),
            torch.maximum(
                targets.get("motion_label_confidence", torch.zeros_like(targets["motion_target"])),
                targets.get("appearance_label_confidence", torch.zeros_like(targets["appearance_target"])),
            ),
        ],
        dim=-1,
    )
    cue_bce_per_sample = F.binary_cross_entropy(
        outputs["cue"].clamp(1e-6, 1 - 1e-6),
        cue_target,
        reduction="none",
    ).mean(dim=-1)

    loss_per_sample = (
        gate_loss_per_sample
        + 0.1 * policy_kl_per_sample * policy_valid
        + 0.1 * brier_per_sample
        + 0.2 * cue_bce_per_sample
    )

    # Apply sample weights
    sample_weight = targets.get("sample_weight", torch.ones_like(loss_per_sample)) * valid_any.float()
    weighted_loss = (loss_per_sample * sample_weight).sum() / sample_weight.sum().clamp(min=1e-8)

    return weighted_loss, {
        "loss": weighted_loss.detach(),
        "gate_loss": (gate_loss_per_sample * sample_weight).sum() / sample_weight.sum().clamp(min=1).detach(),
        "policy_loss": (policy_kl_per_sample * policy_valid).sum().detach() / policy_valid.sum().clamp(min=1.0),
        "brier_loss": (brier_per_sample * sample_weight).sum().detach() / sample_weight.sum().clamp(min=1.0),
        "cue_loss": (cue_bce_per_sample * sample_weight).sum().detach() / sample_weight.sum().clamp(min=1.0),
    }
