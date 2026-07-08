from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agentguard.models.iwg import IWG
from agentguard.models.tgr import TGR
from agentguard.training.checkpointing import load_checkpoint, save_checkpoint
from agentguard.training.logging_utils import log_metrics, save_training_summary, setup_logging
from agentguard.training.metrics import MetricsTracker
from agentguard.training.optim import build_optimizer, build_scheduler


def train_tgr(
    tgr_model: TGR,
    frozen_iwg: IWG,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
    output_dir: str,
    norm_mean: Optional[np.ndarray] = None,
    norm_std: Optional[np.ndarray] = None,
    full_val_loader: Optional[torch.utils.data.DataLoader] = None,
) -> Dict[str, Any]:
    """Train a TGR model with a frozen IWG model.

    Only the TGR model's parameters are trained; the IWG ``EventEncoder``
    and gate network are kept frozen.

    Parameters
    ----------
    tgr_model : TGR
        The TGR model to train.
    frozen_iwg : IWG
        Pre-trained IWG model in evaluation mode (frozen).
    train_loader : DataLoader
        Training data loader (yields windows of events).
    val_loader : DataLoader
        Validation data loader.
    config : dict
        Training configuration.  Defaults are applied for missing keys:

        ======================  ======  =====================================
        Key                     Default  Description
        ======================  ======  =====================================
        ``epochs``              30      Number of training epochs.
        ``batch_size``          128     Batch size (for reference).
        ``learning_rate``       2e-4    Peak learning rate.
        ``weight_decay``        1e-4    Weight decay.
        ``warmup``              1       Warmup duration in **epochs**.
        ``gradient_clip_norm``  5.0     Max gradient norm for clipping.
        ``amp``                 True    Enable automatic mixed precision.
        ``seed``                42      Random seed.
        ``early_stop_patience`` 5       Epochs without val improvement to
                                        stop training.
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
        "batch_size": 128,
        "learning_rate": 2e-4,
        "weight_decay": 1e-4,
        "warmup": 1,
        "gradient_clip_norm": 5.0,
        "amp": True,
        "seed": 42,
        "early_stop_patience": 5,
        "full_val_every": 0,
        "full_val_device": "",
        "resume_from": "",
    }
    cfg.update(config)

    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)
    logger.info("=" * 60)
    logger.info("TGR Training — start")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Config: {json.dumps(cfg, indent=2, default=str)}")

    # Reproducibility.
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])

    device = next(tgr_model.parameters()).device
    logger.info(f"Device: {device}")

    # ------------------------------------------------------------------
    # Freeze IWG.
    # ------------------------------------------------------------------
    frozen_iwg.eval()
    for param in frozen_iwg.parameters():
        param.requires_grad = False
    logger.info("IWG model frozen.")

    # Move IWG to the same device as TGR.
    frozen_iwg.to(device)

    # ------------------------------------------------------------------
    # Optimizer & scheduler (TGR only).
    # ------------------------------------------------------------------
    optimizer = build_optimizer(
        tgr_model, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
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
    start_epoch = 0

    resume_from = str(cfg.get("resume_from") or "")
    if resume_from:
        resume_metadata = load_checkpoint(resume_from, tgr_model, optimizer, scheduler)
        start_epoch = int(resume_metadata.get("epoch", -1)) + 1
        global_step = start_epoch * steps_per_epoch
        logger.info(
            f"Resumed TGR from {resume_from} at epoch {start_epoch} "
            f"(continuing to {cfg['epochs']})."
        )

        best_ckpt_path = os.path.join(output_dir, "tgr_best.pt")
        if os.path.isfile(best_ckpt_path):
            best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            best_meta = best_ckpt.get("metadata", {})
            best_selection = best_meta.get("best_selection_metrics") or {}
            best_val_loss = float(
                best_selection.get(
                    "loss",
                    best_meta.get("full_val_metrics", best_meta.get("val_metrics", {})).get(
                        "loss",
                        float("inf"),
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

    metrics_file = os.path.join(output_dir, "metrics.jsonl")
    csv_path = os.path.join(output_dir, "epoch_metrics.csv")

    resume_mode = start_epoch > 0
    metrics_fh = open(metrics_file, "a" if resume_mode else "w")

    import csv
    if not resume_mode or not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "val_loss", "gate_accuracy",
                "motion_mae", "appearance_mae", "learning_rate", "epoch_time_s",
            ])

    train_tracker = MetricsTracker()

    # ------------------------------------------------------------------
    # Training loop.
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, cfg["epochs"]):
        epoch_start = time.time()
        logger.info(f"--- Epoch {epoch + 1}/{cfg['epochs']} ---")

        # ---- Training ----
        tgr_model.train()
        train_tracker.reset()

        for batch_idx, batch in enumerate(train_loader):
            # Move inputs to device.
            track_feats = batch["track_feats"].to(device, non_blocking=True)
            det_feats = batch["det_feats"].to(device, non_blocking=True)
            scalar_feats = batch["scalar_feats"].to(device, non_blocking=True)
            target_gate = batch["target_gate"].to(device, non_blocking=True)
            sample_weight = batch["sample_weight"].to(device, non_blocking=True)
            has_detection_mask = batch["has_detection_mask"].to(device, non_blocking=True)
            valid_motion = batch.get("valid_motion", has_detection_mask).to(device, non_blocking=True)
            valid_appearance = batch.get("valid_appearance", has_detection_mask).to(device, non_blocking=True)

            # Use pre-computed IWG outputs from the data loader.
            # These were stored during data collection (per-event iwg_policy_probs,
            # iwg_gate fields) and are already aligned per timestep.
            iwg_policy_probs = batch["iwg_policy_probs"].to(device, non_blocking=True)
            iwg_gates = batch["iwg_gates"].to(device, non_blocking=True)

            # The frozen IWG is not called during TGR training — the TGR dataset
            # carries pre-computed IWG outputs for each event.  This avoids the
            # dimension mismatch (IWG produces a *single* output for the whole
            # sequence, while TGR needs per-timestep values).

            optimizer.zero_grad(set_to_none=True)

            # Forward TGR.
            with torch.amp.autocast("cuda", enabled=use_amp):
                g_revised, gate_residual = tgr_model(
                    track_feats, det_feats, scalar_feats,
                    iwg_policy_probs, iwg_gates,
                    has_detection_mask, padding_mask=None,
                )
                loss, loss_components = _compute_tgr_loss(
                    g_revised, gate_residual, target_gate, has_detection_mask,
                    sample_weight, valid_motion, valid_appearance,
                )

            # Backward.
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    tgr_model.parameters(), cfg["gradient_clip_norm"]
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    tgr_model.parameters(), cfg["gradient_clip_norm"]
                )
                optimizer.step()

            scheduler.step()
            global_step += 1

            # Track training metrics.
            train_tracker.update(
                loss_dict={**loss_components, "grad_norm": grad_norm},
                preds={"gate": g_revised},
                targets={
                    "motion_target": target_gate[..., 0],
                    "appearance_target": target_gate[..., 1],
                    "valid_motion": valid_motion,
                    "valid_appearance": valid_appearance,
                    "sample_weight": sample_weight,
                },
            )

            # Step-level logging.
            if batch_idx % max(1, len(train_loader) // 5) == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                log_metrics(
                    {"loss": loss_components["loss"], "tgr_loss": loss_components["tgr_bce"], "lr": current_lr},
                    step=global_step,
                    epoch=epoch + 1,
                )

        # ---- Validation ----
        val_metrics = _evaluate_tgr_loader(tgr_model, val_loader, device, use_amp)
        full_val_metrics = None
        if (
            full_val_loader is not None
            and int(cfg.get("full_val_every", 0)) > 0
            and (epoch + 1) % int(cfg["full_val_every"]) == 0
        ):
            full_val_device_name = str(cfg.get("full_val_device") or "").strip()
            full_val_device = torch.device(full_val_device_name) if full_val_device_name else device
            original_device = device
            if full_val_device != original_device:
                logger.info(f"Running TGR full-val on {full_val_device}.")
                tgr_model.to(full_val_device)
            try:
                full_val_metrics = _evaluate_tgr_loader(
                    tgr_model,
                    full_val_loader,
                    full_val_device,
                    use_amp and full_val_device.type == "cuda",
                )
            finally:
                if full_val_device != original_device:
                    tgr_model.to(original_device)

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
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f"{train_metrics.get('loss', 0):.6f}",
                f"{val_metrics.get('loss', 0):.6f}",
                f"{val_metrics.get('gate_accuracy', 0):.4f}",
                f"{val_metrics.get('motion_mae', 0):.6f}",
                f"{val_metrics.get('appearance_mae', 0):.6f}",
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
        last_ckpt_path = os.path.join(output_dir, "tgr_last.pt")
        save_checkpoint(
            tgr_model, optimizer, scheduler, epoch,
            metadata={
                "config": cfg,
                "val_metrics": val_metrics,
                "train_metrics": train_metrics,
                "reid_dim": cfg.get("reid_dim"),
                "scalar_dim": cfg.get("scalar_dim", 63),
                "event_dim": cfg.get("event_dim", 128),
            },
            path=last_ckpt_path,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        # Save best checkpoint.
        if selection_loss is not None and selection_loss < best_val_loss:
            best_val_loss = selection_loss
            best_epoch = epoch + 1
            best_metric_source = selection_source
            best_metrics = dict(selection_metrics)
            epochs_no_improve = 0
            best_ckpt_path = os.path.join(output_dir, "tgr_best.pt")
            save_checkpoint(
                tgr_model, optimizer, scheduler, epoch,
                metadata={
                    "config": cfg,
                    "val_metrics": val_metrics,
                    "full_val_metrics": full_val_metrics,
                    "best_metric_source": best_metric_source,
                    "best_selection_metrics": best_metrics,
                    "train_metrics": train_metrics,
                    "is_best": True,
                    "reid_dim": cfg.get("reid_dim"),
                    "scalar_dim": cfg.get("scalar_dim", 63),
                    "event_dim": cfg.get("event_dim", 128),
                },
                path=best_ckpt_path,
                norm_mean=norm_mean,
                norm_std=norm_std,
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

    best_ckpt_path = os.path.join(output_dir, "tgr_best.pt")
    if os.path.isfile(best_ckpt_path):
        best_metadata = load_checkpoint(best_ckpt_path, tgr_model)
        logger.info(f"Restored best model from epoch {best_metadata.get('epoch', '?')}")

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
            "best": os.path.join(output_dir, "tgr_best.pt"),
            "last": os.path.join(output_dir, "tgr_last.pt"),
        },
    }

    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(training_summary, f, indent=2, default=str)
    logger.info(f"Training summary saved to {summary_path}")

    history_path = os.path.join(output_dir, "metrics_history.json")
    save_training_summary(
        [{"epoch": e, "split": "train", **m} for e, m in zip(range(1, len(train_metrics_history) + 1), train_metrics_history)]
        + [{"epoch": e, "split": "val", **m} for e, m in zip(range(1, len(val_metrics_history) + 1), val_metrics_history)],
        history_path,
    )

    logger.info("TGR training complete.")
    logger.info("=" * 60)
    return training_summary


def _evaluate_tgr_loader(
    tgr_model: TGR,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, Any]:
    tgr_model.eval()
    tracker = MetricsTracker()
    tracker.reset()
    with torch.no_grad():
        for batch in loader:
            track_feats = batch["track_feats"].to(device, non_blocking=True)
            det_feats = batch["det_feats"].to(device, non_blocking=True)
            scalar_feats = batch["scalar_feats"].to(device, non_blocking=True)
            target_gate = batch["target_gate"].to(device, non_blocking=True)
            sample_weight = batch["sample_weight"].to(device, non_blocking=True)
            has_detection_mask = batch["has_detection_mask"].to(device, non_blocking=True)
            valid_motion = batch.get("valid_motion", has_detection_mask).to(device, non_blocking=True)
            valid_appearance = batch.get("valid_appearance", has_detection_mask).to(device, non_blocking=True)
            iwg_policy_probs = batch["iwg_policy_probs"].to(device, non_blocking=True)
            iwg_gates = batch["iwg_gates"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                g_revised, gate_residual = tgr_model(
                    track_feats,
                    det_feats,
                    scalar_feats,
                    iwg_policy_probs,
                    iwg_gates,
                    has_detection_mask,
                    padding_mask=None,
                )
                _, loss_components = _compute_tgr_loss(
                    g_revised,
                    gate_residual,
                    target_gate,
                    has_detection_mask,
                    sample_weight,
                    valid_motion,
                    valid_appearance,
                )

            tracker.update(
                loss_dict=loss_components,
                preds={"gate": g_revised},
                targets={
                    "motion_target": target_gate[..., 0],
                    "appearance_target": target_gate[..., 1],
                    "valid_motion": valid_motion,
                    "valid_appearance": valid_appearance,
                    "sample_weight": sample_weight,
                },
            )
    return tracker.compute()


# ------------------------------------------------------------------
# Loss computation
# ------------------------------------------------------------------

def _compute_tgr_loss(
    g_revised: torch.Tensor,
    gate_residual: torch.Tensor,
    target_gate: torch.Tensor,
    has_detection_mask: torch.Tensor,
    sample_weight: torch.Tensor,
    valid_motion: torch.Tensor | None = None,
    valid_appearance: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the TGR training loss.

    Loss components:

    * ``tgr_bce``: Masked binary cross-entropy between the revised gates
      and target gates (only where ``has_detection_mask`` is ``True``).
    * ``delta_penalty``: L2 penalty on the gate residual magnitude
      (encourages minimal revision).

    ``L_TGR = L_BCE + 0.02 * |delta_gate|^2``

    The final loss is averaged over the `(B, seq_len)` positions that have
    a detection (``has_detection_mask == True``).
    """
    # Clamp predictions for numerical stability.
    g_clipped = torch.clamp(g_revised, 1e-6, 1.0 - 1e-6)

    # Masked BCE.
    if valid_motion is None:
        valid_motion = has_detection_mask
    if valid_appearance is None:
        valid_appearance = has_detection_mask
    gate_valid = torch.stack([valid_motion, valid_appearance], dim=-1).float()
    gate_valid = gate_valid * has_detection_mask.unsqueeze(-1).float()
    bce = F.binary_cross_entropy(g_clipped, target_gate, reduction="none")  # (B, seq_len, 2)
    bce = (bce * gate_valid).sum(dim=-1) / gate_valid.sum(dim=-1).clamp(min=1.0)
    mask = (gate_valid.sum(dim=-1) > 0).float()
    masked_bce = (bce * mask * sample_weight).sum() / (mask * sample_weight).sum().clamp(min=1.0)
    # Avoid NaN when mask is all-zero.
    if torch.isnan(masked_bce) or torch.isinf(masked_bce):
        masked_bce = torch.tensor(0.0, device=g_revised.device)

    # Delta penalty (L2 on gate_residual, averaged over valid positions).
    delta_sq = gate_residual.pow(2).mean(dim=-1)  # (B, seq_len)
    delta_penalty = (delta_sq * mask).sum() / mask.sum().clamp(min=1.0)
    if torch.isnan(delta_penalty) or torch.isinf(delta_penalty):
        delta_penalty = torch.tensor(0.0, device=g_revised.device)

    loss = masked_bce + 0.02 * delta_penalty

    return loss, {
        "loss": loss.detach(),
        "tgr_bce": masked_bce.detach(),
        "delta_penalty": delta_penalty.detach(),
    }
