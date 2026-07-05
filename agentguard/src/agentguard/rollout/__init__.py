from agentguard.rollout.motion import (
    compute_motion_benefit,
    compute_motion_rollout,
)
from agentguard.rollout.appearance import (
    compute_appearance_benefit,
    ema_update,
)
from agentguard.rollout.window import (
    compute_tgr_window_labels,
    generate_window_augmentations,
)
from agentguard.rollout.losses import (
    iou_loss,
    l1_normalized_loss,
    motion_frame_loss,
    appearance_loss,
    sigmoid,
)
from agentguard.labels import (
    compute_dataset_stats,
    compute_soft_target as compute_soft_targets,
)

__all__ = [
    # Motion
    "compute_motion_benefit",
    "compute_motion_rollout",
    # Appearance
    "compute_appearance_benefit",
    "ema_update",
    # Window / TGR
    "compute_tgr_window_labels",
    "generate_window_augmentations",
    # Losses
    "iou_loss",
    "l1_normalized_loss",
    "motion_frame_loss",
    "appearance_loss",
    "sigmoid",
    # Labels (re-exported)
    "compute_dataset_stats",
    "compute_soft_targets",
]
