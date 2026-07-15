"""Bridge between TrackTrack's argparse Namespace and AgentGuard configuration."""


def build_runtime_config(args) -> dict:
    """Extract AgentGuard-relevant configuration from an ``argparse.Namespace``.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command-line arguments used by TrackTrack.

    Returns
    -------
    dict
        A flat dictionary with keys ``mode``, ``iwg_checkpoint``,
        ``tgr_checkpoint``, and ``device``.  Each value defaults to a safe
        fallback when the corresponding attribute is absent from *args*.
    """
    return {
        "mode": getattr(args, "agentguard_mode", "off"),
        "iwg_checkpoint": getattr(args, "iwg_checkpoint", None),
        "tgr_checkpoint": getattr(args, "tgr_checkpoint", None),
        "agentguard_checkpoint": getattr(args, "agentguard_checkpoint", None),
        "joint_output": getattr(args, "joint_output", "final"),
        "iwg_attn_output": getattr(args, "iwg_attn_output", "final"),
        # Joint runtime injects this from the strictly validated checkpoint.
        "joint_window_size": None,
        "joint_max_frame_gap": 30,
        "iwg_attn_max_frame_gap": 30,
        "device": getattr(args, "agentguard_device", "cpu"),
        "replay_diff_threshold": float(
            getattr(args, "agentguard_replay_diff_threshold", 0.0) or 0.0
        ),
        "tgr_frame_stride": max(
            int(getattr(args, "agentguard_tgr_frame_stride", 1) or 1),
            1,
        ),
    }
