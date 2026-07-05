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
        "device": getattr(args, "agentguard_device", "cpu"),
    }
