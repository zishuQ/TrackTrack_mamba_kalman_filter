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
    A flat dictionary with the current Runtime keys ``mode``,
    ``rg_cma_output``, ``rg_cma_alpha``, ``disable_kf_gate``,
    ``disable_ema_gate``, ``fallback_threshold``,
    ``return_attention_diagnostics``, ``rg_cma_max_gap``, and ``device``.
    Each value defaults to a safe fallback when the corresponding attribute
    is absent from *args*.
    """
    return {
        "mode": getattr(args, "agentguard_mode", "off"),
        "rg_cma_output": getattr(args, "rg_cma_output", "final"),
        "rg_cma_alpha": getattr(args, "rg_cma_alpha", 1.0),
        "disable_kf_gate": getattr(args, "agentguard_disable_kf_gate", False),
        "disable_ema_gate": getattr(args, "agentguard_disable_ema_gate", False),
        "fallback_threshold": getattr(args, "agentguard_fallback_threshold", 0.0),
        "return_attention_diagnostics": getattr(
            args, "agentguard_attention_diagnostics", False
        ),
        "rg_cma_max_gap": 30,
        "device": getattr(args, "agentguard_device", "cpu"),
    }
