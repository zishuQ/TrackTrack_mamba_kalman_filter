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
    ``rg_cma_output``, ``rg_cma_alpha``, ``rg_cma_max_gap``, and ``device``. Each value
    defaults to a safe fallback when the corresponding attribute is absent
    from *args*.
    """
    return {
        "mode": getattr(args, "agentguard_mode", "off"),
        "rg_cma_output": getattr(args, "rg_cma_output", "final"),
        "rg_cma_alpha": getattr(args, "rg_cma_alpha", 1.0),
        "rg_cma_max_gap": 30,
        "device": getattr(args, "agentguard_device", "cpu"),
    }
