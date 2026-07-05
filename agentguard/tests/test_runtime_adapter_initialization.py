"""Tests for runtime/adapter initialisation invariants."""

import numpy as np
import pytest


def test_runtime_event_sink_is_none_by_default():
    from agentguard.runtime.manager import AgentGuardRuntime

    runtime = AgentGuardRuntime({"mode": "off"})
    assert runtime.event_sink is None


def test_runtime_feature_builder_initialised_once():
    from agentguard.runtime.manager import AgentGuardRuntime

    runtime = AgentGuardRuntime({"mode": "off"})
    runtime.init_feature_builder(reid_dim=64)

    fb = runtime.feature_builder
    assert fb is not None
    assert runtime._feature_builder_initialised is True

    # Second call should be a no-op
    runtime.init_feature_builder(reid_dim=128)
    assert runtime.feature_builder is fb
    assert int(fb.reid_dim) == 64


def test_adapter_uses_getattr_for_event_sink():
    class FakeRuntime:
        event_sink = "test_sink"
        mode = "off"
        stats = None
        event_buffers = {}
        checkpoints = None
        iwg = None
        feature_builder = None

        def init_feature_builder(self, *a, **kw):
            pass

        def init_motion_model(self):
            pass

        def finalize_first_stage(self, *a, **kw):
            return {}

        def get_or_create_buffer(self, tid):
            return (None, None)

        def cleanup_track(self, tid):
            pass

        def run_iwg_inference(self, seq):
            return {
                "policy_probs": np.ones(5) / 5,
                "gate": np.ones(2),
                "event_logits": np.zeros(10),
                "cue": np.ones(3),
                "gate_residual": np.zeros(2),
            }

    from integrations.agentguard.adapter import AgentGuardTrackerAdapter

    class _Args:
        dataset = "test"
        reid_dim = 64

    adapter = AgentGuardTrackerAdapter(_Args(), "test_seq", agentguard_runtime=FakeRuntime())
    assert adapter.event_sink == "test_sink"


def test_adapter_no_reinit_feature_builder():
    call_count = [0]

    class FakeRuntime:
        mode = "off"
        stats = None
        event_buffers = {}
        checkpoints = None
        iwg = None
        feature_builder = "already_built"

        def init_feature_builder(self, *a, **kw):
            call_count[0] += 1

        def finalize_first_stage(self, *a, **kw):
            return {}

        def get_or_create_buffer(self, tid):
            return (None, None)

        def cleanup_track(self, tid):
            pass

        def run_iwg_inference(self, seq):
            return {
                "policy_probs": np.ones(5) / 5,
                "gate": np.ones(2),
                "event_logits": np.zeros(10),
                "cue": np.ones(3),
                "gate_residual": np.zeros(2),
            }

    from integrations.agentguard.adapter import AgentGuardTrackerAdapter

    class _Args:
        dataset = "test"
        reid_dim = None

    AgentGuardTrackerAdapter(_Args(), "test_seq", agentguard_runtime=FakeRuntime())
    assert call_count[0] == 0, "init_feature_builder should NOT be called from adapter"


def test_full_mode_missing_model_raises():
    from agentguard.runtime.manager import AgentGuardRuntime

    runtime = AgentGuardRuntime({"mode": "full"}, iwg_model=None, tgr_model=None)
    with pytest.raises(RuntimeError, match="TGR model is None"):
        runtime.finalize_first_stage()


def test_iwg_inference_error_in_iwg_mode():
    from agentguard.runtime.manager import AgentGuardRuntime

    runtime = AgentGuardRuntime({"mode": "iwg"}, iwg_model=None, tgr_model=None)
    with pytest.raises(RuntimeError, match="IWG model or feature builder"):
        runtime.run_iwg_inference([])
