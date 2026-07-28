"""Tests for Runtime and TrackTrack adapter initialization boundaries."""

from types import SimpleNamespace

from agentguard.runtime.manager import AgentGuardRuntime


def test_runtime_event_sink_is_none_by_default():
    assert AgentGuardRuntime({"mode": "off"}).event_sink is None


def test_runtime_feature_builder_initialised_once():
    runtime = AgentGuardRuntime({"mode": "capture"})
    runtime.init_feature_builder(reid_dim=64)
    builder = runtime.feature_builder
    runtime.init_feature_builder(reid_dim=128)
    assert runtime.feature_builder is builder
    assert builder.reid_dim == 64


def test_adapter_reads_optional_event_sink_without_loading_a_model():
    from integrations.agentguard.adapter import AgentGuardTrackerAdapter

    runtime = AgentGuardRuntime({"mode": "capture"})
    runtime.event_sink = object()
    args = SimpleNamespace(dataset="test", agentguard_mode="capture")
    adapter = AgentGuardTrackerAdapter(args, "test_seq", agentguard_runtime=runtime)
    assert adapter.event_sink is runtime.event_sink
    assert adapter.capture_only is True


def test_adapter_does_not_initialize_feature_builder():
    from integrations.agentguard.adapter import AgentGuardTrackerAdapter

    runtime = AgentGuardRuntime({"mode": "capture"})
    args = SimpleNamespace(dataset="test", agentguard_mode="capture")
    AgentGuardTrackerAdapter(args, "test_seq", agentguard_runtime=runtime)
    assert runtime.feature_builder is None
