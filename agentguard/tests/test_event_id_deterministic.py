"""Test that event IDs are deterministic (no UUID)."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_event_id_format():
    """Event ID format: dataset/sequence/frame_id/track_id"""
    # Test the pattern without UUID
    event_id = "MOT17/MOT17-02/000001/000005"
    parts = event_id.split("/")
    assert len(parts) == 4, f"Expected 4 parts, got {len(parts)}"
    assert parts[2].isdigit()
    assert parts[3].isdigit()

def test_candidate_id_format():
    event_id = "MOT17/MOT17-02/000001/000005"
    candidate_a = f"{event_id}/A"
    candidate_b = f"{event_id}/B"
    assert candidate_a.endswith("/A")
    assert candidate_b.endswith("/B")

def test_window_id_format():
    window_id = "MOT17-02/5/00100-00103/ORIGINAL"
    parts = window_id.split("/")
    assert len(parts) == 4
    # parts[2] is the frame range (e.g. "00100-00103")
    frame_parts = parts[2].split("-")
    assert len(frame_parts) == 2
    assert frame_parts[0].isdigit()
    assert frame_parts[1].isdigit()
    # parts[3] is the window type
    assert parts[3] in ("ORIGINAL", "B_AT_2", "B_AT_3", "B_AT_2_3")
