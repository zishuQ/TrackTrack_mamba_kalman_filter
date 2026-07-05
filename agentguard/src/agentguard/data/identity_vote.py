"""Track identity voting for GT annotation."""

from collections import defaultdict
from typing import Optional, Tuple


class TrackIdentityVoteState:
    """Maintains per-track GT identity voting state.
    
    Identity key = (sequence_name, gt_id).
    Different sequences with same numeric gt_id are NOT mixed.
    """
    
    def __init__(self):
        # {track_id: {identity_key: count}}
        self._votes_by_track: dict[int, dict[Tuple[str, int], int]] = defaultdict(dict)
        # {track_id: target_identity_key} set after resolve
        self._resolved_identity: dict[int, Optional[Tuple[str, int]]] = {}
    
    def resolve_before_current(self, sequence: str, track_id: int) -> Optional[int]:
        """Resolve target GT ID from historical votes (before current frame).
        
        Returns None if identity is not yet reliable.
        Identity is reliable when: valid_votes >= 3 AND top_ratio >= 0.75.
        """
        votes = self._votes_by_track.get(track_id, {})
        if not votes:
            self._resolved_identity[track_id] = None
            return None
        
        total = sum(votes.values())
        if total < 3:
            self._resolved_identity[track_id] = None
            return None
        
        best_key, best_count = max(votes.items(), key=lambda x: x[1])
        ratio = best_count / total
        if ratio < 0.75:
            self._resolved_identity[track_id] = None
            return None
        
        self._resolved_identity[track_id] = best_key
        return best_key[1]  # Return gt_id
    
    def add_current_observation(self, sequence: str, track_id: int, detection_gt_id: int) -> None:
        """Add current frame's detection GT match to voting history.
        
        Must be called AFTER resolve_before_current.
        Ignores gt_id == -1 (no match).
        """
        if detection_gt_id < 0:
            return
        
        key = (sequence, int(detection_gt_id))
        self._votes_by_track[track_id][key] = self._votes_by_track[track_id].get(key, 0) + 1
    
    def get_target_identity_key(self, track_id: int) -> Optional[Tuple[str, int]]:
        """Get the resolved identity key for a track."""
        return self._resolved_identity.get(track_id)
    
    def get_target_gt_id(self, track_id: int) -> Optional[int]:
        """Get just the gt_id part of resolved identity."""
        key = self._resolved_identity.get(track_id)
        return key[1] if key else None
    
    def is_reliable(self, track_id: int) -> bool:
        return self._resolved_identity.get(track_id) is not None
