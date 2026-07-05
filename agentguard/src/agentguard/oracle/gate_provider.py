"""Oracle gate provider using pre-computed rollout labels."""

import numpy as np
from typing import Optional


class OracleGateProvider:
    """Provides oracle gates from pre-computed rollout labels.

    Oracle uses HARD decisions (not soft targets):
    - B >= -1e-6 -> gate = 1 (write: beneficial or neutral)
    - B < -1e-6  -> gate = 0 (skip: write is harmful)

    Benefit sign: B = L_skip - L_write.
      B > 0 → skip loss > write loss → writing is beneficial.
    """
    
    def __init__(self, iwg_labels: dict, tgr_window_labels: Optional[dict] = None):
        """
        Args:
            iwg_labels: {(event_id, candidate_type): dict with motion_benefit, appearance_benefit, ...}
            tgr_window_labels: {window_id: ndarray (4, 2) of hard gates}
        """
        self._iwg_labels = iwg_labels
        self._tgr_labels = tgr_window_labels or {}
        self._iwg_cache: dict[tuple, np.ndarray] = {}
    
    def get_iwg_gate(self, event_id: str, candidate_type: str = "A") -> np.ndarray:
        """Get hard oracle gate for an event.
        
        Returns (2,) array [motion_gate, appearance_gate].
        """
        cache_key = (event_id, candidate_type)
        if cache_key in self._iwg_cache:
            return self._iwg_cache[cache_key]
        
        key = (event_id, candidate_type)
        label = self._iwg_labels.get(key)
        
        if label is None:
            label = self._iwg_labels.get((event_id, "A"))
        
        if label is None:
            gate = np.array([1.0, 1.0], dtype=np.float64)
        else:
            B_m = label.get("motion_benefit", 0.0)
            B_a = label.get("appearance_benefit", 0.0)
            gate = np.array([
                self._benefit_to_hard(B_m),
                self._benefit_to_hard(B_a),
            ], dtype=np.float64)
        
        self._iwg_cache[cache_key] = gate
        return gate
    
    def get_tgr_window_gates(self, window_id: str) -> Optional[np.ndarray]:
        """Get TGR window gates. Returns (4, 2) or None."""
        return self._tgr_labels.get(window_id)
    
    @staticmethod
    def _benefit_to_hard(benefit: float) -> float:
        """Convert benefit to hard gate decision.

        B = L_skip - L_write:
          B >= -1e-6 → gate = 1 (write, beneficial or neutral)
          B < -1e-6  → gate = 0 (skip, write is harmful)
        """
        return 1.0 if benefit >= -1e-6 else 0.0
    
    @classmethod
    def from_label_files(cls, label_dir: str) -> 'OracleGateProvider':
        """Load oracle gates from label files."""
        import json
        from pathlib import Path
        
        label_dir = Path(label_dir)
        iwg_labels = {}
        tgr_labels = {}
        
        for f in label_dir.glob("*.json"):
            with open(f) as fh:
                data = json.load(fh)
                if "window_id" in data:
                    tgr_labels[data["window_id"]] = np.array(data["gates"])
                elif "event_id" in data:
                    ct = data.get("candidate_type", "A")
                    iwg_labels[(data["event_id"], ct)] = data
        
        return cls(iwg_labels, tgr_labels)
