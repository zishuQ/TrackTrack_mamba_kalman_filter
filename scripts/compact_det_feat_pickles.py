#!/usr/bin/env python3
"""Compatibility wrapper for compact detection-feature storage.

This script now builds the compact target-NMS index directly from the
source-NMS pickle. It does not read or write a target pickle.
"""

from build_nms_idx_from_95 import main


if __name__ == "__main__":
    main()
