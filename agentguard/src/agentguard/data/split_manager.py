from __future__ import annotations

import os
from typing import List, Optional

import yaml


class SplitManager:
    """Manages dataset splits for training/validation.

    Reads split configurations from YAML files located in a config
    directory (default: ``agentguard/configs/splits/``).

    Expected YAML format::

        # agentguard/configs/splits/<dataset>.yaml
        train:
          - sequence_name_01
          - sequence_name_02
          ...
        val:
          - sequence_name_03
          ...

    Parameters
    ----------
    config_dir : str, optional
        Directory containing the ``<dataset>.yaml`` split config files.
        Defaults to ``'agentguard/configs/splits'`` relative to the
        current working directory.
    """

    def __init__(self, config_dir: Optional[str] = None) -> None:
        self.config_dir = config_dir or "agentguard/configs/splits"

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def get_sequences(self, dataset: str, mode: str) -> List[str]:
        """Get the list of sequence names for a dataset and split mode.

        Parameters
        ----------
        dataset : str
            Dataset name (e.g. ``'mot17'``, ``'dance'``).
        mode : str
            Split name (e.g. ``'train'`` or ``'val'``).

        Returns
        -------
        list of str
            Sequence names for the requested split.

        Raises
        ------
        FileNotFoundError
            If the YAML config file does not exist.
        KeyError
            If the requested mode is not present in the config.
        """
        config = self._load_config(dataset)
        if mode not in config:
            raise KeyError(
                f"Mode '{mode}' not found in split config for dataset "
                f"'{dataset}'. Available modes: {list(config.keys())}"
            )
        sequences = config[mode]
        if not isinstance(sequences, list):
            raise TypeError(
                f"Expected a list of sequence names for mode '{mode}' "
                f"in dataset '{dataset}', got {type(sequences).__name__}"
            )
        return list(sequences)

    def get_train_sequences(self, dataset: str) -> List[str]:
        """Convenience: return training sequences for a dataset."""
        return self.get_sequences(dataset, "train")

    def get_val_sequences(self, dataset: str) -> List[str]:
        """Convenience: return validation sequences for a dataset."""
        return self.get_sequences(dataset, "val")

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _config_path(self, dataset: str) -> str:
        """Return the expected YAML file path for a dataset."""
        return os.path.join(self.config_dir, f"{dataset}.yaml")

    def _load_config(self, dataset: str) -> dict:
        """Load and parse the YAML split config for *dataset*."""
        path = self._config_path(dataset)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Split config file not found: {path}  "
                f"(expected at {{config_dir}}/{dataset}.yaml)"
            )
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise TypeError(
                f"Expected a top-level mapping in {path}, "
                f"got {type(config).__name__}"
            )
        return config
