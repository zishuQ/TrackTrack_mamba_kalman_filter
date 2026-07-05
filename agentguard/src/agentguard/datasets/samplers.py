from __future__ import annotations

from typing import Iterator, List, Optional

import numpy as np
import torch


class BalancedSampler(torch.utils.data.Sampler):
    """Balanced sampling across sample types (A/B/C/unmatched).

    Ensures each mini-batch (or epoch) contains roughly equal numbers of
    samples from each ``sample_type`` category.  The sampler assumes that
    a ``dataset.labels`` list is available and each label dict contains a
    ``sample_type`` integer key.

    The sampler works in two modes:

    * **Exhaustive** (``samples_per_type=None``): yields a balanced shuffle of
      *all* available samples by over-sampling minority classes.
    * **Fixed-size** (``samples_per_type=k``): yields exactly ``k`` samples
      from each type per epoch (may repeat or trim).

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset whose ``labels`` attribute is a list of dicts containing
        ``sample_type``.
    samples_per_type : int or None
        Number of samples to draw from each type per epoch.  When ``None``
        the largest class size is used (all classes are padded to the same
        count).
    num_types : int
        Number of sample-type categories (default 4: A, B, C, unmatched).
    seed : int
        Random seed for reproducibility (default 42).
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        samples_per_type: Optional[int] = None,
        num_types: int = 4,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.samples_per_type = samples_per_type
        self.num_types = num_types
        self.seed = seed

        # Collect indices per type.
        self._indices_per_type: List[List[int]] = [[] for _ in range(num_types)]
        for i, label in enumerate(dataset.labels):
            t = int(label.get("sample_type", 0))
            if 0 <= t < num_types:
                self._indices_per_type[t].append(i)
            # Silently drop indices with out-of-range types.

        # Determine how many samples to draw per type.
        if samples_per_type is not None:
            self._count = samples_per_type
        else:
            self._count = max(len(indices) for indices in self._indices_per_type)

        self._total_samples = self._count * self.num_types

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._total_samples

    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[int]:
        """Yield a balanced sequence of sample indices."""
        rng = np.random.default_rng(self.seed)

        batches: List[List[int]] = []
        for type_indices in self._indices_per_type:
            if len(type_indices) == 0:
                # No samples for this type — repeat other types instead.
                # Use a placeholder empty list; will be skipped.
                batches.append([])
                continue

            # Resample with replacement if we need more than available.
            if len(type_indices) >= self._count:
                chosen = rng.choice(type_indices, size=self._count, replace=False).tolist()
            else:
                # Over-sample: full set + random picks to fill.
                chosen = type_indices.copy()
                extra = rng.choice(type_indices, size=self._count - len(type_indices), replace=True).tolist()
                chosen.extend(extra)
            # Shuffle within type for randomness.
            rng.shuffle(chosen)
            batches.append(chosen)

        # Interleave across types.
        result: List[int] = []
        for pos in range(self._count):
            for t in range(self.num_types):
                if pos < len(batches[t]):
                    result.append(batches[t][pos])

        # Final global shuffle to avoid deterministic ordering.
        rng.shuffle(result)
        return iter(result)
