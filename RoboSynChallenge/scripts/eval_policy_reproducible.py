#!/usr/bin/env python
"""Run the existing evaluator with reproducible episode resets.

This diagnostic entry point leaves ``scripts/eval_policy.py`` unchanged.  The
base environment currently reseeds only Torch in ``reset(seed=...)``, although
the random task setting also uses Python, NumPy, CUDA, and Warp random number
generators.  This wrapper reseeds all project RNGs immediately before each
episode reset, then delegates everything else to the existing evaluator.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
for path in (REPO_ROOT, REPO_ROOT / "policy", WORKSPACE_ROOT / "EmbodiChain"):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

import eval_policy as base_evaluator  # noqa: E402
from embodichain.utils import set_seed  # noqa: E402


_base_reset = base_evaluator.RecordingEnvProxy.reset


def _reproducible_reset(
    self: Any, *args: Any, **kwargs: Any
) -> tuple[Any, Any]:
    """Reseed every RNG used by reset-time randomizers."""
    seed = kwargs.get("seed")
    if seed is None and args:
        seed = args[0]
    if seed is not None:
        set_seed(int(seed))
    return _base_reset(self, *args, **kwargs)


base_evaluator.RecordingEnvProxy.reset = _reproducible_reset


if __name__ == "__main__":
    base_evaluator.main()
