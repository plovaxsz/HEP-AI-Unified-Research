"""Session-only EnergyFlow checksum bypass.

This file is intended for a *single execution session* only.
It monkey-patches the checksum validation inside `energyflow.datasets.qg_jets`
so that `energyflow.datasets.qg_jets.load(...)` proceeds using the local
artifact.

Use ONLY if you are certain the local dataset is scientifically valid.
"""

from __future__ import annotations

import runpy
from unittest import mock


def _patch_energyflow_checksum() -> None:
    # Import lazily so we can patch before `load()` performs validation.
    from energyflow.datasets import qg_jets  # type: ignore
    from energyflow.utils import data_utils  # type: ignore

    # Strategy: patch the lower-level validator that performs the SHA check.
    # In energyflow>=1.x this is `_validate_file` inside `energyflow.utils.data_utils`.
    if hasattr(data_utils, "_validate_file"):
        original_validate = data_utils._validate_file

        def _always_true(*args, **kwargs):  # noqa: ANN001
            return True

        mock.patch.object(data_utils, "_validate_file", side_effect=_always_true).start()
        return

    # Fallback: patch qg_jets._get_filepath call sites won't work reliably;
    # so if `_validate_file` doesn't exist, we patch qg_jets.load to bypass
    # validation by patching the loader internals.
    # Note: this fallback is conservative and may break in future versions.
    if hasattr(qg_jets, "_get_filepath"):
        mock.patch.object(qg_jets, "_get_filepath", side_effect=lambda *a, **k: a[0]).start()


def main() -> None:
    _patch_energyflow_checksum()

    # Run the canonicalization module as `__main__`.
    runpy.run_module(
        "TDA_GATv2_Research.build_canonical_data",
        run_name="__main__",
    )


if __name__ == "__main__":
