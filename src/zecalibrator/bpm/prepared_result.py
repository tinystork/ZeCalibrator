"""The prepared calibration result — a *separate* type referencing ``CalibrationResult`` v1.

``PreparedCalibrationResult`` is the application-lot result of targeted CFA
preparation. It is **not** a modification of the frozen ``CalibrationResult`` v1
(``core/calibrate.py``) and is **not** exposed through ``api/v1``: it is an
internal product type that *references* the ordinary calibration result and adds
the preparation masks (mission §33–§34, ARCHITECTURE §18.4).

Mask semantics (documented and pinned by test):

* ``measurement_dq`` — the post-calibration **measurement** DQ mask, bit-for-bit
  identical to ``calibration.mask``. Preparation never edits it: DQ v1 semantics
  are not redefined (ARCHITECTURE §18.4).
* ``reconstructed_mask`` — ``1`` at each pixel whose measured value the operator
  actually replaced (reconstructed from same-CFA donors), ``0`` elsewhere.
* ``usable_mask`` — the donor-eligibility population: a pixel is usable iff its
  measurement is valid (``measurement_dq == 0``) **and** it was not reconstructed.
  This is exactly the population the operator admitted as donors.

When nothing is prepared, ``prepared_data`` is bit-for-bit identical to
``calibration.data``, ``reconstructed_mask`` is all-zero and ``usable_mask`` is
``measurement_dq == 0`` (mission §80).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zecalibrator.core.calibrate import CalibrationResult


@dataclass(frozen=True)
class PreparedCalibrationResult:
    """One frame's prepared result: the intact v1 result + the preparation masks.

    ``calibration`` is the **post-calibration** ``CalibrationResult`` (float32 CFA
    + uint16 DQ) the operator consumed — never the raw light. ``prepared_data`` is
    the freshly allocated float32 CFA after preparation (equal to
    ``calibration.data`` when nothing was reconstructed). ``measurement_dq``,
    ``reconstructed_mask`` and ``usable_mask`` are as described in the module
    docstring.
    """

    calibration: CalibrationResult
    prepared_data: np.ndarray  # native contiguous float32
    measurement_dq: np.ndarray  # uint16 (== calibration.mask)
    reconstructed_mask: np.ndarray  # uint8 0/1
    usable_mask: np.ndarray  # bool
    operator_id: str
    operator_version: str
    prepared_site_count: int = 0

    def __post_init__(self) -> None:
        data = np.asarray(self.prepared_data)
        m_dq = np.asarray(self.measurement_dq)
        rec = np.asarray(self.reconstructed_mask)
        usa = np.asarray(self.usable_mask, dtype=bool)
        if data.ndim != 2:
            raise ValueError("prepared_data must be a 2D plane")
        if m_dq.shape != data.shape or rec.shape != data.shape or usa.shape != data.shape:
            raise ValueError(
                f"mask shape mismatch: data {data.shape}, measurement_dq {m_dq.shape}, "
                f"reconstructed {rec.shape}, usable {usa.shape}"
            )
        if isinstance(self.prepared_site_count, bool) or not isinstance(self.prepared_site_count, int) or self.prepared_site_count < 0:
            raise ValueError("prepared_site_count must be a non-negative int")
        object.__setattr__(self, "prepared_data", np.ascontiguousarray(data.astype(np.float32)))
        object.__setattr__(self, "measurement_dq", np.ascontiguousarray(m_dq.astype(np.uint16)))
        object.__setattr__(self, "reconstructed_mask", np.ascontiguousarray(rec.astype(np.uint8)))
        object.__setattr__(self, "usable_mask", np.ascontiguousarray(usa.astype(bool)))
        object.__setattr__(self, "prepared_site_count", int(self.prepared_site_count))

    @property
    def nothing_prepared(self) -> bool:
        """Whether no pixel was reconstructed on this frame (mission §80)."""
        return self.prepared_site_count == 0


__all__ = ["PreparedCalibrationResult"]
