"""median_zp with extinction_mode='from_comparison_stars' fits k."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.table import Table


def test_median_zp_per_night_fits_extinction():
    from ost_photometry.analyze.calibration.backends import median
    from ost_photometry.analyze.extinction import (
        DEFAULT_EXTINCTION,
        ExtinctionCorrector,
        ExtinctionOrder,
    )
    from ost_photometry.analyze.pipeline import PipelineConfig

    rng = np.random.default_rng(2)
    n = 40
    v_std = rng.uniform(11.0, 14.0, n)
    b_std = v_std + rng.uniform(0.2, 1.2, n)
    epochs = {}
    for i, x in enumerate(np.linspace(1.1, 2.3, 8)):
        epochs[f"e{i}"] = Table({
            "id": np.arange(n),
            "mag_V": v_std - 25.0 + 0.30 * x + rng.normal(0, 0.005, n),
            "err_V": np.full(n, 0.005),
            "mag_B": b_std - 24.5 + 0.45 * x + rng.normal(0, 0.005, n),
            "err_B": np.full(n, 0.005),
            "mag_std_V": v_std, "err_std_V": np.full(n, 0.01),
            "mag_std_B": b_std, "err_std_B": np.full(n, 0.01),
            "airmass": np.full(n, x),
        })
    config = PipelineConfig(calibration_strategy="median_zp", calibration_grouping="per_night",
                            extinction_mode="from_comparison_stars")
    corrector = ExtinctionCorrector(order=ExtinctionOrder.FIRST)
    corrector.coefficients = {f: DEFAULT_EXTINCTION[f] for f in ("B", "V")}
    median.fit_epochs(epochs, ["B", "V"], config, extinction_corrector=corrector)
    assert corrector.coefficients["V"].k_prime == pytest.approx(0.30, abs=0.01)
    assert corrector.coefficients["B"].k_prime == pytest.approx(0.45, abs=0.01)
