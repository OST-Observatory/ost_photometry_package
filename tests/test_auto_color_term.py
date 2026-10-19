"""color_term_fit='auto' is decided once per filter for all epochs."""

from __future__ import annotations

import numpy as np
from astropy.table import Table


def _epoch(rng, spread):
    n = 30
    v = rng.uniform(11, 14, n)
    bv = 0.6 + rng.normal(0, spread, n)
    return Table({
        "id": np.arange(n), "mag_V": v - 25 + 0.05 * bv + rng.normal(0, 0.005, n), "err_V": np.full(n, 0.005),
        "mag_B": v + bv - 24.6 + 0.1 * bv + rng.normal(0, 0.005, n), "err_B": np.full(n, 0.005),
        "mag_std_V": v, "err_std_V": np.full(n, 0.01), "mag_std_B": v + bv, "err_std_B": np.full(n, 0.01),
        "airmass": np.full(n, 1.2), "ra": np.full(n, 10.0), "dec": np.full(n, 20.0),
    })


def test_auto_color_term_is_one_decision_for_all_epochs():
    from ost_photometry.analyze.calibration.backends.linear import build_calibrator
    from ost_photometry.analyze.pipeline import PipelineConfig

    rng = np.random.default_rng(4)
    config = PipelineConfig(calibration_strategy="linear_fit", calibration_grouping="per_image",
                            color_term_fit="auto")
    cal = build_calibrator(config)
    # per-epoch spreads straddle 0.1; pooled spread ~0.1+
    for i, spread in enumerate((0.07, 0.13, 0.08, 0.14)):
        cal.add_epoch(f"e{i}", _epoch(rng, spread))
    decision = cal._decide_color_terms(["B", "V"], lambda t: np.isfinite(np.asarray(t["mag_std_V"], float)))
    assert set(decision.values()) <= {"always", "never"}
    results = cal.fit_transformation_parameters(["B", "V"], color_term_fit="auto")
    fitted = {eid: r.transformation["B"].color_term != 0.0 for eid, r in results.items()}
    assert len(set(fitted.values())) == 1
