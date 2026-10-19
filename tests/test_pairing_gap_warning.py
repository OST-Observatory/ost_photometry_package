"""Pairing gaps larger than 2 % of a known OOI period are reported."""

from __future__ import annotations

from types import SimpleNamespace


def test_pairing_gap_warning(capsys, caplog):
    from ost_photometry.analyze.pipeline.bridge import _warn_pairing_gaps_vs_period

    ooi = SimpleNamespace(name="RZ Cas", period=1.1952499)
    ctx = SimpleNamespace(objects_of_interest=[ooi])
    groups = [{"B": SimpleNamespace(jd=2460000.50), "V": SimpleNamespace(jd=2460000.50 + 40 / 1440)}]
    _warn_pairing_gaps_vs_period(ctx, groups)
    # terminal output goes to stdout or, once logging is set up, to the log
    assert "RZ Cas" in capsys.readouterr().out + caplog.text
    caplog.clear()
    close = [{"B": SimpleNamespace(jd=2460000.50), "V": SimpleNamespace(jd=2460000.50 + 5 / 1440)}]
    _warn_pairing_gaps_vs_period(ctx, close)
    assert "RZ Cas" not in capsys.readouterr().out + caplog.text
