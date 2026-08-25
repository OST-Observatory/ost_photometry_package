"""Regression tests for ``isolated_sys_modules`` runtime-module keep rules."""

from __future__ import annotations

import sys
import types

import pytest

from helpers import ensure_stub_package, isolated_sys_modules, restore_sys_modules


def test_astroquery_survives_isolated_sys_modules():
    astroquery = pytest.importorskip("astroquery")
    marker = astroquery
    with isolated_sys_modules():
        import astroquery.simbad  # noqa: F401

    assert sys.modules.get("astroquery") is marker
    import astroquery as again

    assert again is marker


def test_restore_keeps_real_astroquery_module():
    fake = types.ModuleType("astroquery")
    fake.__file__ = "/tmp/fake_astroquery.py"
    original = sys.modules.get("astroquery")
    before = {k: v for k, v in sys.modules.items() if k != "astroquery"}
    sys.modules["astroquery"] = fake
    try:
        restore_sys_modules(before)
        assert sys.modules.get("astroquery") is fake
    finally:
        if original is not None:
            sys.modules["astroquery"] = original
        else:
            sys.modules.pop("astroquery", None)


def test_restore_drops_astroquery_stub():
    saved = {
        name: sys.modules[name]
        for name in list(sys.modules)
        if name == "astroquery" or name.startswith("astroquery.")
    }
    for name in saved:
        del sys.modules[name]
    try:
        with isolated_sys_modules():
            stub = ensure_stub_package("astroquery")
            assert getattr(stub, "__file__", None) is None
        assert "astroquery" not in sys.modules
    finally:
        sys.modules.update(saved)
