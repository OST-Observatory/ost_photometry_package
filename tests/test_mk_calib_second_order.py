"""Tests for mk_calib export and second-order extinction analysis."""

from helpers import load_module_from_path, pkg_src


def _mk_calib_mod():
    return load_module_from_path(
        "ost_photometry.analyze.calibration.mk_calib",
        pkg_src() / "ost_photometry" / "analyze" / "calibration" / "mk_calib.py",
    )


def _second_order_mod():
    return load_module_from_path(
        "ost_photometry.analyze.calibration.second_order_extinction",
        pkg_src()
        / "ost_photometry"
        / "analyze"
        / "calibration"
        / "second_order_extinction.py",
    )


def test_merge_field_transformation_records():
    mk = _mk_calib_mod()
    TransformCoefficient = mk.TransformCoefficient
    FieldTransformationRecord = mk.FieldTransformationRecord

    r1 = FieldTransformationRecord(
        name="field_a",
        filter_pair=("B", "V"),
        jd=2459000.0,
        airmass={"B": 1.1, "V": 1.1},
        coefficients=[
            TransformCoefficient("Cbbv", "B", ("B", "V"), 0.98, 0.01),
        ],
    )
    r2 = FieldTransformationRecord(
        name="field_a",
        filter_pair=("V", "R"),
        jd=2459000.0,
        airmass={"V": 1.2, "R": 1.2},
        coefficients=[
            TransformCoefficient("Cvvr", "V", ("V", "R"), 1.01, 0.02),
        ],
    )
    merged = mk.merge_field_transformation_records([r1, r2])
    cols = {c.column for c in merged.coefficients}
    assert cols == {"Cbbv", "Cvvr"}
    assert merged.airmass["B"] == 1.1
    assert merged.airmass["R"] == 1.2


def test_json_roundtrip(tmp_path):
    mk = _mk_calib_mod()
    TransformCoefficient = mk.TransformCoefficient
    FieldTransformationRecord = mk.FieldTransformationRecord

    rec = FieldTransformationRecord(
        name="NGC188",
        filter_pair=("B", "V"),
        jd=2459001.5,
        airmass={"B": 1.05, "V": 1.05},
        coefficients=[
            TransformCoefficient("Cbbv", "B", ("B", "V"), 0.97, 0.01),
            TransformCoefficient("Cvbv", "V", ("B", "V"), 1.03, 0.02),
        ],
        n_comparison_stars=42,
    )
    path = tmp_path / "trans_para_NGC188.json"
    mk.write_field_transformation_json(rec, path)
    loaded = mk.load_field_transformation_record(path)
    assert loaded.name == "NGC188"
    assert loaded.coefficient_by_column("Cbbv").c == 0.97
    assert loaded.n_comparison_stars == 42


def test_second_order_fit_synthetic():
    mk = _mk_calib_mod()
    second = _second_order_mod()
    TransformCoefficient = mk.TransformCoefficient
    FieldTransformationRecord = mk.FieldTransformationRecord

    records = []
    for i, (name, x, c) in enumerate(
        [
            ("a", 1.0, 0.10),
            ("b", 1.3, 0.07),
            ("c", 1.6, 0.04),
        ]
    ):
        records.append(
            FieldTransformationRecord(
                name=name,
                filter_pair=("B", "V"),
                jd=2459000.0 + i,
                airmass={"B": x},
                coefficients=[
                    TransformCoefficient(
                        "Cbbv", "B", ("B", "V"), c, 0.01
                    ),
                ],
            )
        )

    t, t_err, k, k_err = second.fit_second_order_extinction(
        [r.airmass["B"] for r in records],
        [r.coefficient_by_column("Cbbv").c for r in records],
        [r.coefficient_by_column("Cbbv").c_err for r in records],
        apply_weights=True,
    )
    assert abs(k - (-0.10)) < 0.01
    assert abs(t - 0.20) < 0.05
