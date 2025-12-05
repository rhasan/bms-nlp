"""Tests for bms.extract_point_names module."""

import json
from pathlib import Path

import pandas as pd

from src.bms import extract_point_names as epn

# ---------------------------
# is_bms_style_string tests
# ---------------------------


def test_is_bms_style_string_true_for_classic_bms_names():
    """Test that is_bms_style_string returns True for typical BMS point names."""
    assert epn.is_bms_style_string("BLDG1_FL03_AHU2_SAT_AI")
    assert epn.is_bms_style_string("AHU-03.SAT")
    assert epn.is_bms_style_string("Effective_RA_CO_dac_AV")
    assert epn.is_bms_style_string("VAV12_CLG_CMD")


def test_is_bms_style_string_false_for_generic_headers():
    """Test that is_bms_style_string returns False for common non-BMS headers."""
    for s in ["bas_raw", "haystack", "Label", "TagSet", "name", "units"]:
        assert not epn.is_bms_style_string(s)


def test_is_bms_style_string_handles_non_string_and_short_inputs():
    """Test that is_bms_style_string returns False for non-string and short inputs."""
    assert not epn.is_bms_style_string(None)
    assert not epn.is_bms_style_string(123)
    assert not epn.is_bms_style_string("A")  # too short
    assert not epn.is_bms_style_string("AB")  # too short


# ---------------------------
# detect_header tests
# ---------------------------


def test_detect_header_with_explicit_header_like_b3_ibm():
    """Test detect_header with a DataFrame that has a header row."""
    # Simulate b3_ibm.csv after read_csv(header=None)
    df = pd.DataFrame(
        [
            ["Label", "TagSet", "AssetType"],
            ["1F_MID_OPENOFF_CO2", "CO2_Sensor", "Sensor"],
            ["1F_NRTH_OPENOFF_CO2", "CO2_Sensor", "Sensor"],
        ]
    )
    assert epn.detect_header(df) is True


def test_detect_header_for_headerless_points_file():
    """Test detect_header with a DataFrame that has no header row."""
    # No header, just data rows
    df = pd.DataFrame(
        [
            ["1F_MID_OPENOFF_CO2", "CO2_Sensor"],
            ["1F_NRTH_OPENOFF_CO2", "CO2_Sensor"],
            ["1F_SOUTH_OPENOFF_CO2", "CO2_Sensor"],
        ]
    )
    assert epn.detect_header(df) is False


def test_detect_header_with_bas_raw_haystack_header():
    """Test detect_header with a DataFrame that has 'bas_raw' and 'haystack' header."""
    df = pd.DataFrame(
        [
            ["bas_raw", "haystack"],
            ["CMU/SCSC Gates/Eighth Floor/8126 Machine Room CRAC-9/% Capacity", "equip chiller coolingCapacity"],
            ["CMU/SCSC Gates/Eighth Floor/8127 Machine Room CRAC-10/% Capacity", "equip chiller coolingCapacity"],
        ]
    )
    assert epn.detect_header(df) is True


def test_detect_header_two_row_fallback_header():
    """Test detect_header with only two rows, where the first looks like a header."""
    # Only 2 rows, header + one data row
    df = pd.DataFrame(
        [
            ["Label", "TagSet"],
            ["1F_MID_OPENOFF_CO2", "CO2_Sensor"],
        ]
    )
    assert epn.detect_header(df) is True


def test_detect_header_two_row_fallback_no_header():
    """Test detect_header with only two rows, both looking like data."""
    # Only 2 rows, both look like data
    df = pd.DataFrame(
        [
            ["1F_MID_OPENOFF_CO2", "CO2_Sensor"],
            ["1F_NRTH_OPENOFF_CO2", "CO2_Sensor"],
        ]
    )
    assert epn.detect_header(df) is False


# ---------------------------
# guess_point_label_column tests
# ---------------------------


def test_guess_point_label_column_prefers_known_names(monkeypatch):
    """Test that guess_point_label_column prefers known column names."""
    # Make sure PREFERRED_POINT_COLS includes "label"
    monkeypatch.setattr(epn, "PREFERRED_POINT_COLS", {"label"})
    df = pd.DataFrame(
        {
            "Label": ["1F_MID_OPENOFF_CO2", "1F_NRTH_OPENOFF_CO2"],
            "Other": ["foo", "bar"],
        }
    )
    col = epn.guess_point_label_column(df)
    assert col == "Label"


def test_guess_point_label_column_uses_bms_style_score():
    """Test that guess_point_label_column uses BMS-style scoring when no preferred names."""
    df = pd.DataFrame(
        {
            "c1": ["foo", "bar", "baz"],
            "c2": ["1F_MID_OPENOFF_CO2", "1F_NRTH_OPENOFF_CO2", "1F_SOUTH_OPENOFF_CO2"],
        }
    )
    col = epn.guess_point_label_column(df)
    assert col == "c2"


# ---------------------------
# derive_building_id_from_filename tests
# ---------------------------


def test_derive_building_id_strips_leading_digits_and_underscore():
    """Test that derive_building_id_from_filename works as expected."""
    assert epn.derive_building_id_from_filename("01_office_singapore.csv") == "office_singapore"
    assert epn.derive_building_id_from_filename("001-BLDG_A.csv") == "BLDG_A"
    assert epn.derive_building_id_from_filename("ghc_cmu.csv") == "ghc_cmu"


# ---------------------------
# load_all_bms_points tests (integration-style with tmp_path)
# ---------------------------


def test_load_all_bms_points_with_header_file(tmp_path: Path):
    """Test load_all_bms_points with a file that has a header row."""
    # Simulate a b3_ibm-like file with a header
    csv_content = """Label,TagSet
1F_MID_OPENOFF_CO2,CO2_Sensor
1F_NRTH_OPENOFF_CO2,CO2_Sensor
"""
    csv_path = tmp_path / "b3_ibm.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    out_path = tmp_path / "all_points.jsonl"
    epn.load_all_bms_points(raw_dir=tmp_path, bms_output_file=out_path)

    # Read JSONL
    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert {r["point_label"] for r in rows} == {
        "1F_MID_OPENOFF_CO2",
        "1F_NRTH_OPENOFF_CO2",
    }
    assert all(r["building_id"] == "b3_ibm" for r in rows)


def test_load_all_bms_points_with_headerless_file(tmp_path: Path):
    """Test load_all_bms_points with a file that has no header row."""
    # Headerless file: all rows are data
    csv_content = """1F_MID_OPENOFF_CO2,CO2_Sensor
1F_NRTH_OPENOFF_CO2,CO2_Sensor
1F_SOUTH_OPENOFF_CO2,CO2_Sensor
"""
    csv_path = tmp_path / "no_header.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    out_path = tmp_path / "all_points.jsonl"
    epn.load_all_bms_points(raw_dir=tmp_path, bms_output_file=out_path)

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    # We expect 3 rows (no header removed)
    assert len(rows) == 3
    labels = [r["point_label"] for r in rows]
    assert labels == [
        "1F_MID_OPENOFF_CO2",
        "1F_NRTH_OPENOFF_CO2",
        "1F_SOUTH_OPENOFF_CO2",
    ]
    assert all(r["building_id"] == "no_header" for r in rows)


def test_load_all_bms_points_mixed_header_and_headerless(tmp_path: Path):
    """Test load_all_bms_points with mixed header and headerless files."""
    # File 1: with header
    csv1 = """Label,TagSet
1F_MID_OPENOFF_CO2,CO2_Sensor
"""
    (tmp_path / "b3_ibm.csv").write_text(csv1, encoding="utf-8")

    # File 2: headerless
    csv2 = """1F_NRTH_OPENOFF_CO2,CO2_Sensor
1F_SOUTH_OPENOFF_CO2,CO2_Sensor
"""
    (tmp_path / "no_header.csv").write_text(csv2, encoding="utf-8")

    out_path = tmp_path / "all_points.jsonl"
    epn.load_all_bms_points(raw_dir=tmp_path, bms_output_file=out_path)

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]

    # 1 from headered file + 2 from headerless = 3
    assert len(rows) == 3
    labels = sorted(r["point_label"] for r in rows)
    assert labels == [
        "1F_MID_OPENOFF_CO2",
        "1F_NRTH_OPENOFF_CO2",
        "1F_SOUTH_OPENOFF_CO2",
    ]
    # building_id derived from filenames
    assert any(r["building_id"] == "b3_ibm" for r in rows)
    assert any(r["building_id"] == "no_header" for r in rows)
