"""Unit tests for generate_bms_vocab module."""

import json
from collections import Counter, defaultdict

import pytest

from src.bms import generate_bms_vocab as gmv

# -----------------------------
# Primitive helpers
# -----------------------------


def test_is_io_type_recognises_known_io():
    """Test that is_io_type correctly identifies known IO types."""
    assert gmv.is_io_type("AI")
    assert gmv.is_io_type("DO")
    assert not gmv.is_io_type("XYZ")


def test_is_vendor_recognises_known_and_net_suffix():
    """Test that is_vendor correctly identifies known vendors and -NET suffix."""
    assert gmv.is_vendor("SIEMENS")
    assert gmv.is_vendor("BACNET")
    assert gmv.is_vendor("MYNET")  # suffix -NET, short
    assert not gmv.is_vendor("LONGNETNAME")  # too long
    assert not gmv.is_vendor("NOTVENDOR")


def test_passes_global_thresholds_true_when_freq_and_buildings_high(monkeypatch):
    """Test that passes_global_thresholds returns True when frequency and building count are high enough."""
    token_counter = Counter({"AHU": 15})
    token_buildings = {"AHU": {"B1", "B2"}}

    # ensure thresholds are not accidentally changed in future
    assert gmv.passes_global_thresholds("AHU", token_counter, token_buildings)


def test_passes_global_thresholds_false_when_freq_low():
    """Test that passes_global_thresholds returns False when frequency is too low."""
    token_counter = Counter({"AHU": 5})
    token_buildings = {"AHU": {"B1", "B2"}}
    assert not gmv.passes_global_thresholds("AHU", token_counter, token_buildings)


def test_score_functions_behave_monotonically():
    """Test that scoring functions produce higher scores for better inputs."""
    equip_score_low = gmv.score_equip({"freq": 1, "buildings": 1, "numid_bigrams": 0})
    equip_score_high = gmv.score_equip({"freq": 10, "buildings": 3, "numid_bigrams": 2})
    assert equip_score_high > equip_score_low

    sub_low = gmv.score_subcomp({"freq": 1, "buildings": 1})
    sub_high = gmv.score_subcomp({"freq": 5, "buildings": 3})
    assert sub_high > sub_low

    pf_low = gmv.score_pointfunc({"freq": 1, "buildings": 1})
    pf_high = gmv.score_pointfunc({"freq": 3, "buildings": 4})
    assert pf_high > pf_low


# -----------------------------
# likely_* helpers
# -----------------------------


def test_likely_equip_accepts_seed_even_with_low_freq(monkeypatch):
    """Test that likely_equip accepts seed equipment tokens even if frequency is low."""
    token_counter = Counter({"AHU": 1})
    token_buildings = {"AHU": {"B1"}}
    token_numid_bigram = Counter()

    # seeds are always accepted if present at least once
    assert gmv.likely_equip("AHU", token_counter, token_buildings, token_numid_bigram)


def test_likely_subcomponent_detects_measurement_keywords():
    """Test that likely_subcomponent detects measurement-related subcomponent tokens."""
    token_counter = Counter({"TEMP": 20})
    token_buildings = {"TEMP": {"B1", "B2", "B3"}}
    assert gmv.likely_subcomponent("TEMP", token_counter, token_buildings)


def test_likely_point_func_recognises_cmd_and_status():
    """Test that likely_point_func recognizes CMD and STATUS point function tokens."""
    token_counter = Counter({"CMD": 20, "STATUS": 20})
    token_buildings = {
        "CMD": {"B1", "B2"},
        "STATUS": {"B1", "B2"},
    }
    assert gmv.likely_point_func("CMD", token_counter, token_buildings)
    assert gmv.likely_point_func("STATUS", token_counter, token_buildings)


def test_likely_equip_requires_uppercase_and_length(monkeypatch):
    """Test that likely_equip rejects lowercase and short tokens."""
    # This token is lowercase and should be rejected by shape checks
    token_counter = Counter({"ahu": 20})
    token_buildings = {"ahu": {"B1", "B2"}}
    token_numid_bigram = Counter({"ahu": 10})

    assert not gmv.likely_equip("ahu", token_counter, token_buildings, token_numid_bigram)


def test_likely_equip_rejects_stopwords():
    """Test that likely_equip rejects tokens in EQUIP_STOPWORDS."""
    token_counter = Counter({"FLOW": 50})
    token_buildings = {"FLOW": {"B1", "B2", "B3"}}
    token_numid_bigram = Counter({"FLOW": 10})

    # FLOW is in EQUIP_STOPWORDS and should not be equipment
    assert not gmv.likely_equip("FLOW", token_counter, token_buildings, token_numid_bigram)


# -----------------------------
# extract_vocab on a tiny JSONL
# -----------------------------


def test_extract_vocab_small_dataset(tmp_path):
    """
    Build a tiny artificial dataset where:
    - AHU appears with numeric ID and SAT, AI, CMD
    - vendor SIEMENS appears
    and check that all basic vocab groups are populated as expected.
    """
    jsonl_path = tmp_path / "all_points.jsonl"

    lines = []
    # create enough repetitions to pass global thresholds
    for i in range(12):
        lines.append(
            {
                "building_id": "B1" if i < 6 else "B2",
                "point_label": "SIEMENS_AHU-01.SAT_AI_CMD",
            }
        )

    with jsonl_path.open("w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj) + "\n")

    vocabs = gmv.extract_vocab(jsonl_path)

    # IO type
    assert "AI" in vocabs["io_type_vocab"]
    # Vendor
    assert "SIEMENS" in vocabs["vendor_vocab"]
    # Equipment
    assert "AHU" in vocabs["equip_vocab"]
    # Subcomponent
    assert "SAT" in vocabs["subcomp_vocab"]
    # Point function (CMD is in seeds)
    assert "CMD" in vocabs["point_func_vocab"]
