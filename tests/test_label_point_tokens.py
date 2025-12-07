"""Unit tests for label_point_tokens module."""

import pytest

from src.bms import label_point_tokens as lpt

# -----------------------------
# Helper: tiny vocab fixture
# -----------------------------


@pytest.fixture
def small_vocabs():
    """Provide a small vocabulary set for testing."""
    return {
        "EQUIP": {"AHU"},
        "SUBCOMP": {"SAT", "TEMP"},
        "POINT_FUNC": {"CMD", "STATUS"},
        "IO_TYPE": {"AI", "DI"},
        "VENDOR_TAG": {"SIEMENS"},
    }


# -----------------------------
# categories_to_bio
# -----------------------------


def test_categories_to_bio_basic_sequence():
    """Test basic conversion of categories to BIO tags."""
    cats = ["EQUIP", "EQUIP", "SUBCOMP", "MISC", "EQUIP"]
    bio = lpt.categories_to_bio(cats)
    assert bio == ["B-EQUIP", "I-EQUIP", "B-SUBCOMP", "O", "B-EQUIP"]


def test_categories_to_bio_handles_misc_and_none():
    """Test handling of MISC and None categories."""
    cats = ["MISC", None, "EQUIP", "EQUIP", "MISC"]
    bio = lpt.categories_to_bio(cats)
    assert bio == ["O", "O", "B-EQUIP", "I-EQUIP", "O"]


# -----------------------------
# label_token
# -----------------------------


def test_label_token_vendor_takes_precedence(small_vocabs):
    """Test that VENDOR_TAG takes precedence over other categories."""
    assert lpt.label_token("Siemens", small_vocabs) == "VENDOR_TAG"


def test_label_token_io_type_detected(small_vocabs):
    """Test that IO_TYPE tokens are correctly identified."""
    assert lpt.label_token("AI", small_vocabs) == "IO_TYPE"
    assert lpt.label_token("Di", small_vocabs) == "IO_TYPE"


def test_label_token_equipment_from_vocab(small_vocabs):
    """Test that EQUIP tokens from vocabulary are correctly identified."""
    assert lpt.label_token("AHU", small_vocabs) == "EQUIP"
    assert lpt.label_token("ahu", small_vocabs) == "EQUIP"


def test_label_token_floor_pattern():
    """Test that FLOOR tokens are correctly identified by pattern."""
    vocabs = {"EQUIP": set(), "SUBCOMP": set(), "POINT_FUNC": set(), "IO_TYPE": set(), "VENDOR_TAG": set()}
    assert lpt.label_token("FL03", vocabs) == "FLOOR"
    assert lpt.label_token("F3", vocabs) == "FLOOR"
    assert lpt.label_token("Floor", vocabs) == "FLOOR"


def test_label_token_zone_room_pattern():
    """Test that ZONE/ROOM tokens are correctly identified by pattern."""
    vocabs = {"EQUIP": set(), "SUBCOMP": set(), "POINT_FUNC": set(), "IO_TYPE": set(), "VENDOR_TAG": set()}
    assert lpt.label_token("RM1203E", vocabs) == "ZONE"
    assert lpt.label_token("2130", vocabs) == "ZONE"
    assert lpt.label_token("2SE21", vocabs) == "ZONE"


def test_label_token_equip_id_pattern_numeric():
    """Test that EQUIP_ID tokens are correctly identified by numeric pattern."""
    vocabs = {"EQUIP": set(), "SUBCOMP": set(), "POINT_FUNC": set(), "IO_TYPE": set(), "VENDOR_TAG": set()}
    assert lpt.label_token("03", vocabs) == "EQUIP_ID"


# -----------------------------
# build_structured / annotate_record
# -----------------------------


def test_build_structured_aggregates_first_equip_and_zone(small_vocabs):
    """Test that build_structured correctly aggregates tokens into structured fields."""
    tokens = ["BLDG1", "FL03", "RM1203E", "AHU", "03", "SAT", "AI", "CMD"]
    cats = ["BLDG", "FLOOR", "ZONE", "EQUIP", "EQUIP_ID", "SUBCOMP", "IO_TYPE", "POINT_FUNC"]

    structured = lpt.build_structured(tokens, cats)

    assert structured["bldg"] == "BLDG1"
    assert structured["floor"] == "FL03"
    assert structured["zone"] == "RM1203E"
    assert structured["equip"] == "AHU"
    assert structured["equip_id"] == "03"
    assert structured["subcomp"] == "SAT"
    assert structured["io_type"] == "AI"
    assert structured["point_func"] == "CMD"


def test_annotate_record_end_to_end_small_example(small_vocabs, monkeypatch):
    """Test end-to-end annotation of a small example record."""
    # Use a trivial tokenizer to isolate labelling logic if needed,
    # but here we rely on the real tokenizer via point_label.
    raw = {
        "point_label": "AHU-03.SAT_AI",
        "building_id": "B1",
        "source_file": "file.csv",
        "point_label_col": "Label",
    }

    annotated = lpt.annotate_record(raw, small_vocabs)

    assert annotated["point_label"] == "AHU-03.SAT_AI"
    assert annotated["building_id"] == "B1"
    assert annotated["tokens"] == ["AHU", "03", "SAT", "AI"]
    assert annotated["token_labels"] == ["EQUIP", "EQUIP_ID", "SUBCOMP", "IO_TYPE"]
    assert annotated["bio_tags"] == ["B-EQUIP", "B-EQUIP_ID", "B-SUBCOMP", "B-IO_TYPE"]
    assert annotated["structured"]["equip"] == "AHU"
    assert annotated["structured"]["equip_id"] == "03"
    assert annotated["structured"]["subcomp"] == "SAT"
    assert annotated["structured"]["io_type"] == "AI"
