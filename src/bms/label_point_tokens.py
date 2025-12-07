"""
Weak rule-based labelling of Building Management System (BMS) point names.

This module takes raw BMS point names (such as "AHU-03.SAT_AI" or
"ZONE.AHU01.RM3218:VLV1 COMD") and converts them into a structured,
token-level representation that is easier to analyse and use in downstream
applications.

The module assumes as input a JSONL file where each line describes a single
point with at least these fields:
- point_label: the original point name as a string
- building_id: an optional building identifier
- source_file / point_label_col: optional provenance metadata

The labelling process works in several steps:

1. Tokenisation of point names
   Each point name is split into smaller parts (tokens) using separators
   such as underscores, dots, dashes, slashes and spaces, and by splitting
   between letters and digits (for example, "RM1203E" becomes "RM", "1203",
   "E"). This produces a list of tokens that roughly correspond to building,
   floor, equipment type, equipment ID, measurement type and so on.

2. Use of vocabularies
   The module loads pre-computed vocabularies from a JSON file. These
   vocabularies list known terms for:
   - equipment types (EQUIP), e.g. AHU, VAV, FCU
   - subcomponents or measured quantities (SUBCOMP), e.g. SAT, TEMP, FLOW
   - point functions (POINT_FUNC), e.g. CMD, STATUS, RUN
   - input/output types (IO_TYPE), e.g. AI, AO, DI, DO
   - vendor-specific tags (VENDOR_TAG), e.g. SIEMENS, BACNET
   These vocabularies are used as look-up tables when assigning labels to
   tokens.

3. Rule-based token labelling
   For each token, the module assigns a semantic category such as:
   - BLDG      building identifier
   - FLOOR     floor indicator
   - ZONE      room or zone identifier
   - EQUIP     equipment type
   - EQUIP_ID  equipment index or number
   - SUBCOMP   subcomponent or measured quantity
   - POINT_FUNC control or status function
   - IO_TYPE   input/output type
   - VENDOR_TAG vendor-specific tag
   - MISC      everything that does not match the above
   The assignment is based on simple rules:
   - first, the token is checked against the vocabularies;
   - then, regular expressions identify floors, rooms and IDs
     (for example, patterns like "FL03" or "RM1202E");
   - any token not matched by these rules is marked as MISC.

4. BIO sequence tagging
   In addition to plain categories, the module produces BIO tags. These
   tags mark the beginning (B-), inside (I-) or outside (O) of each
   labelled segment:
   - tokens with category MISC are labelled as "O"
   - the first token of a run of the same category is labelled "B-CATEGORY"
   - subsequent tokens with the same category are labelled "I-CATEGORY"
   This format is commonly used in sequence labelling tasks and makes
   it easier to train and evaluate machine learning models later, if
   desired.

5. Simple structured interpretation
   The module also builds a simple structured view of each point by
   aggregating token labels into high-level fields, for example:
   - building (bldg)
   - floor
   - zone
   - equipment type (equip)
   - equipment ID (equip_id)
   - subcomponent (subcomp)
   - point function (point_func)
   - IO type (io_type)
   - vendor
   This step uses straightforward heuristics, such as taking the first
   token labelled as equipment as the equipment type, and grouping all
   zone tokens into a single zone string.

6. Input and output files
   The module reads:
   - a JSONL file with all extracted points (all_points.jsonl)
   - a JSON file containing the vocabularies (bms_vocabs.json)
   Both locations can be controlled via an environment variable that
   points to the parser output directory.

   For each input line, the module writes an annotated JSON object to
   a new JSONL file (point_names_labeled.jsonl). Each output line
   contains:
   - the original point_label
   - the list of tokens
   - the list of token-level categories
   - the list of BIO tags
   - building and provenance metadata
   - the simple structured interpretation

Overall, this module provides a weak, rule-based labelling pipeline for
BMS point names. It does not require any machine learning model and is
designed to be understandable and adjustable by domain experts. The
resulting labelled dataset can be used directly for analysis, or as
training data for more advanced AI models in later stages.
"""

import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.bms.tokenizer import tokenize

# ---------------------------------------------------------
# Load vocabularies
# ---------------------------------------------------------


def load_vocabs(vocab_path: str):
    """Load vocabularies from JSON file."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    equip_vocab = set(data.get("equip_vocab", []))
    subcomp_vocab = set(data.get("subcomp_vocab", []))
    point_func_vocab = set(data.get("point_func_vocab", []))
    io_type_vocab = set(data.get("io_type_vocab", []))
    vendor_vocab = set(data.get("vendor_vocab", []))

    return {
        "EQUIP": equip_vocab,
        "SUBCOMP": subcomp_vocab,
        "POINT_FUNC": point_func_vocab,
        "IO_TYPE": io_type_vocab,
        "VENDOR_TAG": vendor_vocab,
    }


# ---------------------------------------------------------
# Regex patterns for floors / rooms / equip IDs / building
# ---------------------------------------------------------

FLOOR_PATTERNS = [
    re.compile(r"^FL?\d+$", re.I),  # F3, FL03, FL12
    re.compile(r"^Floor$", re.I),
]

ROOM_PATTERNS = [
    re.compile(r"^RM\d+[A-Z]?$", re.I),  # RM148A, RM1202E
    re.compile(r"^\d{3,4}[A-Z]?$", re.I),  # 2130, 1309, 7019E
    re.compile(r"^\d[A-Z]{1,3}\d{1,3}$", re.I),  # NEW: 2SE21 style
]

EQUIP_ID_PATTERNS = [
    re.compile(r"^\d+$"),  # 01, 2, 8
    re.compile(r"^[A-Z]?\d{2,3}[A-Z]?$"),  # 2SE21, A10, etc.
]

BUILDING_HINT_PATTERNS = [
    re.compile(r"^BLDG\d+$", re.I),
]


# ---------------------------------------------------------
# Weak rule-based labelling for a single token
# ---------------------------------------------------------


def label_token(token: str, vocabs: Dict[str, set]) -> str:
    """
    Return one of:
      BLDG, FLOOR, ZONE, EQUIP, EQUIP_ID,
      SUBCOMP, POINT_FUNC, IO_TYPE, VENDOR_TAG, MISC
    """
    t = token.upper()

    # Vendor
    if t in vocabs["VENDOR_TAG"]:
        return "VENDOR_TAG"

    # IO type
    if t in vocabs["IO_TYPE"]:
        return "IO_TYPE"

    # Equipment type
    if t in vocabs["EQUIP"]:
        return "EQUIP"

    # Subcomponent
    if t in vocabs["SUBCOMP"]:
        return "SUBCOMP"

    # Point function
    if t in vocabs["POINT_FUNC"]:
        return "POINT_FUNC"

    # Floor
    if any(p.match(token) for p in FLOOR_PATTERNS):
        return "FLOOR"

    # Room / Zone
    if any(p.match(token) for p in ROOM_PATTERNS):
        return "ZONE"

    # Equipment ID
    if any(p.match(token) for p in EQUIP_ID_PATTERNS):
        return "EQUIP_ID"

    # Building hints
    if any(p.match(token) for p in BUILDING_HINT_PATTERNS):
        return "BLDG"

    # Fallback
    return "MISC"


def weak_label_tokens(tokens: List[str], vocabs: Dict[str, set]) -> List[str]:
    """Label a list of tokens with weak rule-based categories."""
    return [label_token(tok, vocabs) for tok in tokens]


# ---------------------------------------------------------
# Category -> BIO conversion
# ---------------------------------------------------------


def categories_to_bio(categories: Sequence[str | None]) -> list[str]:
    """
    categories: e.g. ["EQUIP", "EQUIP", "SUBCOMP", "MISC", None, "EQUIP"]
    returns:         ["B-EQUIP", "I-EQUIP", "B-SUBCOMP", "O", "O", "B-EQUIP"]
    """
    bio = []
    prev_cat: Optional[str] = None

    for cat in categories:
        if cat is None or cat == "MISC":
            bio.append("O")
            prev_cat = None
            continue

        if cat != prev_cat:
            bio.append(f"B-{cat}")
        else:
            bio.append(f"I-{cat}")

        prev_cat = cat

    return bio


# ---------------------------------------------------------
# Build a simple structured interpretation
# ---------------------------------------------------------


def build_structured(tokens: List[str], cats: List[str]) -> Dict[str, Optional[str]]:
    """
    Very simple heuristic mapping from token-level categories to structured fields.
    You can always refine this later.
    """
    bldg = None
    floor = None
    zone_tokens = []
    equip = None
    equip_id = None
    subcomp = None
    point_func = None
    io_type = None
    vendor = None

    for tok, c in zip(tokens, cats):
        if c == "BLDG" and bldg is None:
            bldg = tok
        elif c == "FLOOR" and floor is None:
            floor = tok
        elif c == "ZONE":
            zone_tokens.append(tok)
        elif c == "EQUIP" and equip is None:
            equip = tok
        elif c == "EQUIP_ID" and equip_id is None:
            equip_id = tok
        elif c == "SUBCOMP":
            subcomp = tok  # last one wins
        elif c == "POINT_FUNC":
            point_func = tok  # last one wins
        elif c == "IO_TYPE":
            io_type = tok  # last one wins
        elif c == "VENDOR_TAG":
            vendor = tok

    zone = " ".join(zone_tokens) if zone_tokens else None

    return {
        "bldg": bldg,
        "floor": floor,
        "zone": zone,
        "equip": equip,
        "equip_id": equip_id,
        "subcomp": subcomp,
        "point_func": point_func,
        "io_type": io_type,
        "vendor": vendor,
    }


# ---------------------------------------------------------
# Annotate one record (one JSONL line)
# ---------------------------------------------------------


def annotate_record(raw_record: Dict[str, Any], vocabs: Dict[str, set]) -> Dict[str, Any]:
    """Annotate one raw record with tokens, labels, BIO tags, structured interpretation."""
    point_label = raw_record["point_label"]
    building_id = raw_record.get("building_id")

    tokens = tokenize(point_label)
    token_labels = weak_label_tokens(tokens, vocabs)  # coarse categories
    bio_tags = categories_to_bio(token_labels)  # BIO scheme
    structured = build_structured(tokens, token_labels)

    return {
        "point_label": point_label,
        "tokens": tokens,
        "token_labels": token_labels,
        "bio_tags": bio_tags,
        "building_id": building_id,
        "source_file": raw_record.get("source_file"),
        "point_label_col": raw_record.get("point_label_col"),
        "label_source": "rule",
        "structured": structured,
    }


# ---------------------------------------------------------
# Main: read JSONL, annotate, write JSONL
# ---------------------------------------------------------


def main():
    """Annotate all BMS point names in the input JSONL file and write to output JSONL file."""
    INPUT = Path(os.getenv("PARSER_OUTPUT_DIR", "data/output/point-name-parser")) / "all_points.jsonl"
    VOCABS = Path(os.getenv("PARSER_OUTPUT_DIR", "data/output/point-name-parser")) / "bms_vocabs.json"

    OUTPUT = Path(os.getenv("PARSER_OUTPUT_DIR", "data/output/point-name-parser")) / Path("point_names_labeled.jsonl")

    vocabs = load_vocabs(str(VOCABS))

    num_in = 0
    num_out = 0

    with INPUT.open("r", encoding="utf-8") as fin, OUTPUT.open("w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            num_in += 1
            raw_record = json.loads(line)
            annotated = annotate_record(raw_record, vocabs)
            fout.write(json.dumps(annotated) + "\n")
            num_out += 1

    print(f"Done. Read {num_in} records, wrote {num_out} annotated records to {OUTPUT}")


if __name__ == "__main__":
    main()
