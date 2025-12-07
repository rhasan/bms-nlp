"""
Automatic vocabulary extraction for Building Management System (BMS) point names.

This module analyses a large collection of BMS point names and automatically
builds compact vocabularies for different semantic groups, such as equipment
types, measurement types and point functions. The goal is to discover the
most common and useful terms directly from real data, instead of maintaining
these lists manually.

The module assumes as input a JSONL file where each line describes a single
point with at least:
- point_label: the original point name as a string
- building_id: an optional building identifier

Typical point labels look like:
- "AHU-03.SAT_AI"
- "ZONE.AHU01.RM3218:VLV1 COMD"
- "CMU/SCSC Gates/Seventh Floor/... CRAC-9/% Capacity"

The main steps are:

1. Tokenisation of point names
   Each point name is split into smaller parts (tokens) using the shared
   tokenizer from `src.bms.tokenizer`. The tokenizer:
   - splits on characters such as underscores, dots, dashes, slashes and spaces;
   - splits between letters and digits (e.g. "RM1203E" becomes "RM", "1203", "E").
   All tokens are converted to uppercase for counting and analysis.

2. Counting tokens across buildings
   For every token, the module tracks:
   - how often it appears in the whole dataset (global frequency);
   - in how many different buildings it appears (building support);
   - how often it is followed by a numeric token (e.g. "AHU 03", "VAV 12").
   This information is stored in counters and is later used to filter out
   rare or building-specific artefacts.

3. Seed vocabularies and thresholds
   The module starts from a small set of hand-picked "seed" terms that are
   known to be typical examples of:
   - equipment types (e.g. AHU, VAV, FCU),
   - subcomponents or measured quantities (e.g. SAT, TEMP, FLOW),
   - point functions (e.g. CMD, STATUS, RUN).
   It then applies global thresholds such as:
   - minimum number of occurrences in the dataset;
   - minimum number of distinct buildings where the token appears.
   These constraints help to focus on stable, widely used terms instead
   of one-off abbreviations or typos.

4. Heuristic classification of tokens
   Using the collected statistics and simple rules, tokens are assigned
   to one of several groups:
   - IO_TYPE: well-defined input/output types like AI, AO, DI, DO;
   - VENDOR_TAG: vendor-related tokens such as SIEMENS or BACNET;
   - EQUIP: likely equipment types (short uppercase tokens, often followed
     by a number that looks like an equipment ID, and not in a stopword list);
   - SUBCOMP: measurement-related tokens (TEMP, FLOW, PRESS, etc.);
   - POINT_FUNC: control or status functions (CMD, STATUS, START, STOP, etc.).
   Tokens that do not satisfy any of these criteria are not added to the
   vocabularies and remain only in the raw frequency statistics.

5. Scoring and trimming vocabularies
   For each candidate token in the EQUIP, SUBCOMP and POINT_FUNC groups, the
   module computes a simple score based on:
   - how often the token appears (frequency),
   - in how many buildings it appears (building support),
   - for equipment candidates, how often it precedes a numeric ID.
   These scores are used to:
   - keep only the top-N equipment tokens (to avoid overly large vocabularies);
   - discard very low-scoring subcomponents and point functions.
   This results in smaller, higher-quality vocabularies that are more suitable
   for rule-based labelling and later model training.

6. Output file
   The module writes a JSON file (bms_vocabs.json) containing:
   - equip_vocab: sorted list of discovered equipment tokens;
   - subcomp_vocab: sorted list of discovered subcomponent / measurement tokens;
   - point_func_vocab: sorted list of discovered point function tokens;
   - io_type_vocab: sorted list of IO type tokens;
   - vendor_vocab: sorted list of vendor tokens;
   - frequency: a full dictionary of all token frequencies (for inspection);
   - stats: basic statistics such as number of distinct tokens and buildings.
   It also prints a short summary to the console and shows the "weakest" equipment
   candidates, which can be reviewed and optionally added to a manual blacklist.

In summary, this module provides a data-driven way to bootstrap and maintain
vocabularies for BMS point name analysis. It does not rely on any machine learning
model, only on token counts and simple heuristics, so it remains transparent and
easy to adjust by domain experts. The resulting vocabularies are used by the
labelling module to assign semantic categories to tokens in individual point names.
"""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Set

from src.bms.tokenizer import tokenize

###############################################
# Global thresholds & seed vocabularies
###############################################

# Frequency and building-support thresholds
MIN_GLOBAL_FREQ = 10  # token must appear at least this many times overall
MIN_BUILDINGS = 2  # token must appear in at least this many buildings
MIN_EQUIP_NUMID_BIGRAM = 5  # times token is followed by a numeric token

# Seeds: we trust these as prototypical examples
SEED_EQUIP = {"AHU", "VAV", "FCU", "CRAC", "MAU", "EF", "SF", "HWP", "PUMP", "CHW", "HHW", "HX", "FAN", "FANCOIL"}

SEED_SUBCOMP = {"SAT", "DAT", "RAT", "MAT", "OAT", "TEMP", "FLOW", "POS", "SPEED", "PRESS", "PRESSURE", "STATIC"}

SEED_POINT_FUNC = {
    "CMD",
    "COMD",
    "STATUS",
    "START",
    "STOP",
    "RUN",
    "ENABLE",
    "ALARM",
    "ALM",
    "MODE",
    "PROOF",
    "DAY",
    "NIGHT",
}

# IO types are well-defined
KNOWN_IO = {"AI", "AO", "DI", "DO", "AV", "BV", "UI", "UO"}

# Vendor hints
KNOWN_VENDOR_HINTS = {"JCI", "SIEMENS", "BAC", "BACNET", "HONEYWELL", "TRANE", "N2", "SCHNEIDER"}

# Words we *don't* want to classify as equipment, even if short & uppercase
EQUIP_STOPWORDS = {
    "AIR",
    "FLOW",
    "SUP",
    "SUPPLY",
    "RET",
    "RETURN",
    "ZONE",
    "HOT",
    "COLD",
    "HEAT",
    "COOL",
    "TEMP",
    "MODE",
    "FILTER",
    "ALARM",
    "ALM",
    "RUN",
    "START",
    "STOP",
    "DAY",
    "NIGHT",
}

# Optional: explicit blacklist if you later find bad equipment tokens
EQUIP_BLACKLIST: set[str] = set()


###############################################
# Scoring helpers (for trimming)
###############################################


def score_equip(stats):
    """
    Heuristic scoring for equipment tokens:
    freq + 2 * buildings + 3 * numid_bigrams
    """
    return stats["freq"] + 2 * stats["buildings"] + 3 * stats.get("numid_bigrams", 0)


def score_subcomp(stats):
    """
    Heuristic scoring for subcomponents:
    freq + 2 * buildings
    """
    return stats["freq"] + 2 * stats["buildings"]


def score_pointfunc(stats):
    """
    Heuristic scoring for point functions:
    freq + buildings
    """
    return stats["freq"] + stats["buildings"]


###############################################
# Helper functions for category heuristics
###############################################


def is_io_type(tok: str) -> bool:
    """Check if token is a known IO type."""
    return tok in KNOWN_IO


def is_vendor(tok: str) -> bool:
    """Check if token is likely a vendor tag."""
    if tok in KNOWN_VENDOR_HINTS:
        return True
    # Very light heuristic: BACNET-like
    if tok.endswith("NET") and len(tok) <= 8:
        return True
    return False


def passes_global_thresholds(tok: str, token_counter: Counter, token_buildings: dict) -> bool:
    """Check if token passes global frequency and building-support thresholds."""
    freq = token_counter[tok]
    building_support = len(token_buildings.get(tok, set()))
    return (freq >= MIN_GLOBAL_FREQ) and (building_support >= MIN_BUILDINGS)


def likely_equip(tok: str, token_counter: Counter, token_buildings: dict, token_numid_bigram: Counter) -> bool:
    """
    Decide if a token is likely an equipment type.
    Uses:
      - seeds
      - frequency + building support
      - uppercase short tokens
      - pattern: often followed by numeric ID (AHU 01, VAV 12, CRAC 9)
    """
    t = tok

    if t in EQUIP_BLACKLIST:
        return False

    # Always accept seeds if they appear at least once
    if t in SEED_EQUIP and token_counter[t] > 0:
        return True

    # Must pass global thresholds
    if not passes_global_thresholds(t, token_counter, token_buildings):
        return False

    # Shape: short uppercase alphabetic tokens
    if not t.isalpha():
        return False
    if not t.isupper():
        return False
    if not (2 <= len(t) <= 6):
        return False
    if t in EQUIP_STOPWORDS:
        return False

    # Evidence: token is often followed by numeric token
    numid_support = token_numid_bigram.get(t, 0)
    if numid_support >= MIN_EQUIP_NUMID_BIGRAM:
        return True

    # As a fallback, require it to be in many buildings & medium freq
    # (already ensured by passes_global_thresholds)
    return True


def likely_subcomponent(tok: str, token_counter: Counter, token_buildings: dict) -> bool:
    """
    Subcomponents are measurement-ish terms and abbreviations.
    """
    t = tok

    # Seeds first
    if t in SEED_SUBCOMP and token_counter[t] > 0:
        return True

    if not passes_global_thresholds(t, token_counter, token_buildings):
        return False

    measurement_keywords = ["TEMP", "FLOW", "PRESS", "HUM", "SPEED", "POS", "LEVEL", "STATIC"]
    if any(k in t for k in measurement_keywords):
        return True

    # Tokens like SAT/DAT/RAT/MAT/OAT (already covered by seeds)
    # but keep them if uppercase and short
    if t.isupper() and len(t) <= 4 and t.endswith("T"):
        return True

    return False


def likely_point_func(tok: str, token_counter: Counter, token_buildings: dict) -> bool:
    """
    Commands or states: CMD, STATUS, RUN, START, etc.
    """
    t = tok

    # Seeds first
    if t in SEED_POINT_FUNC and token_counter[t] > 0:
        return True

    if not passes_global_thresholds(t, token_counter, token_buildings):
        return False

    fn_keywords = [
        "CMD",
        "COMD",
        "STAT",
        "STATUS",
        "START",
        "STOP",
        "ENABLE",
        "ENBL",
        "ALARM",
        "ALM",
        "MODE",
        "PROOF",
        "RUN",
    ]
    if any(t == k or t.startswith(k) for k in fn_keywords):
        return True

    # Also consider DAY/NIGHT if they appear often enough in building states
    if t in {"DAY", "NIGHT"}:
        return True

    return False


###############################################
# Main vocabulary extraction
###############################################


def extract_vocab(jsonl_path: Path):
    """Extract BMS vocabularies from a JSONL file of point labels."""
    # All stats are collected on UPPERCASE tokens
    token_counter: Counter[str] = Counter()  # token -> freq
    token_buildings: dict[str, set[str]] = defaultdict(set)  # token -> set(building_ids)
    token_numid_bigram: Counter[str] = Counter()  # token -> count of (token, numeric) bigrams
    per_building_counter: dict[str, Counter[str]] = defaultdict(Counter)  # building -> Counter(token)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            lbl = obj["point_label"]
            bldg = obj.get("building_id", "unknown")

            toks = tokenize(lbl)
            if not toks:
                continue

            toks_upper = [t.upper() for t in toks]

            token_counter.update(toks_upper)
            per_building_counter[bldg].update(toks_upper)

            for t in set(toks_upper):
                token_buildings[t].add(bldg)

            # record bigrams (TOKEN, NUMERIC_TOKEN) on original tokens
            for i in range(len(toks) - 1):
                t_curr = toks[i].upper()
                t_next = toks[i + 1]
                if t_next.isdigit():
                    token_numid_bigram[t_curr] += 1

    # Candidate collections (with stats)
    equip_candidates = {}
    subcomp_candidates = {}
    pointfunc_candidates = {}

    io_vocab = set()
    vendor_vocab = set()

    # Collect candidates
    for tok, freq in token_counter.items():
        t = tok  # already uppercase

        # IO first (very strict & small set)
        if is_io_type(t):
            io_vocab.add(t)
            continue

        # Vendor second
        if is_vendor(t):
            vendor_vocab.add(t)
            continue

        # Point function
        if likely_point_func(t, token_counter, token_buildings):
            pointfunc_candidates[t] = {"freq": freq, "buildings": len(token_buildings.get(t, []))}
            continue

        # Subcomponent
        if likely_subcomponent(t, token_counter, token_buildings):
            subcomp_candidates[t] = {"freq": freq, "buildings": len(token_buildings.get(t, []))}
            continue

        # Equipment
        if likely_equip(t, token_counter, token_buildings, token_numid_bigram):
            equip_candidates[t] = {
                "freq": freq,
                "buildings": len(token_buildings.get(t, [])),
                "numid_bigrams": token_numid_bigram.get(t, 0),
            }
            continue

    # === Scoring and trimming ===

    # 1) Equipment: keep only top K by score to improve precision
    MAX_EQUIP = 150  # adjust if you want more/less
    sorted_equip = sorted(
        equip_candidates.items(),
        key=lambda kv: score_equip(kv[1]),
        reverse=True,
    )
    equip_vocab = {tok for tok, _stats in sorted_equip[:MAX_EQUIP]}

    # 2) Subcomponents: trim by minimum score
    MIN_SUBCOMP_SCORE = 15
    subcomp_vocab = {tok for tok, stats in subcomp_candidates.items() if score_subcomp(stats) >= MIN_SUBCOMP_SCORE}

    # 3) Point functions: trim by minimum score
    MIN_POINTFUNC_SCORE = 5
    point_func_vocab = {
        tok for tok, stats in pointfunc_candidates.items() if score_pointfunc(stats) >= MIN_POINTFUNC_SCORE
    }

    # === Optional: print bottom equipment candidates for inspection ===
    if sorted_equip:
        bottom_equip = sorted_equip[-30:]
        print("\nBottom 30 equipment candidates (for potential blacklist):")
        for tok, stats in bottom_equip:
            print(
                f"  {tok:10s}  freq={stats['freq']}, buildings={stats['buildings']}, numid_bigrams={stats['numid_bigrams']}"
            )

    # Build output JSON
    vocabs = {
        "frequency": dict(token_counter),
        "equip_vocab": sorted(equip_vocab),
        "subcomp_vocab": sorted(subcomp_vocab),
        "point_func_vocab": sorted(point_func_vocab),
        "io_type_vocab": sorted(io_vocab),
        "vendor_vocab": sorted(vendor_vocab),
        "stats": {
            "num_tokens": len(token_counter),
            "num_buildings": len(per_building_counter),
        },
    }

    return vocabs


###############################################
# Run and write output file
###############################################


def main():
    """Main entry point to extract vocabularies and write to JSON file."""
    INPUT = Path(os.getenv("PARSER_OUTPUT_DIR", "data/output/point-name-parser")) / "all_points.jsonl"
    OUTPUT = Path(os.getenv("PARSER_OUTPUT_DIR", "data/output/point-name-parser")) / "bms_vocabs.json"

    vocabs = extract_vocab(INPUT)
    with open(OUTPUT, "w") as f:
        json.dump(vocabs, f, indent=2)

    print("\nDone! Created:", OUTPUT)
    print("Summary:")
    print("  tokens:", vocabs["stats"]["num_tokens"])
    print("  buildings:", vocabs["stats"]["num_buildings"])
    print("  equip_vocab:", len(vocabs["equip_vocab"]))
    print("  subcomp_vocab:", len(vocabs["subcomp_vocab"]))
    print("  point_func_vocab:", len(vocabs["point_func_vocab"]))
    print("  io_type_vocab:", len(vocabs["io_type_vocab"]))
    print("  vendor_vocab:", len(vocabs["vendor_vocab"]))


if __name__ == "__main__":
    main()
