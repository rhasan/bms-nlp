"""
This module extracts BMS point names from a heterogeneous collection of CSV
files and produces a clean, uniform JSONL dataset suitable for downstream
tokenization and labeling.

The module scans a directory of CSV files, loads each file, identifies which
column contains the point names, and outputs one JSON object per point. Each
record includes the point label, the column it came from, the source filename,
and a building identifier derived from the filename.

Because raw files vary widely in structure, the module uses heuristics rather
than fixed schemas. It first applies a pattern-based detector that determines
whether individual strings resemble BMS-style point names, based on features
such as uppercase abbreviations, separators (underscores, hyphens, dots), and
letter–digit mixes.

Using this detector, the loader decides whether the first row of each file is a
header. It examines the first three rows column by column. If the second and
third rows contain point-like strings in the same column while the first row
does not, the first row is treated as a header. If that pattern is not found,
a simpler fallback is used: if the first row contains no point-like strings but
the second row does, the first row is again considered a header; otherwise the
file is treated as headerless and synthetic column names are assigned.

After header handling, the module identifies the point-name column. It first
checks for known preferred column names (such as “Label” or “bas_raw”).
If none are present, it evaluates each text column by how frequently its values
look like BMS-style strings and selects the column with the strongest signal.

For each row in the chosen column, the module constructs a normalized record
containing the extracted point label and relevant metadata. All records from
all files are combined and written to a JSONL file, one point per line.

Overall, this module serves as a normalization layer that converts messy,
vendor-specific CSV dumps into a consistent corpus of BMS point names, ready for
token-level annotation, weak labeling, BIO tagging, and structured semantic
interpretation.
"""

import os
import random
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)


POINT_COLS_DICT = {
    "b3_ibm.csv": "Label",
    "ebu3b_ucsd.csv": "Johnson Controls Name",
    "ghc_cmu.csv": "bas_raw",
}

PREFERRED_POINT_COLS = {c.lower(): c for c in POINT_COLS_DICT.values()}


def is_bms_style_string(s: str) -> bool:
    """
    Heuristic: does this look like a BMS point name?

    We try to avoid returning True for generic headers like 'bas_raw', 'haystack',
    while still matching things like:
      - BLDG1_FL03_AHU2_SAT_AI
      - AHU-03.SAT
      - Effective_RA_CO_dac_AV
    """
    if not isinstance(s, str):
        return False

    s = s.strip()
    if len(s) < 3:
        return False

    has_sep = any(ch in "_.-" for ch in s)
    has_digit = any(ch.isdigit() for ch in s)
    has_alpha = any(ch.isalpha() for ch in s)
    has_upper = any(ch.isupper() for ch in s)

    # fraction of uppercase among alphabetic chars
    alpha_only = "".join(ch for ch in s if ch.isalpha())
    if alpha_only:
        upper_frac = sum(ch.isupper() for ch in alpha_only) / len(alpha_only)
    else:
        upper_frac = 0.0

    mostly_upper = upper_frac >= 0.7

    # BMS-ish patterns:
    #  - mostly uppercase tokens (EBU3B.CHWP3-VFD.ACC-TIME)
    #  - has separator AND (digit or uppercase)   (BLDG1_FL03, Effective_RA_CO_dac_AV)
    #  - has digit AND uppercase (AHU3_SAT_AI)
    if mostly_upper:
        return True
    if has_sep and (has_digit or has_upper):
        return True
    if has_digit and has_upper:
        return True

    # Everything else (like 'bas_raw', 'haystack', 'name', 'units') → False
    return False


def detect_header(df: pd.DataFrame) -> bool:
    """
    Decide whether the first row should be treated as header.

    Main heuristic (when we have at least 3 rows):
      - For each column:
          - Let b0, b1, b2 = is_bms_style_string(row0[col]), row1[col], row2[col]
          - If (!b0) and b1 and b2 for ANY column -> treat first row as header.
        Intuition: second and third rows look like data (point names) in the
        same column, but the first row in that column does not.

    Fallback (for short files):
      - If only 1 row: assume *no* header (cannot tell).
      - If 2 rows:
          - If row0 has 0 point-like cells and row1 has >0 -> header.
          - Else -> no header.
    """
    n_rows = df.shape[0]
    if n_rows == 0:
        return False

    # --- Case A: we have at least 3 rows -> use column-wise pattern ---
    if n_rows >= 3:
        # Work column-wise
        for col in df.columns:
            col_vals = df[col].astype(str)
            b0 = is_bms_style_string(col_vals.iloc[0])
            b1 = is_bms_style_string(col_vals.iloc[1])
            b2 = is_bms_style_string(col_vals.iloc[2])

            # Candidate point column: row1 & row2 look like points, row0 does not.
            if (not b0) and b1 and b2:
                return True  # first row is header

        # If no column satisfies the pattern, fall through to fallback below.

    # --- Case B: fallback for small or ambiguous cases ---
    first_row = df.iloc[0].astype(str)
    first_point_like = sum(is_bms_style_string(v) for v in first_row)

    if n_rows == 1:
        # With only one row we can't be sure; default to "no header"
        return False

    # n_rows >= 2 here
    second_row = df.iloc[1].astype(str)
    second_point_like = sum(is_bms_style_string(v) for v in second_row)

    # Classic simple rule: if row0 has no point-like strings at all,
    # and row1 does, it's likely a header row.
    if first_point_like == 0 and second_point_like > 0:
        return True

    # Otherwise, treat first row as data.
    return False


def guess_point_label_column(df: pd.DataFrame) -> str | None:
    """Try to guess which column contains point labels."""
    # 1) Preferred name if exists (case-insensitive)
    lower_cols = {c.lower(): c for c in df.columns}
    for pref in PREFERRED_POINT_COLS:
        if pref in lower_cols:
            return lower_cols[pref]

    # 2) Score all object-like columns by BMS-style frequency
    best_col = None
    best_score = -1
    for col in df.columns:
        if not pd.api.types.is_object_dtype(df[col]):
            continue
        series = df[col].dropna().astype(str)
        if len(series) == 0:
            continue
        sample = series.sample(min(200, len(series)), random_state=42)
        score = sample.apply(is_bms_style_string).mean()
        if score > best_score:
            best_score = score
            best_col = col

    # Require a minimum reasonable score (tunable)
    return best_col if best_score > 0.2 else None


def derive_building_id_from_filename(fname: str) -> str:
    """
    Very simple heuristic:
    - Strip extension
    - Remove leading digits + underscore/hyphen
    """
    stem = Path(fname).stem
    # e.g. "01_office_singapore" -> "office_singapore"
    stem = re.sub(r"^\d+[_\-]*", "", stem)
    return stem


def load_all_bms_points(raw_dir: Path, bms_output_file):
    """Load all BMS point names from CSV files in raw_dir and save to JSONL."""
    records = []

    for csv_path in raw_dir.glob("*.csv"):
        print(f"Processing {csv_path}")
        try:
            # Always read without header first
            df = pd.read_csv(csv_path, header=None, dtype=str)
        except Exception as e:
            print(f"Skipping {csv_path}: {e}")
            continue

        if df.empty:
            print(f"WARNING: Empty file {csv_path}, skipping.")
            continue

        # Decide if first row is header
        if detect_header(df):
            # Promote first row to header
            df.columns = df.iloc[0].astype(str)
            df = df.iloc[1:].reset_index(drop=True)
        else:
            # Synthetic column names
            df.columns = [f"col_{i}" for i in range(df.shape[1])]

        point_col = guess_point_label_column(df)
        if point_col is None:
            print(f"WARNING: No point label column found in {csv_path}")
            continue

        building_id = derive_building_id_from_filename(csv_path.name)

        tmp = df.copy()
        tmp["point_label"] = tmp[point_col].astype(str)
        tmp["point_label_col"] = point_col
        tmp["building_id"] = building_id
        tmp["source_file"] = csv_path.name

        records.append(tmp[["building_id", "source_file", "point_label", "point_label_col"]])

    if not records:
        raise RuntimeError("No valid CSV files with point labels found.")

    full_df = pd.concat(records, ignore_index=True)
    full_df.to_json(bms_output_file, orient="records", lines=True)


def sample_one_point_per_building(jsonl_path, seed=42):
    """
    Read the extracted BMS JSONL file, select one random point name for each
    building_id, and return the selected point names separated by newlines.
    """
    # Fix randomness for reproducibility
    random.seed(seed)

    # Load JSONL into a DataFrame
    df = pd.read_json(jsonl_path, lines=True)

    if "building_id" not in df.columns or "point_label" not in df.columns:
        raise ValueError("JSONL must contain 'building_id' and 'point_label' columns.")

    # Group by building and sample one point per building
    samples = df.groupby("building_id")["point_label"].apply(lambda s: s.sample(1, random_state=seed).iloc[0])

    # Join into newline-separated string
    return "\n".join(samples.tolist())


def main():
    """Entry point for pattern parser."""
    bms_input_directory = Path(os.getenv("BMS_INPUT_DIR", "data/bms-fierro/buildings"))
    bms_output_file = Path(os.getenv("PARSER_OUTPUT_DIR", "data/output/point-name-parser")) / "all_points.jsonl"

    load_all_bms_points(raw_dir=bms_input_directory, bms_output_file=bms_output_file)
    print(sample_one_point_per_building(jsonl_path=bms_output_file))


if __name__ == "__main__":
    main()
