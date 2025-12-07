"""
Tokenization utilities for BMS labels.
"""

import re

DELIM_RE = re.compile(r"[ _\.\-\/:]+")  # separators: space, _, ., -, /, :


def split_alpha_num(token: str):
    """
    Split token on transitions between letters and digits.
    Example: 'RM1203E' -> ['RM', '1203', 'E']
    """
    parts = re.findall(r"[A-Za-z]+|\d+|[^A-Za-z0-9]", token)
    return [p for p in parts if p.strip()]


def tokenize(label: str):
    """Tokenize a BMS point label into meaningful tokens."""
    rough = [t for t in DELIM_RE.split(label) if t.strip()]
    tokens = []
    for t in rough:
        tokens.extend(split_alpha_num(t))
    return tokens
