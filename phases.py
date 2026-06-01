"""Cholec80 phase definitions and label <-> id mapping.

Cholec80 has 7 surgical phases. The official annotation files use these exact
strings (CamelCase, no spaces). We fix an ordering 0..6 and use it everywhere.
"""

# Canonical order used by virtually all Cholec80 papers (TeCNO included).
PHASES = [
    "Preparation",                 # 0
    "CalotTriangleDissection",     # 1
    "ClippingCutting",             # 2
    "GallbladderDissection",       # 3
    "GallbladderPackaging",        # 4
    "CleaningCoagulation",         # 5
    "GallbladderRetraction",       # 6
]

NUM_PHASES = len(PHASES)

# name -> id   and   id -> name
PHASE2ID = {name: i for i, name in enumerate(PHASES)}
ID2PHASE = {i: name for i, name in enumerate(PHASES)}


def phase_to_id(name: str) -> int:
    """Map a phase string from the annotation file to its integer id."""
    return PHASE2ID[name.strip()]
