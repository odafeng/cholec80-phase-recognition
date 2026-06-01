"""Train / val / test split for Cholec80.

Standard protocol variants:
  - 40/40   : train 1-40,  test 41-80           (original TeCNO paper)
  - 32/8/40 : train 1-32, val 33-40, test 41-80 (most common; lets us tune)

We use 32/8/40. Video ids are 1-based (video01 .. video80).
"""

TRAIN_IDS = list(range(1, 33))    # 01..32
VAL_IDS   = list(range(33, 41))   # 33..40
TEST_IDS  = list(range(41, 81))   # 41..80

ALL_IDS = TRAIN_IDS + VAL_IDS + TEST_IDS


def vid_name(i: int) -> str:
    """Zero-padded video name, e.g. 1 -> 'video01'."""
    return f"video{i:02d}"


if __name__ == "__main__":
    print(f"train: {len(TRAIN_IDS)} videos {TRAIN_IDS[0]}..{TRAIN_IDS[-1]}")
    print(f"val  : {len(VAL_IDS)} videos {VAL_IDS[0]}..{VAL_IDS[-1]}")
    print(f"test : {len(TEST_IDS)} videos {TEST_IDS[0]}..{TEST_IDS[-1]}")
