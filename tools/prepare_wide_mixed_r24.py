"""Seal the two prior failures and fresh matched wide/swapped placements."""
from fit_wide_mixed_r24 import MOTION, NECK, PARENT, STUDY
from prepare_wide_mixed_r20 import prepare


if __name__ == "__main__":
    prepare(PARENT, STUDY, (NECK,), (MOTION,), {NECK: 0, MOTION: 2}, "SA_A_wide_mixed_r24",
            2611232100, 2611232700, "Swapped 04 bottle-neck confirmation and normal 13 mid-pour trajectory.")
