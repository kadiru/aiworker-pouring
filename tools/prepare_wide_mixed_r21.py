"""Seal the three original failed placements and a fresh matched benchmark."""
from fit_wide_mixed_r21 import NECK, MOTION, PARENT, STUDY
from prepare_wide_mixed_r20 import prepare

if __name__ == "__main__":
    prepare(PARENT, STUDY, NECK, MOTION, {"01_normal": 0, "10_normal": 0, "08_swap": 2},
            "SA_A_wide_mixed_r21", 2611223100, 2611223700,
            "Normal 01/10 neck confirmation and swapped 08 terminal mouth trajectory")
