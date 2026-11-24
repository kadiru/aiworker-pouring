"""Seal the two shared detector changes, one motion change and four targeted pours."""
from diagnose_wide_mixed_r25 import PARENT, STUDY, TARGETS, MOTION
from prepare_wide_mixed_r20 import prepare


if __name__ == "__main__":
    losses = {**dict.fromkeys(TARGETS,0), MOTION:4}
    prepare(PARENT, STUDY, ("12_normal","14_normal"), (MOTION,), losses, "SA_A_wide_mixed_r25",
            2611241100, 2611241700, "Bottle detection on pairs 12 and 00/14, pair 12 cup proposal and normal 09 coarse motion.",
            cup_key="12_normal")
