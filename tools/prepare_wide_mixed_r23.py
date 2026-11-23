"""Seal the targeted spill repair and 15 fresh matched pairs with frozen localization."""
from fit_wide_mixed_r23 import LAYOUT, PARENT, STUDY
from prepare_wide_mixed_r20 import prepare

if __name__ == "__main__":
    prepare(PARENT, STUDY, (), (LAYOUT,), {LAYOUT: 2}, "SA_A_wide_mixed_r23",
            2611231100, 2611231700, "Normal 05 mid-pour motion only. All localization fields and sources frozen.")
