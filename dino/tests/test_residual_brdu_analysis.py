import numpy as np
import pandas as pd
import pytest

from dino.residual_brdu_analysis import summarize_field


def field():
    x = np.arange(100, dtype=float)
    return pd.DataFrame({
        "cell": [f"cell{i}" for i in range(100)],
        "dataset": "slide", "roi": "1", "tile": 1,
        "brdu_median_raw_adu": x + 1000,
        "final_median_raw_adu": 2000 + x,
        "edu_median_raw_adu": 3000 + (x >= 50) * 1000,
        "brdu_positive": x >= 50, "edu_positive": x >= 50,
    })


def test_brightness_contrast_removes_round_specific_offsets():
    row, bins = summarize_field(field())
    assert row["n_bright"] == 10
    assert row["residual_excess_adu"] == 70
    assert row["ratio_median"] == pytest.approx(0.07)
    assert row["ratio_q10"] == pytest.approx(0.07)
    assert row["slope_final_adu_per_brdu_adu"] == pytest.approx(1)
    assert bins.n.sum() == 100
    assert len(bins) == 20


def test_signed_negative_excess_is_retained():
    data = field()
    data["final_median_raw_adu"] = 2000 - np.arange(100)
    row, _ = summarize_field(data)
    assert row["ratio_median"] == pytest.approx(-0.07)


def test_nonpositive_edu_reference_is_rejected():
    data = field()
    data["edu_median_raw_adu"] = 3000
    with pytest.raises(ValueError, match="does not exceed"):
        summarize_field(data)
