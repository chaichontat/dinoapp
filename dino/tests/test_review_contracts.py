import numpy as np
import pytest

from dino.review_contracts import combined_uncertainty


def test_uncertainty_uses_saved_cutoffs_and_keeps_exact_cutoff_infinite():
    result = combined_uncertainty(
        np.array([.25, .75, 1.]), np.array([.5, 1., .75]),
        brdu_threshold=.25, edu_threshold=.75,
    )
    np.testing.assert_array_equal(result, [np.inf, 4., np.inf])


@pytest.mark.parametrize("brdu,edu,cutoff", [
    ([], [], .5), ([.2], [.2, .4], .5), ([np.nan], [.2], .5),
    ([-.1], [.2], .5), ([.2], [.2], 0.), ([.2], [.2], np.inf),
])
def test_uncertainty_rejects_invalid_probabilities_or_cutoffs(brdu, edu, cutoff):
    with pytest.raises(ValueError):
        combined_uncertainty(np.asarray(brdu), np.asarray(edu), brdu_threshold=cutoff, edu_threshold=.5)
