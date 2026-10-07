from pathlib import Path

import numpy as np
import pytest

from wordle.priors import (
    PLURAL_PENALTY,
    PRIORS_FILE,
    WEIGHT_FLOOR,
    compute_priors,
    frequency_weights,
    is_simple_plural,
    load_priors,
    plural_multipliers,
)

WORDS = ["about", "crane", "hopes", "boxes", "glass", "fungi", "geese", "xylyl", "qajaq"]


def test_priors_are_a_probability_distribution_with_no_zeros() -> None:
    priors = compute_priors(WORDS)
    assert priors.sum() == pytest.approx(1.0)
    assert (priors > 0).all()


def test_weights_never_drop_below_the_floor() -> None:
    weights = frequency_weights(np.array([0.0, -5.0, -1000.0]))
    assert (weights >= WEIGHT_FLOOR).all()
    assert weights[-1] == WEIGHT_FLOOR


def test_frequency_weights_increase_with_frequency() -> None:
    weights = frequency_weights(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert (np.diff(weights) > 0).all()


def test_common_words_outweigh_obscure_words() -> None:
    priors = dict(zip(WORDS, compute_priors(WORDS)))
    assert priors["about"] > priors["crane"] > priors["qajaq"]


@pytest.mark.parametrize("word", ["hopes", "boxes", "cards", "games"])
def test_simple_plurals_are_detected(word: str) -> None:
    assert is_simple_plural(word)


@pytest.mark.parametrize("word", ["glass", "geese", "fungi", "crane", "abyss"])
def test_non_simple_plurals_are_not_penalised(word: str) -> None:
    assert not is_simple_plural(word)


def test_plural_penalty_multiplies_but_never_removes() -> None:
    multipliers = dict(zip(WORDS, plural_multipliers(WORDS)))
    assert multipliers["hopes"] == PLURAL_PENALTY
    assert multipliers["glass"] == 1.0
    assert compute_priors(WORDS)[WORDS.index("hopes")] > 0


def test_priors_are_cached(tmp_path: Path) -> None:
    first = load_priors(WORDS, tmp_path)
    assert (tmp_path / PRIORS_FILE).exists()
    np.testing.assert_array_equal(load_priors(WORDS, tmp_path), first)
