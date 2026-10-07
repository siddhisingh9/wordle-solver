import math

import numpy as np
import pytest

from wordle.entropy import (
    entropy_of_group_weights,
    grouped_guess_entropies,
    guess_entropies,
    pattern_group_weights,
)
from wordle.feedback import NUM_PATTERNS, feedback
from wordle.pattern_matrix import compute_pattern_matrix

WORDS = ["crane", "crate", "trace", "slate", "abbey", "babes", "speed", "abide", "hello", "lolly"]
RANDOM_SEED = 2022


@pytest.fixture
def matrix() -> np.ndarray:
    return compute_pattern_matrix(WORDS, WORDS)


def reference_entropy(guess: str, candidates: list[str], weights: list[float]) -> float:
    totals: dict[int, float] = {}
    for candidate, weight in zip(candidates, weights):
        pattern = feedback(guess, candidate)
        totals[pattern] = totals.get(pattern, 0.0) + weight
    total = sum(totals.values())
    return -sum(w / total * math.log2(w / total) for w in totals.values())


def test_group_weights_sum_candidate_weights_per_pattern() -> None:
    weights = pattern_group_weights(np.array([0, 5, 5, 242]), np.array([0.1, 0.2, 0.3, 0.4]))
    assert weights.shape == (NUM_PATTERNS,)
    assert weights[5] == pytest.approx(0.5)
    assert weights[242] == pytest.approx(0.4)


def test_entropy_of_uniform_split_is_log2_of_group_count() -> None:
    group_weights = np.zeros(NUM_PATTERNS)
    group_weights[:4] = 1.0
    assert entropy_of_group_weights(group_weights) == pytest.approx(2.0)


def test_entropy_of_single_group_is_zero() -> None:
    group_weights = np.zeros(NUM_PATTERNS)
    group_weights[7] = 3.0
    assert entropy_of_group_weights(group_weights) == pytest.approx(0.0)


def test_entropy_uses_weights_not_counts() -> None:
    group_weights = np.zeros(NUM_PATTERNS)
    group_weights[0], group_weights[1] = 0.9, 0.1
    expected = -(0.9 * math.log2(0.9) + 0.1 * math.log2(0.1))
    assert entropy_of_group_weights(group_weights) == pytest.approx(expected)


def test_vectorised_entropies_match_reference(matrix: np.ndarray) -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    weights = rng.random(len(WORDS)) + 0.01
    candidates = np.arange(len(WORDS))
    entropies = guess_entropies(matrix, candidates, weights)
    for g, guess in enumerate(WORDS):
        assert entropies[g] == pytest.approx(reference_entropy(guess, WORDS, list(weights)))


def test_guess_subset_and_candidate_subset(matrix: np.ndarray) -> None:
    candidates = np.array([0, 1, 2, 3])
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    entropies = guess_entropies(matrix, candidates, weights, guess_indices=np.array([4, 0]))
    subset = [WORDS[i] for i in candidates]
    assert entropies[0] == pytest.approx(reference_entropy("abbey", subset, list(weights)))
    assert entropies[1] == pytest.approx(reference_entropy("crane", subset, list(weights)))


def test_grouped_entropies_match_separate_calls(matrix: np.ndarray) -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    candidates = np.arange(len(WORDS))
    weights = rng.random(len(WORDS)) + 0.01
    group_ids = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
    grouped = grouped_guess_entropies(matrix, candidates, weights, group_ids, 3)
    for group in range(3):
        members = group_ids == group
        separate = guess_entropies(matrix, candidates[members], weights[members])
        np.testing.assert_allclose(grouped[:, group], separate, atol=1e-9)
