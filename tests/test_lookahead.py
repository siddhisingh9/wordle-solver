import math

import numpy as np
import pytest

from wordle.feedback import feedback
from wordle.lookahead import (
    choose_guess,
    endgame_guess,
    expected_follow_up_entropy,
    lookahead_scores,
    top_guesses_by_entropy,
)
from wordle.pattern_matrix import compute_pattern_matrix
from wordle.state import CandidateState

WORDS = [
    "crane", "crate", "trace", "slate", "stale", "steal", "least", "abbey", "babes", "speed",
    "abide", "hello", "lolly", "mount", "pound", "round", "sound", "wound", "found", "bound",
]
RANDOM_SEED = 2022


def reference_entropy(guess: str, candidates: list[str], weights: list[float]) -> float:
    totals: dict[int, float] = {}
    for candidate, weight in zip(candidates, weights):
        pattern = feedback(guess, candidate)
        totals[pattern] = totals.get(pattern, 0.0) + weight
    total = sum(totals.values())
    return -sum(w / total * math.log2(w / total) for w in totals.values())


def reference_score(first_guess: str, weights: list[float]) -> float:
    groups: dict[int, list[int]] = {}
    for i, candidate in enumerate(WORDS):
        groups.setdefault(feedback(first_guess, candidate), []).append(i)
    total = sum(weights)
    follow_up = 0.0
    for members in groups.values():
        if len(members) < 3:
            continue
        group_words = [WORDS[i] for i in members]
        group_weights = [weights[i] for i in members]
        best = max(reference_entropy(g, group_words, group_weights) for g in WORDS)
        follow_up += sum(group_weights) / total * best
    return reference_entropy(first_guess, WORDS, weights) + follow_up


@pytest.fixture
def weights() -> np.ndarray:
    return np.random.default_rng(RANDOM_SEED).random(len(WORDS)) + 0.05


@pytest.fixture
def state(weights: np.ndarray) -> CandidateState:
    return CandidateState.initial(compute_pattern_matrix(WORDS, WORDS), weights)


def test_top_guesses_are_sorted_by_entropy(state: CandidateState) -> None:
    guesses, entropies = top_guesses_by_entropy(state.pattern_matrix, state.indices, state.weights, 5)
    assert len(guesses) == 5
    assert (np.diff(entropies) <= 0).all()


def test_follow_up_entropy_matches_brute_force(state: CandidateState, weights: np.ndarray) -> None:
    for first in ["crane", "sound", "abbey"]:
        g = WORDS.index(first)
        expected = reference_score(first, list(weights)) - reference_entropy(first, WORDS, list(weights))
        assert expected_follow_up_entropy(state.pattern_matrix, g, state.indices, state.weights) == pytest.approx(expected)


def test_lookahead_scores_match_brute_force(state: CandidateState, weights: np.ndarray) -> None:
    guesses, scores = lookahead_scores(state.pattern_matrix, state.indices, state.weights)
    for g, score in zip(guesses, scores):
        assert score == pytest.approx(reference_score(WORDS[g], list(weights)))


def test_choose_guess_picks_highest_lookahead_score(state: CandidateState, weights: np.ndarray) -> None:
    best_score = max(reference_score(w, list(weights)) for w in WORDS)
    assert reference_score(WORDS[choose_guess(state)], list(weights)) == pytest.approx(best_score)


def test_zero_entropy_guesses_are_never_chosen() -> None:
    matrix = compute_pattern_matrix(WORDS, WORDS)
    three = CandidateState(matrix, np.array([WORDS.index(w) for w in ["pound", "sound", "found"]]), np.ones(3))
    guesses, entropies = top_guesses_by_entropy(matrix, three.indices, three.weights, 50)
    assert (entropies > 0).all()
    assert WORDS[choose_guess(three)] not in {"crane", "crate", "abbey"}


def test_zero_time_budget_scores_only_the_highest_entropy_guess(state: CandidateState) -> None:
    guesses, scores = lookahead_scores(state.pattern_matrix, state.indices, state.weights, time_budget=0.0)
    best_by_entropy, _ = top_guesses_by_entropy(state.pattern_matrix, state.indices, state.weights, 1)
    assert len(guesses) == len(scores) == 1
    assert guesses[0] == best_by_entropy[0]


def test_generous_time_budget_matches_full_lookahead(state: CandidateState) -> None:
    assert choose_guess(state, time_budget=60.0) == choose_guess(state)


def test_endgame_guesses_most_probable_of_two(weights: np.ndarray) -> None:
    matrix = compute_pattern_matrix(WORDS, WORDS)
    two = CandidateState(matrix, np.array([3, 9]), np.array([0.2, 0.3]))
    assert endgame_guess(two) == 9
    assert choose_guess(two) == 9


def test_endgame_guesses_dominant_candidate() -> None:
    matrix = compute_pattern_matrix(WORDS, WORDS)
    dominated = CandidateState(matrix, np.array([0, 1, 2, 3]), np.array([0.1, 0.6, 0.2, 0.1]))
    assert endgame_guess(dominated) == 1


def test_no_endgame_when_uncertain() -> None:
    matrix = compute_pattern_matrix(WORDS, WORDS)
    uncertain = CandidateState(matrix, np.array([0, 1, 2, 3]), np.array([0.3, 0.3, 0.2, 0.2]))
    assert endgame_guess(uncertain) is None
