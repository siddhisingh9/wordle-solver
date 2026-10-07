import numpy as np
import pytest

from wordle.feedback import feedback, string_to_pattern
from wordle.pattern_matrix import compute_pattern_matrix
from wordle.state import CandidateState, InconsistentFeedbackError

WORDS = ["crane", "crate", "trace", "slate", "abbey", "babes"]
PRIORS = np.array([0.3, 0.2, 0.2, 0.1, 0.1, 0.1])


@pytest.fixture
def state() -> CandidateState:
    return CandidateState.initial(compute_pattern_matrix(WORDS, WORDS), PRIORS)


def words_in(state: CandidateState) -> list[str]:
    return [WORDS[i] for i in state.indices]


def test_initial_state_contains_every_word(state: CandidateState) -> None:
    assert state.size == len(WORDS)
    assert state.probabilities.sum() == pytest.approx(1.0)


def test_filter_keeps_exactly_the_consistent_candidates(state: CandidateState) -> None:
    guess = WORDS.index("crane")
    pattern = feedback("crane", "crate")
    filtered = state.filter(guess, pattern)
    assert words_in(filtered) == [w for w in WORDS if feedback("crane", w) == pattern]
    assert "crate" in words_in(filtered)


def test_filter_preserves_prior_weights(state: CandidateState) -> None:
    filtered = state.filter(WORDS.index("slate"), feedback("slate", "crate"))
    for index, weight in zip(filtered.indices, filtered.weights):
        assert weight == PRIORS[index]


def test_filter_does_not_modify_the_original_state(state: CandidateState) -> None:
    state.filter(WORDS.index("crane"), feedback("crane", "abbey"))
    assert state.size == len(WORDS)


def test_all_green_leaves_only_the_guess(state: CandidateState) -> None:
    filtered = state.filter(WORDS.index("abbey"), string_to_pattern("GGGGG"))
    assert words_in(filtered) == ["abbey"]


def test_impossible_feedback_raises(state: CandidateState) -> None:
    with pytest.raises(InconsistentFeedbackError):
        state.filter(WORDS.index("crane"), string_to_pattern("GGGGY"))


def test_most_likely_orders_by_probability(state: CandidateState) -> None:
    top = state.most_likely(2)
    assert [WORDS[i] for i, _ in top] == ["crane", "crate"]
    assert top[0][1] == pytest.approx(0.3)
