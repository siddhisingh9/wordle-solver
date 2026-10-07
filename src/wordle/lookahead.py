"""Guess selection: 2-step entropy lookahead with an endgame rule for going for the win."""

from __future__ import annotations

import time

import numpy as np

from wordle.entropy import grouped_guess_entropies, guess_entropies, pattern_group_weights
from wordle.feedback import NUM_PATTERNS
from wordle.state import CandidateState

FIRST_GUESS_SHORTLIST = 50
MIN_GROUP_SIZE_FOR_FOLLOW_UP = 3
ENDGAME_MAX_CANDIDATES = 2
ENDGAME_MIN_PROBABILITY = 0.5
MIN_INFORMATIVE_ENTROPY = 1e-9


def top_guesses_by_entropy(
    pattern_matrix: np.ndarray, candidate_indices: np.ndarray, weights: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """The `count` highest-entropy guesses over the candidates, as (guess indices, entropies), best first.

    Guesses with zero entropy are never shortlisted: they cannot split the candidates, so playing one
    would make no progress, yet rounding could otherwise let one tie the best lookahead score.
    """
    entropies = guess_entropies(pattern_matrix, candidate_indices, weights)
    count = min(count, int((entropies > MIN_INFORMATIVE_ENTROPY).sum()))
    shortlist = np.argpartition(-entropies, count - 1)[:count]
    shortlist = shortlist[np.argsort(-entropies[shortlist], kind="stable")]
    return shortlist, entropies[shortlist]


def expected_follow_up_entropy(
    pattern_matrix: np.ndarray, first_guess: int, candidate_indices: np.ndarray, weights: np.ndarray
) -> float:
    """Sum over feedback groups k of p_k * H_best(k), where H_best(k) is the best entropy any guess achieves on group k.

    Groups with fewer than MIN_GROUP_SIZE_FOR_FOLLOW_UP candidates contribute 0.
    """
    patterns = pattern_matrix[first_guess, candidate_indices]
    group_sizes = np.bincount(patterns, minlength=NUM_PATTERNS)
    in_large_group = group_sizes[patterns] >= MIN_GROUP_SIZE_FOR_FOLLOW_UP
    if not in_large_group.any():
        return 0.0
    group_patterns, group_ids = np.unique(patterns[in_large_group], return_inverse=True)
    follow_up_entropies = grouped_guess_entropies(
        pattern_matrix,
        candidate_indices[in_large_group],
        weights[in_large_group],
        group_ids,
        len(group_patterns),
    )
    best_per_group = follow_up_entropies.max(axis=0)
    group_probabilities = pattern_group_weights(patterns, weights)[group_patterns] / weights.sum()
    return float(group_probabilities @ best_per_group)


def lookahead_scores(
    pattern_matrix: np.ndarray,
    candidate_indices: np.ndarray,
    weights: np.ndarray,
    time_budget: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Score(g1) = H(g1) + sum_k p_k H_best(k) for the top FIRST_GUESS_SHORTLIST guesses, as (guesses, scores).

    With a time budget in seconds, shortlisted guesses are scored best-entropy first and scoring stops
    once the budget is spent (always after at least one), returning only the guesses that were scored.
    """
    deadline = None if time_budget is None else time.perf_counter() + time_budget
    shortlist, entropies = top_guesses_by_entropy(
        pattern_matrix, candidate_indices, weights, FIRST_GUESS_SHORTLIST
    )
    follow_ups: list[float] = []
    for guess in shortlist:
        follow_ups.append(expected_follow_up_entropy(pattern_matrix, guess, candidate_indices, weights))
        if deadline is not None and time.perf_counter() >= deadline:
            break
    scored = len(follow_ups)
    return shortlist[:scored], entropies[:scored] + np.array(follow_ups)


def endgame_guess(state: CandidateState) -> int | None:
    """The most probable candidate if few remain or one dominates; otherwise None."""
    (best_index, best_probability), = state.most_likely(1)
    if state.size <= ENDGAME_MAX_CANDIDATES or best_probability >= ENDGAME_MIN_PROBABILITY:
        return best_index
    return None


def best_lookahead_guess(state: CandidateState, time_budget: float | None = None) -> int:
    """The shortlisted guess with the highest 2-step lookahead score."""
    shortlist, scores = lookahead_scores(state.pattern_matrix, state.indices, state.weights, time_budget)
    return int(shortlist[np.argmax(scores)])


def choose_guess(state: CandidateState, time_budget: float | None = None) -> int:
    """Pick the next guess: the endgame rule if it applies, otherwise the best 2-step lookahead score.

    Without a time budget the full lookahead always runs; the web server passes one to stay responsive.
    """
    winning_attempt = endgame_guess(state)
    if winning_attempt is not None:
        return winning_attempt
    return best_lookahead_guess(state, time_budget)
