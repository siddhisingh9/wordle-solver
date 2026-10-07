"""Expected information (weighted Shannon entropy) of guesses over a candidate set."""

from __future__ import annotations

import numpy as np

from wordle.feedback import NUM_PATTERNS

ELEMENTS_PER_CHUNK = 1 << 19


def pattern_group_weights(patterns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Total candidate weight falling into each of the 243 feedback patterns for one guess."""
    return np.bincount(patterns, weights=weights, minlength=NUM_PATTERNS)


def entropy_of_group_weights(group_weights: np.ndarray) -> np.ndarray:
    """Entropy in bits of the pattern distribution along the last axis: -sum p_k log2 p_k."""
    totals = group_weights.sum(axis=-1, keepdims=True)
    probabilities = group_weights / totals
    safe = np.where(probabilities > 0, probabilities, 1.0)
    return -(probabilities * np.log2(safe)).sum(axis=-1)


def group_weights_for_guesses(pattern_rows: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Pattern-group weights for several guesses at once using a single offset bincount."""
    rows, columns = pattern_rows.shape
    offsets = (np.arange(rows, dtype=np.int64) * NUM_PATTERNS)[:, None]
    keys = (pattern_rows.astype(np.int64) + offsets).ravel()
    tiled_weights = np.broadcast_to(weights, (rows, columns)).ravel()
    flat = np.bincount(keys, weights=tiled_weights, minlength=rows * NUM_PATTERNS)
    return flat.reshape(rows, NUM_PATTERNS)


def guess_entropies(
    pattern_matrix: np.ndarray,
    candidate_indices: np.ndarray,
    weights: np.ndarray,
    guess_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Entropy H(g) of every guess (all rows of the matrix by default) over the weighted candidates."""
    guess_rows = pattern_matrix if guess_indices is None else pattern_matrix[guess_indices]
    rows_per_chunk = max(1, ELEMENTS_PER_CHUNK // max(1, len(candidate_indices)))
    entropies = np.empty(guess_rows.shape[0], dtype=np.float64)
    for start in range(0, guess_rows.shape[0], rows_per_chunk):
        pattern_rows = np.take(guess_rows[start : start + rows_per_chunk], candidate_indices, axis=1)
        entropies[start : start + len(pattern_rows)] = entropy_of_group_weights(
            group_weights_for_guesses(pattern_rows, weights)
        )
    return entropies


def run_weights_of_sorted_rows(sorted_keys: np.ndarray, sorted_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For row-sorted keys, mark the last element of each run of equal keys and return (run_ends, run_weights)."""
    run_ends = np.ones(sorted_keys.shape, dtype=bool)
    run_ends[:, :-1] = sorted_keys[:, 1:] != sorted_keys[:, :-1]
    cumulative = np.cumsum(sorted_weights, axis=1)
    cumulative_at_ends = np.where(run_ends, cumulative, 0.0)
    before_run = np.zeros_like(cumulative)
    before_run[:, 1:] = np.maximum.accumulate(cumulative_at_ends, axis=1)[:, :-1]
    return run_ends, cumulative - before_run


def grouped_guess_entropies(
    pattern_matrix: np.ndarray,
    candidate_indices: np.ndarray,
    weights: np.ndarray,
    group_ids: np.ndarray,
    num_groups: int,
) -> np.ndarray:
    """Entropy of every guess over each group of candidates separately, as a (guesses, groups) array.

    Equivalent to calling guess_entropies once per group, but done in one pass: each guess row is
    sorted by (group, pattern), so every run of equal keys is one pattern bucket of one group, and
    H = log2(T) - (1/T) * sum_k W_k log2 W_k is accumulated per (guess, group) with a single bincount.
    """
    group_totals = np.bincount(group_ids, weights=weights, minlength=num_groups)
    group_offsets = group_ids.astype(np.int32) * NUM_PATTERNS
    num_guesses = pattern_matrix.shape[0]
    rows_per_chunk = max(1, ELEMENTS_PER_CHUNK // max(1, len(candidate_indices)))
    entropies = np.empty((num_guesses, num_groups), dtype=np.float64)
    for start in range(0, num_guesses, rows_per_chunk):
        keys = np.take(pattern_matrix[start : start + rows_per_chunk], candidate_indices, axis=1) + group_offsets
        order = np.argsort(keys, axis=1, kind="stable")
        sorted_keys = np.take_along_axis(keys, order, axis=1)
        run_ends, run_weights = run_weights_of_sorted_rows(sorted_keys, weights[order])
        rows, columns = np.nonzero(run_ends)
        bucket_weights = run_weights[rows, columns]
        groups = sorted_keys[rows, columns] // NUM_PATTERNS
        chunk_rows = keys.shape[0]
        weighted_logs = np.bincount(
            rows * num_groups + groups,
            weights=bucket_weights * np.log2(bucket_weights),
            minlength=chunk_rows * num_groups,
        ).reshape(chunk_rows, num_groups)
        with np.errstate(divide="ignore", invalid="ignore"):
            entropies[start : start + chunk_rows] = np.log2(group_totals) - weighted_logs / group_totals
    return entropies
