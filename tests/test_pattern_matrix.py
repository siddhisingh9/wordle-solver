import random
from pathlib import Path

import numpy as np

from wordle.feedback import feedback
from wordle.pattern_matrix import (
    PATTERN_MATRIX_FILE,
    compute_pattern_matrix,
    fingerprint_path,
    load_pattern_matrix,
)
from wordle.words import ALLOWED_GUESSES_FILE, DATA_DIR, load_words

RANDOM_SEED = 2022


def random_words(rng: random.Random, alphabet: str, count: int) -> list[str]:
    return ["".join(rng.choices(alphabet, k=5)) for _ in range(count)]


def assert_matrix_matches_feedback(guesses: list[str], candidates: list[str]) -> None:
    matrix = compute_pattern_matrix(guesses, candidates)
    expected = np.array([[feedback(g, c) for c in candidates] for g in guesses], dtype=np.uint8)
    np.testing.assert_array_equal(matrix, expected)


def test_matrix_matches_feedback_on_random_words() -> None:
    rng = random.Random(RANDOM_SEED)
    words = random_words(rng, "abcdefghijklmnopqrstuvwxyz", 70)
    assert_matrix_matches_feedback(words[:35], words[35:] + words[:35])


def test_matrix_matches_feedback_on_duplicate_heavy_words() -> None:
    rng = random.Random(RANDOM_SEED + 1)
    words = random_words(rng, "abe", 60)
    assert_matrix_matches_feedback(words, words)


def test_matrix_matches_feedback_on_random_dictionary_words() -> None:
    rng = random.Random(RANDOM_SEED + 2)
    allowed = load_words(DATA_DIR / ALLOWED_GUESSES_FILE)
    assert_matrix_matches_feedback(rng.sample(allowed, 50), rng.sample(allowed, 80))


def test_matrix_matches_feedback_on_known_tricky_pairs() -> None:
    guesses = ["babes", "speed", "lolly", "eerie", "llama"]
    candidates = ["abbey", "abide", "hello", "speed", "babes"]
    assert_matrix_matches_feedback(guesses, candidates)


def test_matrix_has_expected_shape_and_dtype() -> None:
    matrix = compute_pattern_matrix(["crane", "slate", "abbey"], ["crane", "slate"])
    assert matrix.shape == (3, 2)
    assert matrix.dtype == np.uint8


def test_matrix_cache_is_reused_and_rebuilt_when_words_change(tmp_path: Path) -> None:
    words = ["crane", "slate", "abbey"]
    first = load_pattern_matrix(words, tmp_path)
    cache_file = tmp_path / PATTERN_MATRIX_FILE
    assert cache_file.exists() and fingerprint_path(cache_file).exists()
    np.testing.assert_array_equal(load_pattern_matrix(words, tmp_path), first)
    changed = load_pattern_matrix(words + ["babes"], tmp_path)
    assert changed.shape == (4, 4)
