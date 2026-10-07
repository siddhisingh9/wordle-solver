"""Prior probability of each word being the answer, derived from English word frequency."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from wordfreq import zipf_frequency

from wordle.pattern_matrix import CACHE_DIR, fingerprint, load_cached_array, save_cached_array

PRIORS_FILE = "priors.npy"
LANGUAGE = "en"
SIGMOID_CENTER = 2.5
SIGMOID_WIDTH = 0.4
WEIGHT_FLOOR = 1e-4
PLURAL_PENALTY = 0.1
PLURAL_STEM_MIN_ZIPF = 2.0


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic function 1 / (1 + e^-x), written with tanh so large inputs cannot overflow."""
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def zipf_frequencies(words: list[str]) -> np.ndarray:
    """Zipf frequency of each word in English: log10 of occurrences per billion words, 0 if unseen."""
    return np.array([zipf_frequency(word, LANGUAGE) for word in words], dtype=np.float64)


def frequency_weights(frequencies: np.ndarray) -> np.ndarray:
    """Map Zipf frequencies to weights in (0, 1) with a sigmoid, floored so no weight is zero."""
    weights = sigmoid((frequencies - SIGMOID_CENTER) / SIGMOID_WIDTH)
    return np.maximum(weights, WEIGHT_FLOOR)


def plural_stems(word: str) -> list[str]:
    """Return the stems left after removing a final S or ES."""
    stems = [word[:-1]]
    if word.endswith("es"):
        stems.append(word[:-2])
    return stems


def is_english_word(word: str) -> bool:
    """True if the word is common enough in English to count as a dictionary word."""
    return zipf_frequency(word, LANGUAGE) >= PLURAL_STEM_MIN_ZIPF


def is_simple_plural(word: str) -> bool:
    """True if the word ends in S (not SS) and removing the S or ES leaves an English word."""
    if not word.endswith("s") or word.endswith("ss"):
        return False
    return any(is_english_word(stem) for stem in plural_stems(word))


def plural_multipliers(words: list[str]) -> np.ndarray:
    """Weight multiplier per word: PLURAL_PENALTY for simple S/ES plurals, 1 otherwise."""
    return np.array([PLURAL_PENALTY if is_simple_plural(word) else 1.0 for word in words])


def compute_priors(words: list[str]) -> np.ndarray:
    """Normalised prior probability that each word is the answer."""
    weights = frequency_weights(zipf_frequencies(words)) * plural_multipliers(words)
    return weights / weights.sum()


def priors_fingerprint(words: list[str]) -> str:
    """Fingerprint of the word list and every prior parameter, so changing either rebuilds the cache."""
    return fingerprint(
        words, SIGMOID_CENTER, SIGMOID_WIDTH, WEIGHT_FLOOR, PLURAL_PENALTY, PLURAL_STEM_MIN_ZIPF
    )


def load_priors(words: list[str], cache_dir: Path = CACHE_DIR) -> np.ndarray:
    """Return the cached priors for this word list, computing them if needed."""
    path = cache_dir / PRIORS_FILE
    expected = priors_fingerprint(words)
    cached = load_cached_array(path, expected)
    if cached is not None:
        return cached
    priors = compute_priors(words)
    save_cached_array(path, priors, expected)
    return priors
