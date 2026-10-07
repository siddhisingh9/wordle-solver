"""Precomputed table of feedback patterns for every (guess, candidate) pair."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from wordle.feedback import GREEN, WORD_LENGTH, YELLOW
from wordle.words import DATA_DIR

logger = logging.getLogger(__name__)

CACHE_DIR = DATA_DIR / "cache"
PATTERN_MATRIX_FILE = "pattern_matrix.npy"
GUESS_CHUNK_SIZE = 128
PLACE_VALUES = 3 ** np.arange(WORD_LENGTH, dtype=np.int32)
EARLIER_POSITION_MASK = np.tril(np.ones((WORD_LENGTH, WORD_LENGTH), dtype=bool), k=-1)


def fingerprint(*parts: object) -> str:
    """Return a SHA-256 digest identifying the given words and parameters."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(repr(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def fingerprint_path(array_path: Path) -> Path:
    """Return the sidecar file that stores the fingerprint of a cached array."""
    return array_path.with_suffix(".sha256")


def load_cached_array(array_path: Path, expected_fingerprint: str, memory_map: bool = False) -> np.ndarray | None:
    """Load a cached array if it exists and was built from the same inputs, otherwise return None."""
    sidecar = fingerprint_path(array_path)
    if not array_path.exists() or not sidecar.exists():
        return None
    if sidecar.read_text(encoding="utf-8").strip() != expected_fingerprint:
        logger.info("Inputs changed since %s was cached; recomputing", array_path.name)
        return None
    return np.load(array_path, mmap_mode="r" if memory_map else None)


def save_cached_array(array_path: Path, array: np.ndarray, array_fingerprint: str) -> None:
    """Write an array and its input fingerprint to the cache directory."""
    array_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(array_path, array)
    fingerprint_path(array_path).write_text(array_fingerprint, encoding="utf-8")


def words_to_letter_codes(words: list[str]) -> np.ndarray:
    """Convert words to an (n, 5) uint8 array of letter codes 0-25."""
    raw = np.frombuffer("".join(words).encode("ascii"), dtype=np.uint8)
    return (raw.reshape(len(words), WORD_LENGTH) - ord("a")).astype(np.uint8)


def patterns_for_chunk(guess_codes: np.ndarray, candidate_codes: np.ndarray) -> np.ndarray:
    """Vectorised feedback for every guess in a chunk against every candidate, as a (chunk, n) uint8 array."""
    green = guess_codes[:, None, :] == candidate_codes[None, :, :]
    letter_match = guess_codes[:, None, :, None] == candidate_codes[None, :, None, :]
    available = (letter_match & ~green[:, :, None, :]).sum(axis=-1)
    same_letter_earlier = (guess_codes[:, :, None] == guess_codes[:, None, :]) & EARLIER_POSITION_MASK
    used_earlier = (same_letter_earlier[:, None, :, :] & ~green[:, :, None, :]).sum(axis=-1)
    yellow = ~green & (used_earlier < available)
    colours = GREEN * green.astype(np.int32) + YELLOW * yellow.astype(np.int32)
    return (colours @ PLACE_VALUES).astype(np.uint8)


def compute_pattern_matrix(guesses: list[str], candidates: list[str]) -> np.ndarray:
    """Compute M[g, c] = feedback(guesses[g], candidates[c]) in chunks of guesses."""
    guess_codes = words_to_letter_codes(guesses)
    candidate_codes = words_to_letter_codes(candidates)
    matrix = np.empty((len(guesses), len(candidates)), dtype=np.uint8)
    for start in tqdm(range(0, len(guesses), GUESS_CHUNK_SIZE), desc="Pattern matrix", unit="chunk"):
        stop = start + GUESS_CHUNK_SIZE
        matrix[start:stop] = patterns_for_chunk(guess_codes[start:stop], candidate_codes)
    return matrix


def load_pattern_matrix(words: list[str], cache_dir: Path = CACHE_DIR, memory_map: bool = False) -> np.ndarray:
    """Return the guess x candidate pattern matrix, recomputing it only if the word list changed.

    With memory_map, a valid cache is opened read-only without copying it into memory, so several
    worker processes can share one copy.
    """
    path = cache_dir / PATTERN_MATRIX_FILE
    matrix_fingerprint = fingerprint(words)
    cached = load_cached_array(path, matrix_fingerprint, memory_map)
    if cached is not None:
        return cached
    matrix = compute_pattern_matrix(words, words)
    save_cached_array(path, matrix, matrix_fingerprint)
    return matrix
