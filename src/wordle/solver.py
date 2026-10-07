"""Game loop: the solver, its precomputed opening moves, and a game session with undo."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from wordle.feedback import ALL_GREEN, NUM_PATTERNS, pattern_to_string, string_to_pattern
from wordle.lookahead import (
    ENDGAME_MAX_CANDIDATES,
    ENDGAME_MIN_PROBABILITY,
    FIRST_GUESS_SHORTLIST,
    MIN_GROUP_SIZE_FOR_FOLLOW_UP,
    choose_guess,
    expected_follow_up_entropy,
    top_guesses_by_entropy,
)
from wordle.pattern_matrix import CACHE_DIR, PATTERN_MATRIX_FILE, fingerprint, load_pattern_matrix
from wordle.priors import load_priors
from wordle.state import CandidateState
from wordle.words import ALLOWED_GUESSES_FILE, DATA_DIR, load_words

logger = logging.getLogger(__name__)

MAX_GUESSES = 6
OPENING_MOVES_FILE = "opening_moves.json"
MAX_WORKER_PROCESSES = 6

worker_pattern_matrix: np.ndarray | None = None
worker_priors: np.ndarray | None = None


@dataclass(frozen=True)
class OpeningMoves:
    """The best first guess and the best second guess for every feedback pattern the first guess can produce."""

    first_guess: int
    second_guesses: dict[int, int]


def opening_moves_fingerprint(words: list[str], priors: np.ndarray) -> str:
    """Fingerprint of everything that determines the opening moves."""
    return fingerprint(
        words,
        hashlib.sha256(priors.tobytes()).hexdigest(),
        FIRST_GUESS_SHORTLIST,
        MIN_GROUP_SIZE_FOR_FOLLOW_UP,
        ENDGAME_MAX_CANDIDATES,
        ENDGAME_MIN_PROBABILITY,
    )


def init_worker(pattern_matrix_path: Path, priors: np.ndarray) -> None:
    """Give a worker process read-only, memory-mapped access to the pattern matrix and the priors."""
    global worker_pattern_matrix, worker_priors
    worker_pattern_matrix = np.load(pattern_matrix_path, mmap_mode="r")
    worker_priors = priors


def worker_root_follow_up(first_guess: int) -> float:
    """Expected best follow-up entropy of a first guess, over the full candidate set."""
    candidates = np.arange(len(worker_priors))
    return expected_follow_up_entropy(worker_pattern_matrix, first_guess, candidates, worker_priors)


def worker_second_guess(first_guess: int, pattern: int) -> tuple[int, int]:
    """The guess the solver would choose after seeing `pattern` for the first guess."""
    state = CandidateState.initial(worker_pattern_matrix, worker_priors).filter(first_guess, pattern)
    return pattern, choose_guess(state)


def reachable_patterns(pattern_matrix: np.ndarray, first_guess: int) -> list[int]:
    """Patterns other than all-green that the first guess can produce, largest candidate group first."""
    sizes = np.bincount(pattern_matrix[first_guess], minlength=NUM_PATTERNS)
    patterns = [p for p in np.flatnonzero(sizes) if p != ALL_GREEN]
    return sorted(patterns, key=lambda p: -sizes[p])


def compute_opening_moves(pattern_matrix: np.ndarray, priors: np.ndarray, pattern_matrix_path: Path) -> OpeningMoves:
    """Run the 2-step lookahead for the first move and for every possible second move, in parallel."""
    candidates = np.arange(len(priors))
    shortlist, entropies = top_guesses_by_entropy(pattern_matrix, candidates, priors, FIRST_GUESS_SHORTLIST)
    processes = max(1, min(MAX_WORKER_PROCESSES, (os.cpu_count() or 2) - 1))
    with Pool(processes, initializer=init_worker, initargs=(pattern_matrix_path, priors)) as pool:
        logger.info("Scoring %d first-guess candidates with %d processes", len(shortlist), processes)
        follow_ups = np.array(pool.map(worker_root_follow_up, [int(g) for g in shortlist], chunksize=1))
        first_guess = int(shortlist[np.argmax(entropies + follow_ups)])
        patterns = reachable_patterns(pattern_matrix, first_guess)
        logger.info("Choosing second guesses for %d feedback patterns", len(patterns))
        second_guesses = dict(
            pool.starmap(worker_second_guess, [(first_guess, int(p)) for p in patterns], chunksize=1)
        )
    return OpeningMoves(first_guess=first_guess, second_guesses=second_guesses)


def save_opening_moves(path: Path, moves: OpeningMoves, words: list[str], moves_fingerprint: str) -> None:
    """Write opening moves as JSON, keyed by readable G/Y/B patterns."""
    payload = {
        "fingerprint": moves_fingerprint,
        "first_guess": words[moves.first_guess],
        "second_guesses": {
            pattern_to_string(pattern): words[guess] for pattern, guess in sorted(moves.second_guesses.items())
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def read_opening_moves(path: Path, word_index: dict[str, int], expected_fingerprint: str) -> OpeningMoves | None:
    """Read cached opening moves, or return None if missing or built from different inputs."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("fingerprint") != expected_fingerprint:
        logger.info("Inputs changed since %s was cached; recomputing", path.name)
        return None
    return OpeningMoves(
        first_guess=word_index[payload["first_guess"]],
        second_guesses={
            string_to_pattern(pattern): word_index[guess] for pattern, guess in payload["second_guesses"].items()
        },
    )


def load_opening_moves(
    words: list[str], pattern_matrix: np.ndarray, priors: np.ndarray, cache_dir: Path = CACHE_DIR
) -> OpeningMoves:
    """Return the cached opening moves, computing and caching them if needed."""
    path = cache_dir / OPENING_MOVES_FILE
    expected = opening_moves_fingerprint(words, priors)
    cached = read_opening_moves(path, {word: i for i, word in enumerate(words)}, expected)
    if cached is not None:
        return cached
    logger.info("Precomputing opening moves; this runs once and takes several minutes")
    moves = compute_opening_moves(pattern_matrix, priors, cache_dir / PATTERN_MATRIX_FILE)
    save_opening_moves(path, moves, words, expected)
    return moves


@dataclass
class Solver:
    """Everything needed to play: the dictionary, pattern matrix, priors and opening moves."""

    words: list[str]
    pattern_matrix: np.ndarray
    priors: np.ndarray
    opening_moves: OpeningMoves
    word_index: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.word_index = {word: i for i, word in enumerate(self.words)}

    @classmethod
    def load(cls, data_dir: Path = DATA_DIR, memory_map: bool = False) -> Solver:
        """Load the dictionary and all caches, building any cache that is missing or stale."""
        cache_dir = data_dir / "cache"
        words = load_words(data_dir / ALLOWED_GUESSES_FILE)
        pattern_matrix = load_pattern_matrix(words, cache_dir, memory_map)
        priors = load_priors(words, cache_dir)
        opening_moves = load_opening_moves(words, pattern_matrix, priors, cache_dir)
        return cls(words=words, pattern_matrix=pattern_matrix, priors=priors, opening_moves=opening_moves)

    def index_of(self, word: str) -> int:
        """Index of a word in the dictionary; raises ValueError for words that are not valid guesses."""
        normalised = word.strip().lower()
        if normalised not in self.word_index:
            raise ValueError(f"{word!r} is not in the allowed-guess dictionary")
        return self.word_index[normalised]

    def new_game(self) -> Game:
        """Start a game with every word as a candidate."""
        return Game(solver=self, states=[CandidateState.initial(self.pattern_matrix, self.priors)])

    def simulate(self, answer: str) -> list[str]:
        """Play a full game against a known answer and return every guess made, including the winning one."""
        answer_index = self.index_of(answer)
        game = self.new_game()
        guesses: list[str] = []
        while True:
            guess = game.suggest()
            guesses.append(self.words[guess])
            pattern = int(self.pattern_matrix[guess, answer_index])
            if pattern == ALL_GREEN:
                return guesses
            game.record(guess, pattern)


@dataclass
class Game:
    """One game in progress: the guesses and feedback so far, with the candidate state after each."""

    solver: Solver
    states: list[CandidateState]
    history: list[tuple[int, int]] = field(default_factory=list)

    @property
    def state(self) -> CandidateState:
        """Candidates consistent with all feedback so far."""
        return self.states[-1]

    def suggest(self, time_budget: float | None = None) -> int:
        """The solver's next guess, read from the opening-move cache for the first two moves when possible."""
        opening = self.solver.opening_moves
        if not self.history:
            return opening.first_guess
        if len(self.history) == 1:
            first_guess, first_pattern = self.history[0]
            if first_guess == opening.first_guess and first_pattern in opening.second_guesses:
                return opening.second_guesses[first_pattern]
        return choose_guess(self.state, time_budget)

    def record(self, guess: int, pattern: int) -> None:
        """Apply the feedback for a guess; raises InconsistentFeedbackError if no candidate fits."""
        self.states.append(self.state.filter(guess, pattern))
        self.history.append((guess, pattern))

    def undo(self) -> bool:
        """Remove the last recorded guess; returns False if there was nothing to undo."""
        if not self.history:
            return False
        self.history.pop()
        self.states.pop()
        return True
