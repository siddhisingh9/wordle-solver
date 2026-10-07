import time

import numpy as np
import pytest

from wordle.feedback import ALL_GREEN, feedback, string_to_pattern
from wordle.lookahead import choose_guess
from wordle.pattern_matrix import compute_pattern_matrix
from wordle.solver import OpeningMoves, Solver
from wordle.state import CandidateState, InconsistentFeedbackError

WORDS = [
    "crane", "crate", "trace", "slate", "stale", "steal", "least", "abbey", "babes", "speed",
    "abide", "hello", "lolly", "mount", "pound", "round", "sound", "wound", "found", "bound",
]
SAMPLE_ANSWERS = ["crane", "abbey", "mocha", "usury", "geese", "fungi", "jazzy", "nymph"]


@pytest.fixture
def small_solver() -> Solver:
    matrix = compute_pattern_matrix(WORDS, WORDS)
    priors = np.full(len(WORDS), 1.0 / len(WORDS))
    first = choose_guess(CandidateState.initial(matrix, priors))
    return Solver(words=WORDS, pattern_matrix=matrix, priors=priors, opening_moves=OpeningMoves(first, {}))


@pytest.fixture(scope="module")
def real_solver() -> Solver:
    return Solver.load()


def assert_valid_game(solver: Solver, answer: str, guesses: list[str]) -> None:
    assert guesses[-1] == answer
    assert len(set(guesses)) == len(guesses)
    assert all(feedback(g, answer) != ALL_GREEN for g in guesses[:-1])


@pytest.mark.parametrize("answer", WORDS)
def test_simulate_solves_every_word_in_a_small_dictionary(small_solver: Solver, answer: str) -> None:
    assert_valid_game(small_solver, answer, small_solver.simulate(answer))


def test_simulate_rejects_words_outside_the_dictionary(small_solver: Solver) -> None:
    with pytest.raises(ValueError):
        small_solver.simulate("zzzzz")


def test_game_record_and_undo(small_solver: Solver) -> None:
    game = small_solver.new_game()
    guess = game.suggest()
    game.record(guess, feedback(WORDS[guess], "sound"))
    narrowed = game.state.size
    assert narrowed < len(WORDS)
    assert game.undo()
    assert game.state.size == len(WORDS)
    assert not game.undo()


def test_game_rejects_impossible_feedback(small_solver: Solver) -> None:
    game = small_solver.new_game()
    with pytest.raises(InconsistentFeedbackError):
        game.record(WORDS.index("crane"), string_to_pattern("GGGGY"))
    assert game.history == []


@pytest.mark.parametrize("answer", SAMPLE_ANSWERS)
def test_real_solver_solves_sample_answers(real_solver: Solver, answer: str) -> None:
    guesses = real_solver.simulate(answer)
    assert_valid_game(real_solver, answer, guesses)


def test_first_two_moves_come_from_the_opening_cache(real_solver: Solver) -> None:
    game = real_solver.new_game()
    start = time.perf_counter()
    first = game.suggest()
    game.record(first, int(real_solver.pattern_matrix[first, real_solver.index_of("mocha")]))
    second = game.suggest()
    elapsed = time.perf_counter() - start
    assert first == real_solver.opening_moves.first_guess
    assert second == real_solver.opening_moves.second_guesses[game.history[0][1]]
    assert elapsed < 0.5


def test_opening_cache_covers_every_reachable_pattern(real_solver: Solver) -> None:
    first = real_solver.opening_moves.first_guess
    reachable = set(np.unique(real_solver.pattern_matrix[first]).tolist()) - {ALL_GREEN}
    assert set(real_solver.opening_moves.second_guesses) == reachable
