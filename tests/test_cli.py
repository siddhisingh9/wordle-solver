import numpy as np
import pytest

from wordle.cli import build_parser, play, simulate
from wordle.feedback import feedback, pattern_to_string
from wordle.lookahead import choose_guess
from wordle.pattern_matrix import compute_pattern_matrix
from wordle.solver import OpeningMoves, Solver
from wordle.state import CandidateState

WORDS = [
    "crane", "crate", "trace", "slate", "stale", "steal", "least", "abbey", "babes", "speed",
    "abide", "hello", "lolly", "mount", "pound", "round", "sound", "wound", "found", "bound",
]


@pytest.fixture
def solver() -> Solver:
    matrix = compute_pattern_matrix(WORDS, WORDS)
    priors = np.full(len(WORDS), 1.0 / len(WORDS))
    first = choose_guess(CandidateState.initial(matrix, priors))
    return Solver(words=WORDS, pattern_matrix=matrix, priors=priors, opening_moves=OpeningMoves(first, {}))


def run_play(solver: Solver, inputs: list[str]) -> list[str]:
    remaining = iter(inputs)
    output: list[str] = []
    play(solver, read=lambda prompt: next(remaining), write=output.append)
    return output


def colours(guess: str, answer: str) -> str:
    return pattern_to_string(feedback(guess, answer))


def test_play_accepts_suggestion_and_reports_win(solver: Solver) -> None:
    first = WORDS[solver.opening_moves.first_guess]
    output = run_play(solver, ["", colours(first, first)])
    assert output[-1] == "Solved in 1 guesses."


def test_play_with_own_guess_shows_candidates_and_next_suggestion(solver: Solver) -> None:
    output = run_play(solver, ["crane", colours("crane", "sound"), "quit"])
    text = "\n".join(output)
    assert "candidates remain" in text
    assert "SOUND" in text
    assert text.count("Suggested:") == 2


def test_play_undo_restores_previous_state(solver: Solver) -> None:
    output = run_play(solver, ["crane", colours("crane", "sound"), "undo", "quit"])
    assert "Undid the last entry." in output
    assert output[-1].startswith("\nGuess 1.")


def test_play_rejects_bad_input_and_impossible_feedback(solver: Solver) -> None:
    output = run_play(solver, ["qqqqq", "crane", "GGXGG", "GGGGY", "quit"])
    text = "\n".join(output)
    assert "not in the allowed-guess dictionary" in text
    assert "G, Y or B" in text
    assert "colours were probably entered incorrectly" in text


def test_simulate_command_prints_each_guess(solver: Solver) -> None:
    output: list[str] = []
    simulate(solver, "wound", write=output.append)
    assert output[-2].endswith("GGGGG")
    assert output[-1].startswith("solved in")


def test_parser_accepts_the_four_commands() -> None:
    parser = build_parser()
    assert parser.parse_args(["simulate", "--answer", "crane"]).answer == "crane"
    for command in ["play", "evaluate", "precompute"]:
        assert parser.parse_args([command]).command == command
