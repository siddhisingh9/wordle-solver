"""Command-line interface: play, simulate, evaluate, precompute."""

from __future__ import annotations

import argparse
import logging
import time
from typing import Callable

from wordle.evaluate import run_evaluation
from wordle.feedback import ALL_GREEN, pattern_to_string, string_to_pattern
from wordle.solver import MAX_GUESSES, Game, Solver
from wordle.webapp import DEFAULT_PORT, serve
from wordle.state import InconsistentFeedbackError

TOP_CANDIDATES_SHOWN = 10
UNDO = "undo"
QUIT = "quit"


class UndoRequested(Exception):
    """The player asked to take back the last entry."""


class QuitRequested(Exception):
    """The player asked to stop."""


def ask(prompt: str, read: Callable[[str], str]) -> str:
    """Read one line, turning the undo and quit keywords into exceptions."""
    answer = read(prompt).strip()
    if answer.lower() == UNDO:
        raise UndoRequested
    if answer.lower() == QUIT:
        raise QuitRequested
    return answer


def ask_guess(game: Game, suggestion: int, read: Callable[[str], str], write: Callable[[str], None]) -> int:
    """Ask which word was actually played; an empty answer accepts the suggestion."""
    words = game.solver.words
    while True:
        entered = ask(f"Word you played [Enter = {words[suggestion].upper()}]: ", read)
        if not entered:
            return suggestion
        try:
            return game.solver.index_of(entered)
        except ValueError as error:
            write(str(error))


def ask_pattern(word: str, read: Callable[[str], str], write: Callable[[str], None]) -> int:
    """Ask for the colours shown for a word as G/Y/B."""
    while True:
        entered = ask(f"Colours for {word.upper()} (G=green, Y=yellow, B=gray): ", read)
        try:
            return string_to_pattern(entered)
        except ValueError as error:
            write(str(error))


def describe_state(game: Game, write: Callable[[str], None]) -> None:
    """Show the remaining candidate count and the most likely answers."""
    state = game.state
    write(f"{state.size} candidates remain. Most likely answers:")
    for word_index, probability in state.most_likely(TOP_CANDIDATES_SHOWN):
        write(f"  {game.solver.words[word_index].upper()}  {probability:7.2%}")


def play(solver: Solver, read: Callable[[str], str] = input, write: Callable[[str], None] = print) -> None:
    """Interactive helper for the real NYT game, with 'undo' and 'quit' at any prompt."""
    write("Type 'undo' at any prompt to take back the last entry, or 'quit' to stop.")
    game = solver.new_game()
    while True:
        try:
            suggestion = game.suggest()
            write(f"\nGuess {len(game.history) + 1}. Suggested: {solver.words[suggestion].upper()}")
            guess = ask_guess(game, suggestion, read, write)
            pattern = ask_pattern(solver.words[guess], read, write)
            if pattern == ALL_GREEN:
                write(f"Solved in {len(game.history) + 1} guesses.")
                return
            game.record(guess, pattern)
            describe_state(game, write)
        except UndoRequested:
            write("Undid the last entry." if game.undo() else "Nothing to undo.")
            if game.history:
                describe_state(game, write)
        except InconsistentFeedbackError as error:
            write(str(error))
        except QuitRequested:
            return


def simulate(solver: Solver, answer: str, write: Callable[[str], None] = print) -> None:
    """Play against a known answer and print each guess with its feedback."""
    guesses = solver.simulate(answer)
    answer_index = solver.index_of(answer)
    for turn, word in enumerate(guesses, start=1):
        pattern = int(solver.pattern_matrix[solver.index_of(word), answer_index])
        write(f"{turn}. {word.upper()}  {pattern_to_string(pattern)}")
    outcome = "solved" if len(guesses) <= MAX_GUESSES else "LOST (more than 6 guesses)"
    write(f"{outcome} in {len(guesses)} guesses")


def precompute() -> None:
    """Build every cache (pattern matrix, priors, opening moves) that is missing or out of date."""
    start = time.perf_counter()
    solver = Solver.load()
    print(f"Caches ready in {time.perf_counter() - start:.1f} s")
    print(f"Opening guess: {solver.words[solver.opening_moves.first_guess].upper()}")


def build_parser() -> argparse.ArgumentParser:
    """Argument parser with the four sub-commands."""
    parser = argparse.ArgumentParser(prog="wordle", description="Entropy-based Wordle solver")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("play", help="get guess suggestions while playing the real game")
    simulate_parser = commands.add_parser("simulate", help="watch the solver play against a known answer")
    simulate_parser.add_argument("--answer", required=True, help="the hidden 5-letter answer")
    commands.add_parser("evaluate", help="benchmark on both answer lists and write results/")
    commands.add_parser("precompute", help="build the pattern matrix, priors and opening-move caches")
    ui_parser = commands.add_parser("ui", help="start the web interface")
    ui_parser.add_argument("--host", default="127.0.0.1", help="address to listen on (0.0.0.0 to accept outside connections)")
    ui_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="port to listen on")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for `python -m wordle.cli`."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    if args.command == "play":
        play(Solver.load())
    elif args.command == "simulate":
        simulate(Solver.load(), args.answer)
    elif args.command == "evaluate":
        run_evaluation()
    elif args.command == "precompute":
        precompute()
    elif args.command == "ui":
        serve(args.host, args.port)


if __name__ == "__main__":
    main()
