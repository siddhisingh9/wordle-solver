"""Benchmark the solver on every word of both answer lists and write CSVs, plots and a summary."""

from __future__ import annotations

import csv
import os
import time
from collections import Counter
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from wordle.solver import MAX_GUESSES, MAX_WORKER_PROCESSES, Solver
from wordle.words import DATA_DIR, NYT_PAST_ANSWERS_FILE, ORIGINAL_ANSWERS_FILE, load_word_lists

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
BAR_COLOUR = "#2a78d6"
SURFACE_COLOUR = "#fcfcfb"
PRIMARY_TEXT = "#0b0b0b"
SECONDARY_TEXT = "#52514e"
GRID_COLOUR = "#e4e3df"

worker_solver: Solver | None = None


@dataclass(frozen=True)
class GameResult:
    """Outcome of one simulated game."""

    answer: str
    guesses: list[str]
    seconds: float

    @property
    def num_guesses(self) -> int:
        return len(self.guesses)

    @property
    def won(self) -> bool:
        return self.num_guesses <= MAX_GUESSES


@dataclass(frozen=True)
class Summary:
    """Aggregate metrics over a set of games."""

    name: str
    games: int
    mean_guesses: float
    win_rate: float
    distribution: dict[int, int]
    worst_case: int
    worst_words: list[str]
    mean_seconds: float
    failed_words: list[str]


def init_worker(data_dir: Path) -> None:
    """Load a solver in each worker, sharing the pattern matrix through a memory map."""
    global worker_solver
    worker_solver = Solver.load(data_dir, memory_map=True)


def play_one(answer: str) -> GameResult:
    """Simulate one game in a worker and time it."""
    start = time.perf_counter()
    guesses = worker_solver.simulate(answer)
    return GameResult(answer=answer, guesses=guesses, seconds=time.perf_counter() - start)


def simulate_all(answers: list[str], data_dir: Path, label: str) -> list[GameResult]:
    """Simulate every answer in parallel, returning results in the original answer order."""
    processes = max(1, min(MAX_WORKER_PROCESSES, (os.cpu_count() or 2) - 1))
    with Pool(processes, initializer=init_worker, initargs=(data_dir,)) as pool:
        results = list(
            tqdm(pool.imap(play_one, answers, chunksize=4), total=len(answers), desc=label, unit="game")
        )
    return results


def summarise(name: str, results: list[GameResult]) -> Summary:
    """Compute mean guesses, win rate, distribution, worst case, runtime and failures."""
    counts = [r.num_guesses for r in results]
    worst = max(counts)
    return Summary(
        name=name,
        games=len(results),
        mean_guesses=sum(counts) / len(counts),
        win_rate=sum(r.won for r in results) / len(results),
        distribution=dict(sorted(Counter(counts).items())),
        worst_case=worst,
        worst_words=[r.answer for r in results if r.num_guesses == worst],
        mean_seconds=sum(r.seconds for r in results) / len(results),
        failed_words=[r.answer for r in results if not r.won],
    )


def write_csv(path: Path, results: list[GameResult]) -> None:
    """One row per game: answer, guess count, win flag, runtime and the guesses played."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["answer", "num_guesses", "won", "seconds", "guesses"])
        for r in results:
            writer.writerow([r.answer, r.num_guesses, int(r.won), f"{r.seconds:.4f}", " ".join(r.guesses)])


def style_axes(axes: plt.Axes) -> None:
    """Recessive grid and axes so the bars carry the message."""
    axes.set_facecolor(SURFACE_COLOUR)
    axes.grid(axis="y", color=GRID_COLOUR, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(SECONDARY_TEXT)
    axes.tick_params(colors=SECONDARY_TEXT, length=0)


def plot_histogram(path: Path, summary: Summary) -> None:
    """Bar chart of how many games took each number of guesses, with the 6-guess limit marked."""
    guess_counts = list(range(1, max(summary.worst_case, MAX_GUESSES) + 1))
    games = [summary.distribution.get(n, 0) for n in guess_counts]
    figure, axes = plt.subplots(figsize=(7, 4), dpi=150)
    figure.patch.set_facecolor(SURFACE_COLOUR)
    style_axes(axes)
    bars = axes.bar(guess_counts, games, width=0.6, color=BAR_COLOUR)
    axes.bar_label(bars, labels=[str(g) if g else "" for g in games], padding=3, color=PRIMARY_TEXT, fontsize=9)
    axes.axvline(MAX_GUESSES + 0.5, color=SECONDARY_TEXT, linestyle="--", linewidth=1)
    axes.text(
        MAX_GUESSES + 0.55, max(games) * 0.95, "6-guess limit", color=SECONDARY_TEXT, fontsize=9, va="top"
    )
    axes.set_xticks(guess_counts)
    axes.set_xlim(0.4, max(guess_counts[-1], MAX_GUESSES + 1) + 0.5)
    axes.set_xlabel("Guesses to solve", color=SECONDARY_TEXT)
    axes.set_ylabel("Games", color=SECONDARY_TEXT)
    axes.set_title(
        f"{summary.name}: {summary.games} answers, mean {summary.mean_guesses:.3f} guesses, "
        f"win rate {summary.win_rate:.2%}",
        color=PRIMARY_TEXT,
        fontsize=10,
        loc="left",
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor=SURFACE_COLOUR)
    plt.close(figure)


def format_summary(summary: Summary) -> str:
    """Human-readable summary of one answer list."""
    distribution = "  ".join(f"{n}:{count}" for n, count in summary.distribution.items())
    failed = ", ".join(summary.failed_words) if summary.failed_words else "none"
    return "\n".join(
        [
            f"== {summary.name} ({summary.games} answers) ==",
            f"Mean guesses:     {summary.mean_guesses:.4f}",
            f"Win rate (<= 6):  {summary.win_rate:.2%}",
            f"Distribution:     {distribution}",
            f"Worst case:       {summary.worst_case} guesses ({', '.join(summary.worst_words)})",
            f"Mean time/game:   {summary.mean_seconds:.3f} s",
            f"Failed words:     {failed}",
        ]
    )


def evaluate_answer_list(name: str, answers: list[str], data_dir: Path, results_dir: Path) -> Summary:
    """Simulate one answer list and write its CSV and histogram."""
    results = simulate_all(answers, data_dir, name)
    summary = summarise(name, results)
    stem = Path(name).stem
    write_csv(results_dir / f"{stem}.csv", results)
    plot_histogram(results_dir / f"{stem}_histogram.png", summary)
    return summary


def run_evaluation(data_dir: Path = DATA_DIR, results_dir: Path = RESULTS_DIR) -> list[Summary]:
    """Evaluate both answer lists, print their summaries and return them."""
    Solver.load(data_dir, memory_map=True)
    lists = load_word_lists(data_dir)
    summaries = []
    for name, answers in [
        (ORIGINAL_ANSWERS_FILE, lists.original_answers),
        (NYT_PAST_ANSWERS_FILE, lists.nyt_past_answers),
    ]:
        summary = evaluate_answer_list(name, answers, data_dir, results_dir)
        print(format_summary(summary), flush=True)
        summaries.append(summary)
    return summaries
