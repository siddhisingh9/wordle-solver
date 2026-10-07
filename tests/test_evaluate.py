import csv
from pathlib import Path

import pytest

from wordle.evaluate import GameResult, format_summary, plot_histogram, summarise, write_csv

RESULTS = [
    GameResult("crane", ["tarse", "crane"], 0.1),
    GameResult("abbey", ["tarse", "lingo", "abbey"], 0.3),
    GameResult("jazzy", ["tarse", "a", "b", "c", "d", "e", "jazzy"], 0.5),
    GameResult("mocha", ["tarse", "lingo", "mocha"], 0.3),
]


def test_summary_metrics() -> None:
    summary = summarise("sample.txt", RESULTS)
    assert summary.games == 4
    assert summary.mean_guesses == pytest.approx(15 / 4)
    assert summary.win_rate == pytest.approx(0.75)
    assert summary.distribution == {2: 1, 3: 2, 7: 1}
    assert summary.worst_case == 7
    assert summary.worst_words == ["jazzy"]
    assert summary.failed_words == ["jazzy"]
    assert summary.mean_seconds == pytest.approx(0.3)


def test_csv_has_one_row_per_game(tmp_path: Path) -> None:
    path = tmp_path / "sample.csv"
    write_csv(path, RESULTS)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert [r["answer"] for r in rows] == ["crane", "abbey", "jazzy", "mocha"]
    assert rows[2]["won"] == "0"
    assert rows[1]["guesses"] == "tarse lingo abbey"


def test_histogram_is_written(tmp_path: Path) -> None:
    path = tmp_path / "hist.png"
    plot_histogram(path, summarise("sample.txt", RESULTS))
    assert path.stat().st_size > 0


def test_formatted_summary_lists_failures() -> None:
    text = format_summary(summarise("sample.txt", RESULTS))
    assert "75.00%" in text
    assert "jazzy" in text
