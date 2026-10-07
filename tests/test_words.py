import logging
from pathlib import Path

import pytest

from wordle.words import (
    find_missing_words,
    is_valid_word,
    load_answer_words,
    load_word_lists,
    load_words,
    parse_word_lines,
)


def write_lines(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("word", ["crane", "abbey", "zzzzz"])
def test_valid_words_are_accepted(word: str) -> None:
    assert is_valid_word(word)


@pytest.mark.parametrize("word", ["four", "sixsix", "cr4ne", "cr ne", "CRANE", "café!"])
def test_invalid_words_are_rejected(word: str) -> None:
    assert not is_valid_word(word)


def test_parse_normalises_case_and_whitespace_and_skips_blank_lines() -> None:
    result = parse_word_lines(["  Crane \n", "\n", "SLATE"])
    assert result.words == ["crane", "slate"]
    assert result.invalid_lines == []


def test_parse_reports_invalid_lines_with_line_numbers() -> None:
    result = parse_word_lines(["crane", "four", "cr4ne", "slate"])
    assert result.words == ["crane", "slate"]
    assert result.invalid_lines == [(2, "four"), (3, "cr4ne")]


def test_parse_removes_duplicates_keeping_first_occurrence() -> None:
    result = parse_word_lines(["crane", "slate", "Crane", "crane"])
    assert result.words == ["crane", "slate"]
    assert result.duplicates == ["crane", "crane"]


def test_load_words_logs_bad_lines(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = write_lines(tmp_path / "words.txt", ["crane", "toolong", "crane"])
    with caplog.at_level(logging.WARNING):
        assert load_words(path) == ["crane"]
    assert "toolong" in caplog.text
    assert "duplicate" in caplog.text


def test_load_words_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_words(tmp_path / "absent.txt")


def test_find_missing_words() -> None:
    assert find_missing_words(["crane", "qqqqq", "slate"], ["crane", "slate"]) == ["qqqqq"]


def test_load_answer_words_logs_words_absent_from_dictionary(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = write_lines(tmp_path / "answers.txt", ["crane", "qqqqq"])
    with caplog.at_level(logging.WARNING):
        assert load_answer_words(path, ["crane", "slate"]) == ["crane", "qqqqq"]
    assert "qqqqq" in caplog.text


def test_load_word_lists_reads_all_three_files(tmp_path: Path) -> None:
    write_lines(tmp_path / "allowed_guesses.txt", ["crane", "slate", "abbey"])
    write_lines(tmp_path / "original_answers.txt", ["crane"])
    write_lines(tmp_path / "nyt_past_answers.txt", ["abbey", "slate"])
    lists = load_word_lists(tmp_path)
    assert lists.allowed == ["crane", "slate", "abbey"]
    assert lists.original_answers == ["crane"]
    assert lists.nyt_past_answers == ["abbey", "slate"]
