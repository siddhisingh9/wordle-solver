"""Loading and validating the word lists used by the solver."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

WORD_LENGTH = 5
VALID_WORD = re.compile(r"^[a-z]{5}$")

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
ALLOWED_GUESSES_FILE = "allowed_guesses.txt"
ORIGINAL_ANSWERS_FILE = "original_answers.txt"
NYT_PAST_ANSWERS_FILE = "nyt_past_answers.txt"


@dataclass(frozen=True)
class WordFile:
    """The valid words of a word-list file plus every problem found while reading it."""

    words: list[str]
    invalid_lines: list[tuple[int, str]]
    duplicates: list[str]


@dataclass(frozen=True)
class WordLists:
    """The guess dictionary and the two evaluation answer lists."""

    allowed: list[str]
    original_answers: list[str]
    nyt_past_answers: list[str]


def is_valid_word(word: str) -> bool:
    """Return True if the word is exactly five lowercase letters a-z."""
    return VALID_WORD.fullmatch(word) is not None


def parse_word_lines(lines: list[str]) -> WordFile:
    """Normalise, validate and de-duplicate raw lines, keeping first-seen order."""
    words: list[str] = []
    seen: set[str] = set()
    invalid_lines: list[tuple[int, str]] = []
    duplicates: list[str] = []
    for line_number, raw_line in enumerate(lines, start=1):
        word = raw_line.strip().lower()
        if not word:
            continue
        if not is_valid_word(word):
            invalid_lines.append((line_number, raw_line.rstrip("\n")))
        elif word in seen:
            duplicates.append(word)
        else:
            seen.add(word)
            words.append(word)
    return WordFile(words=words, invalid_lines=invalid_lines, duplicates=duplicates)


def report_problems(path: Path, word_file: WordFile) -> None:
    """Log every invalid line and duplicate found in a word-list file."""
    for line_number, content in word_file.invalid_lines:
        logger.warning("%s line %d is not a 5-letter a-z word: %r", path.name, line_number, content)
    if word_file.duplicates:
        logger.warning(
            "%s contains %d duplicate words: %s",
            path.name,
            len(word_file.duplicates),
            ", ".join(word_file.duplicates),
        )


def load_words(path: Path) -> list[str]:
    """Read a one-word-per-line file, report bad lines and duplicates, and return the valid words."""
    if not path.exists():
        raise FileNotFoundError(f"Word list not found: {path}")
    word_file = parse_word_lines(path.read_text(encoding="utf-8").splitlines())
    report_problems(path, word_file)
    return word_file.words


def find_missing_words(words: list[str], dictionary: list[str]) -> list[str]:
    """Return the words that do not appear in the dictionary."""
    known = set(dictionary)
    return [word for word in words if word not in known]


def report_missing_words(list_name: str, missing: list[str]) -> None:
    """Log answer words that are absent from the allowed-guess dictionary."""
    if missing:
        logger.warning(
            "%s has %d words not in %s: %s",
            list_name,
            len(missing),
            ALLOWED_GUESSES_FILE,
            ", ".join(missing),
        )


def load_answer_words(path: Path, allowed: list[str]) -> list[str]:
    """Load an evaluation answer list and log any of its words missing from the dictionary."""
    answers = load_words(path)
    report_missing_words(path.name, find_missing_words(answers, allowed))
    return answers


def load_word_lists(data_dir: Path = DATA_DIR) -> WordLists:
    """Load and cross-check the allowed-guess dictionary and both answer lists."""
    allowed = load_words(data_dir / ALLOWED_GUESSES_FILE)
    return WordLists(
        allowed=allowed,
        original_answers=load_answer_words(data_dir / ORIGINAL_ANSWERS_FILE, allowed),
        nyt_past_answers=load_answer_words(data_dir / NYT_PAST_ANSWERS_FILE, allowed),
    )
