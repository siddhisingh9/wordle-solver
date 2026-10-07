"""Wordle colour feedback and its base-3 integer encoding."""

from __future__ import annotations

from collections import Counter

from wordle.words import WORD_LENGTH

GRAY = 0
YELLOW = 1
GREEN = 2

NUM_PATTERNS = 3**WORD_LENGTH
ALL_GREEN = NUM_PATTERNS - 1

COLOUR_TO_LETTER = {GRAY: "B", YELLOW: "Y", GREEN: "G"}
LETTER_TO_COLOUR = {letter: colour for colour, letter in COLOUR_TO_LETTER.items()}


def encode_colours(colours: list[int]) -> int:
    """Pack per-position colours into an integer 0-242, position 0 as the least-significant digit."""
    return sum(colour * 3**position for position, colour in enumerate(colours))


def decode_pattern(pattern: int) -> list[int]:
    """Unpack an integer 0-242 into its per-position colours."""
    if not 0 <= pattern < NUM_PATTERNS:
        raise ValueError(f"Pattern must be in 0..{ALL_GREEN}, got {pattern}")
    return [(pattern // 3**position) % 3 for position in range(WORD_LENGTH)]


def pattern_to_string(pattern: int) -> str:
    """Render a pattern as a string of G (green), Y (yellow) and B (gray), e.g. 'GYBBG'."""
    return "".join(COLOUR_TO_LETTER[colour] for colour in decode_pattern(pattern))


def string_to_pattern(text: str) -> int:
    """Parse a G/Y/B string such as 'GYBBG' (case-insensitive) into a pattern integer."""
    letters = text.strip().upper()
    if len(letters) != WORD_LENGTH or any(letter not in LETTER_TO_COLOUR for letter in letters):
        raise ValueError(f"Feedback must be {WORD_LENGTH} characters of G, Y or B, got {text!r}")
    return encode_colours([LETTER_TO_COLOUR[letter] for letter in letters])


def feedback_colours(guess: str, answer: str) -> list[int]:
    """Colour each guess letter against the answer using the official duplicate-letter rule."""
    colours = [GRAY] * WORD_LENGTH
    unmatched_answer_letters: Counter[str] = Counter()
    for position, (guess_letter, answer_letter) in enumerate(zip(guess, answer)):
        if guess_letter == answer_letter:
            colours[position] = GREEN
        else:
            unmatched_answer_letters[answer_letter] += 1
    for position, guess_letter in enumerate(guess):
        if colours[position] != GREEN and unmatched_answer_letters[guess_letter] > 0:
            colours[position] = YELLOW
            unmatched_answer_letters[guess_letter] -= 1
    return colours


def feedback(guess: str, answer: str) -> int:
    """Return the encoded feedback pattern (0-242) produced by guessing `guess` when the answer is `answer`."""
    if len(guess) != WORD_LENGTH or len(answer) != WORD_LENGTH:
        raise ValueError(f"Guess and answer must both have {WORD_LENGTH} letters")
    return encode_colours(feedback_colours(guess, answer))
