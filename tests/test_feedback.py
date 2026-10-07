import random
import string

import pytest

from wordle.feedback import (
    ALL_GREEN,
    NUM_PATTERNS,
    decode_pattern,
    encode_colours,
    feedback,
    pattern_to_string,
    string_to_pattern,
)

RANDOM_SEED = 2022
PROPERTY_TEST_SAMPLES = 5000


def random_word(rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_lowercase, k=5))


@pytest.mark.parametrize(
    ("guess", "answer", "expected"),
    [
        ("babes", "abbey", "YYGGB"),
        ("speed", "abide", "BBYBY"),
        ("crane", "crane", "GGGGG"),
        ("zzzzz", "abcde", "BBBBB"),
        ("eerie", "speed", "YYBBB"),
        ("llama", "hello", "YYBBB"),
        ("lolly", "hello", "BYGGB"),
        ("abbey", "babes", "YYGGB"),
    ],
)
def test_feedback_matches_official_duplicate_letter_rule(guess: str, answer: str, expected: str) -> None:
    assert pattern_to_string(feedback(guess, answer)) == expected


def test_identical_guess_and_answer_is_all_green() -> None:
    rng = random.Random(RANDOM_SEED)
    for _ in range(PROPERTY_TEST_SAMPLES):
        word = random_word(rng)
        assert feedback(word, word) == ALL_GREEN == 242


def test_coloured_letter_count_never_exceeds_answer_letter_count() -> None:
    rng = random.Random(RANDOM_SEED)
    for _ in range(PROPERTY_TEST_SAMPLES):
        guess, answer = random_word(rng), random_word(rng)
        colours = pattern_to_string(feedback(guess, answer))
        for letter in set(guess):
            coloured = sum(1 for g, c in zip(guess, colours) if g == letter and c != "B")
            assert coloured == min(guess.count(letter), answer.count(letter))


def test_feedback_rejects_wrong_length_words() -> None:
    with pytest.raises(ValueError):
        feedback("four", "crane")


def test_encoding_uses_base_three_with_position_zero_least_significant() -> None:
    assert encode_colours([0, 0, 0, 0, 0]) == 0
    assert encode_colours([2, 0, 0, 0, 0]) == 2
    assert encode_colours([0, 1, 0, 0, 0]) == 3
    assert encode_colours([0, 0, 0, 0, 2]) == 162
    assert string_to_pattern("GGGGG") == 242


def test_pattern_round_trips_through_colours_and_strings() -> None:
    for pattern in range(NUM_PATTERNS):
        assert encode_colours(decode_pattern(pattern)) == pattern
        assert string_to_pattern(pattern_to_string(pattern)) == pattern


def test_string_to_pattern_is_case_insensitive() -> None:
    assert string_to_pattern("gybbg") == string_to_pattern("GYBBG")


@pytest.mark.parametrize("bad_text", ["GYBB", "GYBBGG", "GYXBG", ""])
def test_string_to_pattern_rejects_malformed_input(bad_text: str) -> None:
    with pytest.raises(ValueError):
        string_to_pattern(bad_text)


@pytest.mark.parametrize("bad_pattern", [-1, 243])
def test_decode_pattern_rejects_out_of_range(bad_pattern: int) -> None:
    with pytest.raises(ValueError):
        decode_pattern(bad_pattern)
