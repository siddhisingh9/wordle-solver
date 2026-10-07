import numpy as np
import pytest
from fastapi.testclient import TestClient

from wordle.feedback import feedback, pattern_to_string
from wordle.lookahead import choose_guess
from wordle.pattern_matrix import compute_pattern_matrix
from wordle.solver import OpeningMoves, Solver
from wordle.state import CandidateState
from wordle.webapp import create_app

WORDS = [
    "crane", "crate", "trace", "slate", "stale", "steal", "least", "abbey", "babes", "speed",
    "abide", "hello", "lolly", "mount", "pound", "round", "sound", "wound", "found", "bound",
]


def small_solver() -> Solver:
    matrix = compute_pattern_matrix(WORDS, WORDS)
    priors = np.linspace(1.0, 2.0, len(WORDS))
    priors = priors / priors.sum()
    first = choose_guess(CandidateState.initial(matrix, priors))
    return Solver(words=WORDS, pattern_matrix=matrix, priors=priors, opening_moves=OpeningMoves(first, {}))


@pytest.fixture
def client() -> TestClient:
    with TestClient(create_app(small_solver)) as test_client:
        yield test_client


def row(guess: str, answer: str) -> dict[str, str]:
    return {"word": guess, "pattern": pattern_to_string(feedback(guess, answer))}


def test_page_and_health_are_served(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"status": "ok"}
    page = client.get("/")
    assert page.status_code == 200
    assert "Wordle Solver" in page.text


def test_empty_history_returns_opening_suggestion(client: TestClient) -> None:
    body = client.post("/api/state", json={"history": []}).json()
    assert body["remaining"] == len(WORDS)
    assert body["suggestion"] in WORDS
    assert body["solved"] is False
    assert len(body["top"]) == 10
    assert sum(c["probability"] for c in body["top"]) <= 1.0


def test_history_narrows_candidates(client: TestClient) -> None:
    body = client.post("/api/state", json={"history": [row("crane", "sound")]}).json()
    expected = sum(1 for w in WORDS if feedback("crane", w) == feedback("crane", "sound"))
    assert body["remaining"] == expected
    assert body["history"] == [row("crane", "sound")]
    assert "sound" in [c["word"] for c in body["top"]]


def test_all_green_marks_solved_without_suggestion(client: TestClient) -> None:
    body = client.post("/api/state", json={"history": [row("crane", "sound"), row("sound", "sound")]}).json()
    assert body["solved"] is True
    assert body["suggestion"] is None
    assert body["remaining"] == 1


def test_requests_are_independent(client: TestClient) -> None:
    client.post("/api/state", json={"history": [row("crane", "sound")]})
    body = client.post("/api/state", json={"history": []}).json()
    assert body["remaining"] == len(WORDS)


def test_suggestions_are_cached_by_history(client: TestClient) -> None:
    history = {"history": [row("crane", "sound")]}
    first = client.post("/api/state", json=history).json()
    second = client.post("/api/state", json=history).json()
    assert first["suggestion"] == second["suggestion"]
    service = client.app.state.service
    assert service.cached_suggestion.cache_info().hits >= 1


@pytest.mark.parametrize(
    ("history", "message"),
    [
        ([{"word": "qqqqq", "pattern": "BBBBB"}], "not in the allowed-guess dictionary"),
        ([{"word": "crane", "pattern": "BBXBB"}], "G, Y or B"),
        ([{"word": "crane", "pattern": "GGGGY"}], "Check the colours"),
        ([{"word": "crane", "pattern": "GGGGG"}, {"word": "crate", "pattern": "BBBBB"}], "all-green"),
    ],
)
def test_invalid_histories_are_rejected_with_a_message(client: TestClient, history: list, message: str) -> None:
    response = client.post("/api/state", json={"history": history})
    assert response.status_code == 400
    assert message in response.json()["detail"]


def test_wrong_length_rows_fail_validation(client: TestClient) -> None:
    response = client.post("/api/state", json={"history": [{"word": "four", "pattern": "BBBB"}]})
    assert response.status_code == 422


def test_simulate_returns_each_guess_with_colours(client: TestClient) -> None:
    body = client.post("/api/simulate", json={"answer": "wound"}).json()
    assert body["answer"] == "wound"
    assert body["rows"][-1] == {"word": "wound", "pattern": "GGGGG"}
    for item in body["rows"]:
        assert item["pattern"] == pattern_to_string(feedback(item["word"], "wound"))


def test_simulate_rejects_unknown_words(client: TestClient) -> None:
    assert client.post("/api/simulate", json={"answer": "zzzzz"}).status_code == 400
