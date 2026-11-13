"""Deployable web interface: a stateless JSON API around the solver plus a single-page front end.

The browser owns the game. Every request carries the full list of guesses and colours so far,
and the server replays it against the pattern matrix, so any number of visitors can use one
server process and no session storage is needed.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import AsyncIterator, Callable

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from wordle.feedback import ALL_GREEN, WORD_LENGTH, pattern_to_string, string_to_pattern
from wordle.solver import MAX_GUESSES, Game, Solver
from wordle.state import InconsistentFeedbackError

PAGE_PATH = Path(__file__).resolve().parent / "static" / "index.html"
MAX_HISTORY = 12
TOP_CANDIDATES_SHOWN = 10
SUGGESTION_CACHE_SIZE = 20000
THINKING_SECONDS = 6.0
DEFAULT_PORT = 7860

History = tuple[tuple[int, int], ...]


class Row(BaseModel):
    """One played guess and the colours the game showed for it."""

    word: str = Field(min_length=WORD_LENGTH, max_length=WORD_LENGTH)
    pattern: str = Field(min_length=WORD_LENGTH, max_length=WORD_LENGTH)


class StateRequest(BaseModel):
    """The game so far, oldest guess first."""

    history: list[Row] = Field(default_factory=list, max_length=MAX_HISTORY)


class SimulateRequest(BaseModel):
    """A known answer for the solver to play against."""

    answer: str = Field(min_length=WORD_LENGTH, max_length=WORD_LENGTH)


class Candidate(BaseModel):
    word: str
    probability: float


class StateResponse(BaseModel):
    history: list[Row]
    solved: bool
    suggestion: str | None
    suggestion_seconds: float
    remaining: int
    top: list[Candidate]
    max_guesses: int


class SimulateResponse(BaseModel):
    answer: str
    rows: list[Row]
    seconds: float
    max_guesses: int


class SolverService:
    """Turns request histories into solver positions and remembers suggestions already computed."""

    def __init__(self, solver: Solver) -> None:
        self.solver = solver
        self.cached_suggestion = lru_cache(maxsize=SUGGESTION_CACHE_SIZE)(self.compute_suggestion)

    def parse_history(self, rows: list[Row]) -> History:
        """Convert words and G/Y/B strings to indices and pattern codes, rejecting anything invalid."""
        parsed = []
        for number, row in enumerate(rows, start=1):
            try:
                parsed.append((self.solver.index_of(row.word), string_to_pattern(row.pattern)))
            except ValueError as error:
                raise HTTPException(status_code=400, detail=f"Guess {number}: {error}") from error
        solved_at = [i for i, (_, pattern) in enumerate(parsed) if pattern == ALL_GREEN]
        if solved_at and solved_at[0] != len(parsed) - 1:
            raise HTTPException(status_code=400, detail="No guesses can follow an all-green row.")
        return tuple(parsed)

    def replay(self, history: History) -> Game:
        """Apply every guess in order, naming the first row whose colours contradict the earlier ones."""
        game = self.solver.new_game()
        for number, (guess, pattern) in enumerate(history, start=1):
            try:
                game.record(guess, pattern)
            except InconsistentFeedbackError as error:
                word = self.solver.words[guess].upper()
                raise HTTPException(
                    status_code=400,
                    detail=f"Guess {number} ({word}): no word fits all the colours so far. Check the colours.",
                ) from error
        return game

    def compute_suggestion(self, history: History) -> int:
        """The solver's next guess for this exact history, within the web thinking-time budget."""
        return self.replay(history).suggest(THINKING_SECONDS)

    def state(self, rows: list[Row]) -> StateResponse:
        """Candidates, probabilities and the next suggestion after the given guesses."""
        history = self.parse_history(rows)
        game = self.replay(history)
        solved = bool(history) and history[-1][1] == ALL_GREEN
        suggestion: str | None = None
        start = time.perf_counter()
        if not solved:
            suggestion = self.solver.words[self.cached_suggestion(history)]
        words = self.solver.words
        return StateResponse(
            history=[Row(word=words[g], pattern=pattern_to_string(p)) for g, p in history],
            solved=solved,
            suggestion=suggestion,
            suggestion_seconds=round(time.perf_counter() - start, 3),
            remaining=game.state.size,
            top=[
                Candidate(word=words[index], probability=probability)
                for index, probability in game.state.most_likely(TOP_CANDIDATES_SHOWN)
            ],
            max_guesses=MAX_GUESSES,
        )

    def simulate(self, answer: str) -> SimulateResponse:
        """Let the solver play a known answer and report each guess with its colours."""
        try:
            answer_index = self.solver.index_of(answer)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        start = time.perf_counter()
        guesses = self.solver.simulate(answer)
        seconds = time.perf_counter() - start
        matrix = self.solver.pattern_matrix
        rows = [
            Row(word=word, pattern=pattern_to_string(int(matrix[self.solver.index_of(word), answer_index])))
            for word in guesses
        ]
        return SimulateResponse(
            answer=self.solver.words[answer_index], rows=rows, seconds=round(seconds, 3), max_guesses=MAX_GUESSES
        )


def load_default_solver() -> Solver:
    """Load the solver with the pattern matrix memory-mapped from the cache."""
    return Solver.load(memory_map=True)


def create_app(load_solver: Callable[[], Solver] = load_default_solver) -> FastAPI:
    """Build the web application; the solver is loaded once when the server starts."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.service = SolverService(load_solver())
        yield

    app = FastAPI(title="Wordle Solver", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    def service(request: Request) -> SolverService:
        return request.app.state.service

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(PAGE_PATH, media_type="text/html")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/state", response_model=StateResponse)
    def state(body: StateRequest, request: Request) -> StateResponse:
        return service(request).state(body.history)

    @app.post("/api/simulate", response_model=SimulateResponse)
    def simulate(body: SimulateRequest, request: Request) -> SimulateResponse:
        return service(request).simulate(body.answer)

    return app


app = create_app()


def serve(host: str, port: int) -> None:
    """Run the web app with uvicorn."""
    uvicorn.run(app, host=host, port=port)
