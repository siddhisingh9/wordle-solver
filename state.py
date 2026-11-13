"""The weighted set of answers still consistent with all feedback seen so far."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class InconsistentFeedbackError(ValueError):
    """Raised when no candidate word is consistent with the observed feedback."""


@dataclass(frozen=True)
class CandidateState:
    """Indices of the remaining candidate words, their prior weights, and the pattern matrix used to filter them."""

    pattern_matrix: np.ndarray
    indices: np.ndarray
    weights: np.ndarray

    @classmethod
    def initial(cls, pattern_matrix: np.ndarray, priors: np.ndarray) -> CandidateState:
        """State before any guess: every word is a candidate with its prior weight."""
        return cls(
            pattern_matrix=pattern_matrix,
            indices=np.arange(len(priors)),
            weights=np.asarray(priors, dtype=np.float64),
        )

    @property
    def size(self) -> int:
        """Number of remaining candidates."""
        return len(self.indices)

    @property
    def probabilities(self) -> np.ndarray:
        """Weights normalised to sum to 1."""
        return self.weights / self.weights.sum()

    def filter(self, guess_idx: int, observed_pattern: int) -> CandidateState:
        """Keep only the candidates that would have produced the observed pattern for this guess."""
        keep = self.pattern_matrix[guess_idx, self.indices] == observed_pattern
        if not keep.any():
            raise InconsistentFeedbackError(
                "No word matches all feedback so far; the colours were probably entered incorrectly."
            )
        return CandidateState(
            pattern_matrix=self.pattern_matrix,
            indices=self.indices[keep],
            weights=self.weights[keep],
        )

    def most_likely(self, count: int) -> list[tuple[int, float]]:
        """The `count` most probable candidates as (word index, probability), most probable first."""
        order = np.argsort(-self.weights, kind="stable")[:count]
        probabilities = self.probabilities
        return [(int(self.indices[i]), float(probabilities[i])) for i in order]
