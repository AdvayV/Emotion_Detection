"""Evidence and counterfactual checks for faithful-model experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class EvidenceCheck:
    original_label: str
    masked_label: str
    evidence_changed_prediction: bool
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "original_label": self.original_label,
            "masked_label": self.masked_label,
            "evidence_changed_prediction": self.evidence_changed_prediction,
            "evidence": list(self.evidence),
        }


def check_evidence(text: str, evidence: tuple[str, ...], predict: Callable[[str], str]) -> EvidenceCheck:
    """Remove highlighted cues and verify whether the prediction responds."""
    original = predict(text)
    masked = text
    for cue in evidence:
        masked = masked.replace(cue, "<evidence>")
    masked_label = predict(masked)
    return EvidenceCheck(original, masked_label, original != masked_label, evidence)
