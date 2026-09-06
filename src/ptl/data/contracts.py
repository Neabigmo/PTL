"""Typed contracts for the four PTL-v2 data surfaces."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PredictionRecord:
    prediction_id: str
    biological_instance_id: str
    environment_id: str
    predictor: str
    predictor_version: str
    predictor_commit: str
    predictor_split_id: str
    seed: str
    prediction_array_path: str
    gene_list_path: str
    native_uq_fields: tuple[str, ...] = field(default_factory=tuple)
    ensemble_members: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        required = {
            "prediction_id": self.prediction_id,
            "biological_instance_id": self.biological_instance_id,
            "environment_id": self.environment_id,
            "predictor": self.predictor,
            "predictor_version": self.predictor_version,
            "predictor_split_id": self.predictor_split_id,
            "prediction_array_path": self.prediction_array_path,
            "gene_list_path": self.gene_list_path,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"missing prediction contract fields: {missing}")


@dataclass(frozen=True)
class DeploymentFeatureRecord:
    prediction_id: str
    blocks: dict[str, tuple[str, ...]]

    def validate(self) -> None:
        if not self.prediction_id:
            raise ValueError("deployment features require prediction_id")
        if set(self.blocks) - {"U", "P", "S", "N", "C"}:
            raise ValueError("deployment feature blocks must be U/P/S/N/C")


@dataclass(frozen=True)
class OutcomeRecord:
    prediction_id: str
    fidelity: float
    continuous_risk: float
    transportability: int | None = None
    deg: float | None = None
    retrieval: float | None = None


@dataclass(frozen=True)
class FoldRecord:
    biological_instance_id: str
    fold: int
    holdout_role: str = "cross_fit"


OUTCOME_TOKENS = (
    "fidelity",
    "continuous_risk",
    "transportability",
    "y_true",
    "outcome",
    "deg",
    "retrieval",
)
FORBIDDEN_BENCHMARK_TOKENS = ("split_family", "stress_family_flag", "heldout_target")


def validate_feature_columns(columns: list[str] | tuple[str, ...]) -> None:
    """Reject outcome-derived and answer-key benchmark columns."""

    violations = []
    for column in columns:
        normalized = column.casefold()
        if any(token in normalized for token in OUTCOME_TOKENS + FORBIDDEN_BENCHMARK_TOKENS):
            violations.append(column)
    if violations:
        raise ValueError(f"outcome/benchmark columns cannot enter deployment features: {violations}")
