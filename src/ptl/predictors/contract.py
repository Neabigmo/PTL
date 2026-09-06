"""A validated, predictor-agnostic output contract."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PredictorOutputContract:
    prediction_id: str
    biological_instance_id: str
    environment_id: str
    predictor: str
    predictor_version: str
    predictor_commit: str
    predictor_split_id: str
    seed: int
    prediction_array_path: str
    gene_list_path: str
    native_uq: dict[str, float] = field(default_factory=dict)
    ensemble_member_seeds: tuple[int, ...] = field(default_factory=tuple)
    training_config_path: str = ""

    def validate(self) -> None:
        for name in (
            "prediction_id",
            "biological_instance_id",
            "environment_id",
            "predictor",
            "predictor_version",
            "predictor_split_id",
            "prediction_array_path",
            "gene_list_path",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"missing predictor output field: {name}")
