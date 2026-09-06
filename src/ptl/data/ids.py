"""Stable PTL-v2 identities.

The identifiers describe biological and prediction identity separately from
run-directory names.  In particular, ``run_id`` is never used as a fold key.
"""

from __future__ import annotations

import re


def _part(value: object | None) -> str:
    text = "na" if value is None or str(value).strip() == "" else str(value).strip().casefold()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-") or "na"


def environment_id(
    dataset: object,
    cell_context: object,
    modality: object,
    condition: object | None = None,
) -> str:
    """Build a portable environment identifier from deployment descriptors."""

    values = [_part(dataset), _part(cell_context), _part(modality)]
    if condition is not None and str(condition).strip():
        values.append(_part(condition))
    return "env__" + "__".join(values)


def biological_instance_id(
    environment: object,
    perturbation: object,
    dose: object | None = None,
    timepoint: object | None = None,
) -> str:
    """Identify one ground-truth response shared by predictors and seeds."""

    values = [_part(environment), _part(perturbation)]
    if dose is not None and str(dose).strip():
        values.append(_part(dose))
    if timepoint is not None and str(timepoint).strip():
        values.append(_part(timepoint))
    return "bio__" + "__".join(values)


def prediction_id(
    biological_instance: object,
    predictor: object,
    predictor_split: object,
    model_seed: object,
) -> str:
    """Identify one predictor output without conflating it with the target."""

    return "pred__" + str(biological_instance) + "__" + "__".join(
        _part(value) for value in (predictor, predictor_split, model_seed)
    )
