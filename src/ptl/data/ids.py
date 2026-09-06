"""Stable PTL-v2 identities.

The identifiers describe biological and prediction identity separately from
run-directory names.  In particular, ``run_id`` is never used as a fold key.
"""

from __future__ import annotations

from urllib.parse import quote


def _part(value: object | None) -> str:
    """Encode one ID component without punctuation-induced collisions.

    Components are case-folded for the project's semantic identifiers, then
    percent-encoded with the component delimiter excluded from the safe set.
    Unlike the old punctuation-stripping normalizer, values such as ``A-B``
    and ``A_B`` remain distinct. Missing values have an explicit marker so a
    literal ``na`` value is not conflated with absence.
    """

    if value is None or str(value).strip() == "":
        return "n~"
    text = str(value).strip().casefold()
    # ``_`` is an RFC 3986 unreserved character, so urllib leaves it alone
    # even when it is omitted from ``safe``.  Encode it explicitly because
    # the surrounding ID format uses ``__`` as its component delimiter.
    encoded = quote(text, safe=".-~").replace("_", "%5F")
    return "v~" + encoded


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
    """Identify one ground-truth response shared by predictors and seeds.

    The perturbation argument is the semantically validated signature-level
    instance key. Use :func:`perturbation_group_id` for repeated signatures
    sharing one perturbation label.
    """

    values = [_part(environment), _part(perturbation)]
    if dose is not None and str(dose).strip():
        values.append(_part(dose))
    if timepoint is not None and str(timepoint).strip():
        values.append(_part(timepoint))
    return "bio__" + "__".join(values)


def perturbation_group_id(environment: object, perturbation: object) -> str:
    """Identify a perturbation group without collapsing signature instances."""

    return "pert__" + "__".join((_part(environment), _part(perturbation)))


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
