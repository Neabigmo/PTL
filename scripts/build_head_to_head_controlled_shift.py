"""Build a leakage-safe external controlled-shift ladder from SCEPTRE results.

The external head-to-head resource contains guide-level perturbation effects,
not model predictions.  This script therefore uses an explicitly named frozen
source-response persistence baseline: a target's source-context mean response
is persisted as the prediction for each destination-context guide.  The
destination guide effects are evaluation outcomes only.  This makes risk
asymmetric when destination guide reproducibility differs, while keeping the
response-program shift descriptive rather than mechanistic or causal.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_COLUMNS = {
    "response_id",
    "grna_id",
    "pass_qc",
    "log_2_fold_change",
}
REFERENCE_COLUMNS = {"guide_id_long", "modality"}
ALLOWED_MODALITIES = {"CRISPRko", "CRISPRi"}


def _load_config(root: Path, path: Path | None) -> dict[str, Any]:
    config_path = path or root / "configs/external_head_to_head_crisprko_crispri.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported external controlled-shift config schema")
    return payload


def _load_reference(root: Path, relative_path: str) -> pd.DataFrame:
    path = root / relative_path
    frame = pd.read_csv(path, dtype=str)
    missing = REFERENCE_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"guide reference is missing columns: {sorted(missing)}")
    frame = frame.loc[frame["modality"].isin(ALLOWED_MODALITIES), ["guide_id_long", "modality"]].copy()
    frame["guide_id_long"] = frame["guide_id_long"].astype(str)
    if frame["guide_id_long"].duplicated().any():
        raise ValueError("reference contains duplicate guide IDs after modality filtering")
    frame["target"] = frame["guide_id_long"].str.rsplit("_", n=1).str[0]
    return frame


def _load_sample(root: Path, data_dir: Path, spec: dict[str, Any], reference: pd.DataFrame) -> dict[str, Any]:
    path = root / data_dir / str(spec["file"])
    if not path.is_file():
        raise FileNotFoundError(f"missing SCEPTRE result file: {path}")
    header = pd.read_csv(path, nrows=0)
    missing = REQUIRED_COLUMNS.difference(header.columns)
    if missing:
        raise ValueError(f"{path.name} is missing SCEPTRE columns: {sorted(missing)}")
    usecols = ["response_id", "grna_id", "pass_qc", "log_2_fold_change"]
    guide_lookup = reference.set_index("guide_id_long")
    allowed_guides = set(guide_lookup.index)
    pieces: list[pd.DataFrame] = []
    raw_rows = 0
    qc_rows = 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=250_000, low_memory=False):
        raw_rows += len(chunk)
        chunk = chunk.loc[chunk["grna_id"].astype(str).isin(allowed_guides)].copy()
        pass_qc = chunk["pass_qc"].astype(str).str.strip().str.casefold().isin({"true", "1", "yes"})
        chunk = chunk.loc[pass_qc].copy()
        chunk["effect"] = pd.to_numeric(chunk["log_2_fold_change"], errors="coerce")
        chunk = chunk.loc[np.isfinite(chunk["effect"].to_numpy(dtype=float))].copy()
        if chunk.empty:
            continue
        qc_rows += len(chunk)
        chunk["grna_id"] = chunk["grna_id"].astype(str)
        chunk["response_id"] = chunk["response_id"].astype(str)
        chunk["modality"] = chunk["grna_id"].map(guide_lookup["modality"])
        chunk["target"] = chunk["grna_id"].map(guide_lookup["target"])
        pieces.append(chunk[["response_id", "grna_id", "modality", "target", "effect"]])
    if not pieces:
        raise ValueError(f"{path.name} has no reference-mapped QC-passing guide effects")
    filtered = pd.concat(pieces, ignore_index=True)
    del pieces
    gc.collect()
    # The published SCEPTRE tables are expected to be one row per guide/response.
    # Mean aggregation is deterministic and makes the contract robust to a
    # duplicated export row without silently choosing an arbitrary record.
    pivot = filtered.pivot_table(
        index=["modality", "target", "grna_id"],
        columns="response_id",
        values="effect",
        aggfunc="mean",
        observed=True,
        sort=True,
    )
    pivot.columns = pivot.columns.astype(str)
    pivot = pivot.sort_index(axis=1)
    report = {
        "file": str(path.relative_to(root).as_posix()),
        "raw_rows_scanned": int(raw_rows),
        "qc_passing_reference_mapped_rows": int(qc_rows),
        "modalities": sorted(str(x) for x in pivot.index.get_level_values("modality").unique()),
        "targets_by_modality": {
            str(modality): int(pivot.xs(modality, level="modality").index.get_level_values("target").nunique())
            for modality in pivot.index.get_level_values("modality").unique()
        },
        "guide_rows": int(len(pivot)),
        "response_ids": int(pivot.shape[1]),
    }
    return {"spec": spec, "pivot": pivot, "report": report}


def _finite_correlation(left: np.ndarray, right: np.ndarray, method: str) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 4 or np.std(left[mask]) == 0.0 or np.std(right[mask]) == 0.0:
        return float("nan")
    if method == "spearman":
        return float(spearmanr(left[mask], right[mask]).statistic)
    return float(kendalltau(left[mask], right[mask], nan_policy="omit").statistic)


def _rank_stats(left_risk: np.ndarray, right_risk: np.ndarray, left_conf: np.ndarray, right_conf: np.ndarray) -> dict[str, Any]:
    if len(left_risk) < 2:
        return {
            "n_comparable_risk_pairs": 0,
            "n_risk_inversions": 0,
            "risk_inversion_rate": float("nan"),
            "confidence_tracking_rate": float("nan"),
            "n_confidence_comparable_inversions": 0,
            "risk_rank_kendall": float("nan"),
            "risk_rank_spearman": float("nan"),
            "mean_normalized_rank_displacement": float("nan"),
        }
    i, j = np.triu_indices(len(left_risk), k=1)
    ld = left_risk[i] - left_risk[j]
    rd = right_risk[i] - right_risk[j]
    ls = np.sign(ld)
    rs = np.sign(rd)
    comparable = np.isfinite(ld) & np.isfinite(rd) & (ls != 0.0) & (rs != 0.0)
    inversions = comparable & (ls != rs)
    lc = np.sign(left_conf[i] - left_conf[j])
    rc = np.sign(right_conf[i] - right_conf[j])
    conf_comparable = inversions & (lc != 0.0) & (rc != 0.0)
    tracked = conf_comparable & (lc == -ls) & (rc == -rs)
    denom = max(len(left_risk) - 1, 1)
    displacement = np.abs(rankdata(left_risk, method="average") - rankdata(right_risk, method="average")) / denom
    return {
        "n_comparable_risk_pairs": int(comparable.sum()),
        "n_risk_inversions": int(inversions.sum()),
        "risk_inversion_rate": float(inversions.sum() / comparable.sum()) if comparable.any() else float("nan"),
        "confidence_tracking_rate": float(tracked.sum() / inversions.sum()) if inversions.any() else float("nan"),
        "n_confidence_comparable_inversions": int(conf_comparable.sum()),
        "risk_rank_kendall": _finite_correlation(left_risk, right_risk, "kendall"),
        "risk_rank_spearman": _finite_correlation(left_risk, right_risk, "spearman"),
        "mean_normalized_rank_displacement": float(np.mean(displacement)),
    }


def _pair_target_rows(left: dict[str, Any], right: dict[str, Any], comparison: dict[str, Any], modality: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    left_pivot = left["pivot"]
    right_pivot = right["pivot"]
    if modality not in left_pivot.index.get_level_values("modality") or modality not in right_pivot.index.get_level_values("modality"):
        return []
    left_slice = left_pivot.xs(modality, level="modality")
    right_slice = right_pivot.xs(modality, level="modality")
    targets = sorted(set(left_slice.index.get_level_values("target")) & set(right_slice.index.get_level_values("target")))
    response_ids = sorted(set(left_pivot.columns) & set(right_pivot.columns))
    min_responses = int(config["analysis"].get("minimum_shared_response_ids", 100))
    rows: list[dict[str, Any]] = []
    for target in targets:
        left_guides = left_slice.xs(target, level="target").loc[:, response_ids]
        right_guides = right_slice.xs(target, level="target").loc[:, response_ids]
        valid = np.isfinite(left_guides.to_numpy(dtype=float)).all(axis=0) & np.isfinite(right_guides.to_numpy(dtype=float)).all(axis=0)
        if int(valid.sum()) < min_responses or len(left_guides) < 2 or len(right_guides) < 2:
            continue
        left_matrix = left_guides.to_numpy(dtype=np.float64)[:, valid]
        right_matrix = right_guides.to_numpy(dtype=np.float64)[:, valid]
        left_mean = left_matrix.mean(axis=0)
        right_mean = right_matrix.mean(axis=0)
        left_within = float(np.mean((left_matrix - left_mean[None, :]) ** 2))
        right_within = float(np.mean((right_matrix - right_mean[None, :]) ** 2))
        left_risk = float(np.mean((right_matrix - left_mean[None, :]) ** 2))
        right_risk = float(np.mean((left_matrix - right_mean[None, :]) ** 2))
        left_norm = float(np.linalg.norm(left_mean))
        right_norm = float(np.linalg.norm(right_mean))
        cosine_shift = float("nan") if left_norm <= 1e-12 or right_norm <= 1e-12 else float(1.0 - np.dot(left_mean, right_mean) / (left_norm * right_norm))
        rows.append({
            "comparison_id": str(comparison["comparison_id"]),
            "axis": str(comparison["axis"]),
            "left_sample": str(comparison["left"]),
            "right_sample": str(comparison["right"]),
            "modality": modality,
            "target": str(target),
            "risk_left_source_to_right_guides": left_risk,
            "risk_right_source_to_left_guides": right_risk,
            "confidence_left_source": float(-np.log1p(left_within)),
            "confidence_right_source": float(-np.log1p(right_within)),
            "source_within_guide_mse_left": left_within,
            "source_within_guide_mse_right": right_within,
            "response_program_shift": cosine_shift,
            "absolute_risk_change": abs(left_risk - right_risk),
            "left_guide_count": int(len(left_guides)),
            "right_guide_count": int(len(right_guides)),
            "shared_response_id_count": int(valid.sum()),
            "response_program_is_descriptive": 1,
            "destination_outcomes_used_for_evaluation_only": 1,
            "predictor": "source_response_persistence_baseline",
        })
    return rows


def _bootstrap_summary(detail: pd.DataFrame, seed: int, replicates: int) -> dict[str, float]:
    if len(detail) < 4:
        return {"response_shift_mean_bootstrap_low": float("nan"), "response_shift_mean_bootstrap_high": float("nan"), "shift_risk_change_spearman_bootstrap_low": float("nan"), "shift_risk_change_spearman_bootstrap_high": float("nan")}
    rng = np.random.default_rng(seed)
    shift_means: list[float] = []
    correlations: list[float] = []
    shifts = detail["response_program_shift"].to_numpy(dtype=float)
    changes = detail["absolute_risk_change"].to_numpy(dtype=float)
    for _ in range(replicates):
        indexes = rng.integers(0, len(detail), size=len(detail))
        shift_means.append(float(np.nanmean(shifts[indexes])))
        correlations.append(_finite_correlation(shifts[indexes], changes[indexes], "spearman"))
    return {
        "response_shift_mean_bootstrap_low": float(np.nanquantile(shift_means, 0.025)),
        "response_shift_mean_bootstrap_high": float(np.nanquantile(shift_means, 0.975)),
        "shift_risk_change_spearman_bootstrap_low": float(np.nanquantile(correlations, 0.025)),
        "shift_risk_change_spearman_bootstrap_high": float(np.nanquantile(correlations, 0.975)),
    }


def _permutation_pvalue(left: np.ndarray, right: np.ndarray, seed: int, replicates: int) -> float:
    observed = _finite_correlation(left, right, "spearman")
    if not np.isfinite(observed):
        return float("nan")
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(replicates):
        null.append(_finite_correlation(left, rng.permutation(right), "spearman"))
    null_array = np.asarray(null, dtype=float)
    null_array = null_array[np.isfinite(null_array)]
    return float((1 + np.sum(np.abs(null_array) >= abs(observed))) / (1 + len(null_array)))


def _summarize(detail: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    analysis = config["analysis"]
    for keys, group in detail.groupby(["comparison_id", "axis", "left_sample", "right_sample", "modality"], sort=True):
        comparison_id, axis, left_sample, right_sample, modality = keys
        left_risk = group["risk_left_source_to_right_guides"].to_numpy(dtype=float)
        right_risk = group["risk_right_source_to_left_guides"].to_numpy(dtype=float)
        left_conf = group["confidence_left_source"].to_numpy(dtype=float)
        right_conf = group["confidence_right_source"].to_numpy(dtype=float)
        stats = _rank_stats(left_risk, right_risk, left_conf, right_conf)
        shift = group["response_program_shift"].to_numpy(dtype=float)
        changes = group["absolute_risk_change"].to_numpy(dtype=float)
        seed = int(analysis.get("random_seed", 20260908)) + sum(ord(c) for c in str(comparison_id) + str(modality))
        row = {
            "comparison_id": comparison_id,
            "axis": axis,
            "left_sample": left_sample,
            "right_sample": right_sample,
            "modality": modality,
            "n_matched_targets": int(len(group)),
            "mean_shared_response_ids": float(group["shared_response_id_count"].mean()),
            "mean_response_program_shift": float(np.nanmean(shift)),
            "median_response_program_shift": float(np.nanmedian(shift)),
            "response_shift_risk_change_spearman": _finite_correlation(shift, changes, "spearman"),
            "response_shift_risk_change_permutation_p": _permutation_pvalue(shift, changes, seed, int(analysis.get("permutation_replicates", 2000))),
            **stats,
            "predictor": "source_response_persistence_baseline",
            "response_program_is_descriptive": 1,
            "destination_outcomes_used_for_evaluation_only": 1,
            "target_matching": "exact_intersection",
            "response_matching": "finite_intersection_per_target",
            "confidence_source": "source_within_target_guide_reproducibility",
        }
        row.update(_bootstrap_summary(group, seed, int(analysis.get("bootstrap_replicates", 1000))))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["axis", "comparison_id", "modality"], kind="stable")


def run(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    config = _load_config(root, config_path)
    reference = _load_reference(root, str(config["reference_guides"]))
    data_dir = Path(str(config["data_dir"]))
    sample_data: dict[str, dict[str, Any]] = {}
    reports: dict[str, Any] = {}
    for sample_id, spec in config["samples"].items():
        sample_data[str(sample_id)] = _load_sample(root, data_dir, spec, reference)
        reports[str(sample_id)] = sample_data[str(sample_id)]["report"]
    detail_rows: list[dict[str, Any]] = []
    for comparison in config["comparisons"]:
        left = sample_data[str(comparison["left"])]
        right = sample_data[str(comparison["right"])]
        modalities = sorted(set(left["pivot"].index.get_level_values("modality")) & set(right["pivot"].index.get_level_values("modality")))
        declared_left = left["spec"].get("modality")
        declared_right = right["spec"].get("modality")
        if declared_left:
            modalities = [str(declared_left)]
        if declared_right:
            modalities = [modality for modality in modalities if modality == declared_right]
        for modality in modalities:
            detail_rows.extend(_pair_target_rows(left, right, comparison, str(modality), config))
    if not detail_rows:
        raise ValueError("controlled-shift ladder produced no matched target rows")
    detail = pd.DataFrame(detail_rows).sort_values(["axis", "comparison_id", "modality", "target"], kind="stable")
    summary = _summarize(detail, config)
    min_targets = int(config["analysis"].get("minimum_targets", 20))
    undersized = summary.loc[summary["n_matched_targets"] < min_targets]
    if not undersized.empty:
        raise ValueError(f"controlled comparisons fell below minimum target intersection: {undersized[['comparison_id', 'modality', 'n_matched_targets']].to_dict('records')}")
    out_dir = root / "artifacts/manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "head_to_head_controlled_shift_ladder.csv"
    summary_path = out_dir / "head_to_head_controlled_shift_ladder_summary.csv"
    report_path = out_dir / "head_to_head_controlled_shift_ladder_summary.json"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    payload = {
        "schema_version": 1,
        "benchmark_id": config["benchmark_id"],
        "config_path": str((root / "configs/external_head_to_head_crisprko_crispri.json").relative_to(root).as_posix()),
        "input_reports": reports,
        "comparison_count": int(summary.shape[0]),
        "detail_rows": int(detail.shape[0]),
        "summary_path": summary_path.relative_to(root).as_posix(),
        "detail_path": detail_path.relative_to(root).as_posix(),
        "estimand": "source-response persistence risk evaluated on destination guide-level SCEPTRE effects",
        "response_representation": "SCEPTRE log_2_fold_change across finite response-id intersections",
        "risk_definition": "mean squared error from source target mean to destination target guide effects",
        "confidence_definition": config["analysis"]["confidence_definition"],
        "response_program_shift_definition": "1 minus cosine similarity of source and destination target means",
        "controls": ["exact target intersection", "reference-mapped guides only", "CRISPRko/CRISPRi modalities separated", "finite response intersection per target", "cluster bootstrap over matched targets", "target-label permutation for shift/risk association"],
        "limitations": ["SCEPTRE effects are not predictions from a trained perturbation model", "guide-level effects are an evaluation outcome and can include measurement variance", "response-program associations are descriptive and not causal mechanism evidence"],
        "status": "executed",
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve(), args.config.resolve() if args.config else None), indent=2, ensure_ascii=False, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
