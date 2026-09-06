# Codex execution packet: Git/bootstrap first

This packet is the first task for the local `H:\2026try\9.5PTL` workspace. Do not start new scientific experiments until this task passes.

## Goal

Turn the current local workspace root into the single Git source of truth for PTL-v2/ICLR 2027 while preserving all current PTL-v1 data, results and manuscript provenance.

## Required actions

1. Inspect the local root and confirm it is not currently a Git repository.
2. Back up/record the exact state of the current workspace before restructuring: file inventory, current code hash/checksum manifest for tracked-size source files, current manuscript path, current result-table inventory and current processed-data registry.
3. Do not delete or rewrite raw data, processed data, existing results, submission packages, or current manuscript outputs.
4. Treat `_github_ptl_work/` as a legacy/minimal Git checkout, not a second editable source tree. Preserve it until the root Git bootstrap is verified; then archive or remove it only in a later explicit cleanup commit.
5. Initialize the workspace root as Git and connect it to `https://github.com/Neabigmo/PTL.git` without force-pushing or rewriting existing remote history.
6. Fetch the remote and check out/create the local branch corresponding to `iclr2027-restructure` (which already exists remotely). Reconcile the remote minimal tree and the richer local tree carefully; local scientific assets must not be silently overwritten by the remote minimal snapshot.
7. Create a root `.gitignore` that ignores at minimum:
   - `data/raw/**`, `data/processed/**`, `data/external/**` large biological data
   - large model/checkpoint/cache directories
   - generated `results/**` by default except explicitly curated small source-data/manifests
   - `submit/**`, `submission_package/**`, compiled PDFs and temporary caches
   - third-party nested repository internals/checkpoints
   - Python caches and local environments
8. Create `.gitattributes` for text normalization and appropriate handling of CSV/TSV/JSON/YAML/Python/LaTeX/Markdown. Do not add Git LFS unless a tracked file genuinely requires it; prefer manifests over large binaries.
9. Create `pyproject.toml` for an importable `ptl` package and development/test tooling. Do not collapse incompatible predictor dependencies into one environment; define a light core environment and separate model-specific environments.
10. Create the ICLR-v2 package/config skeleton described in `docs/ICLR2027_REFACTOR_MASTERPLAN.md`, but do not move scientific logic destructively yet.
11. Create `artifacts/manifests/` and `artifacts/source_data/` as the only generated-artifact areas expected to be selectively tracked. Add README contracts explaining what may be committed.
12. Create `configs/third_party.lock.yaml` containing external repository name, URL, pinned commit SHA, role and license notes. Record the already present GEARS source and future Systema/scPertEval/model dependencies here.
13. Create `docs/V1_PROVENANCE_FREEZE.md` describing the current v1 evidence surface: 54 scPerturb raw files, current nine processed datasets, 525 baseline-run matrix, PTL tables, current CBAC manuscript, figures, and known limitations (GEARS adapter is contract-only; old absolute paths exist; confidence mixes context).
14. Create `docs/ICLR2027_EXECUTION_GATES.md` with gates G0-G7 from the master plan and status fields.
15. Add tests/checks for path portability: no new source/config file may contain hard-coded `H:\2026try\4.24` paths; historical generated v1 manifests may retain them as provenance but must not be used by v2 loaders.
16. Commit the bootstrap as a small, reviewable commit. Report changed files, branch, test/lint result, and any merge/conflict decision. Do not begin data reprocessing in the same commit.

## Safety constraints

- No force push.
- No deletion of v1 data/results/manuscript during bootstrap.
- No blanket `git add .` until ignore rules are validated with `git status`.
- Never commit raw `.h5ad`, `.npz`, large parquet matrices, checkpoints, credentials, local absolute-path caches or environment secrets.
- Do not rewrite the remote CBAC/methods branches.
- If local/remote source conflicts are ambiguous, preserve both versions and report rather than choosing silently.

## G0 acceptance criteria

G0 is complete only when:

- local workspace root is a Git repo on `iclr2027-restructure`;
- remote origin points to `Neabigmo/PTL`;
- `git status` does not propose staging raw/processed biological data or large generated result trees;
- importable package skeleton/configs exist;
- v1 provenance is documented and untouched;
- path portability policy exists;
- bootstrap tests/checks pass;
- the exact commit SHA is reported.

After G0, stop and request review before G1 data-registry work.