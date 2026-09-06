# Codex execution packet: Git/bootstrap first

This packet is the first task for the local `H:\2026try\9.5PTL` workspace.
Do not start new scientific experiments until G0 passes.

## Required actions

1. Confirm the root is not already a Git repository.
2. Record the v1 evidence surface and preserve manuscript/result provenance.
3. Do not delete or rewrite raw data, processed data, results, packages or
   current manuscript outputs.
4. Treat `_github_ptl_work/` as a legacy checkout, not a second editable tree.
5. Initialize the root Git repository and connect it without force-pushing.
6. Fetch the remote ICLR branch without silently replacing local scientific
   assets; preserve both versions when conflicts are ambiguous.
7. Ignore biological data, generated results, submission bundles, caches,
   nested repositories and secrets by default.
8. Add text normalization attributes and keep large binaries outside Git.
9. Add an importable `ptl` package boundary with separate model extras.
10. Add the ICLR-v2 config, artifact and paper skeleton without destructive
    scientific migration.
11. Record external dependency provenance and license notes.
12. Add v1 provenance, G0-G7 checkpoints and a v2 path-portability check.
13. Make one small reviewable bootstrap commit and report its exact SHA.

## Safety constraints

Never force-push, delete v1 material, or use blanket `git add .` before the
ignore rules are validated. Never commit raw `.h5ad`, checkpoints, large
parquet/CSV matrices, credentials or local absolute-path caches.
