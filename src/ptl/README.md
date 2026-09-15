# PTL-v2 package boundary

This namespace is the new source boundary for the ICLR 2027 refactor. G0 only
creates the boundary; scientific logic is migrated incrementally in G1-G6.

Subpackages are intentionally separated by contract:

- `data` - registries, harmonization, signatures and gene spaces
- `predictors` - prediction-only interfaces and implementations
- `uncertainty` - native, ensemble and normalized UQ
- `reliability` - feature blocks, baselines, calibrators and cross-fitting
- `evaluation` - fidelity, selective risk, calibration and statistics
- `experiments` - RQ1-RQ4 orchestration
- `plotting` - paper-facing figure generation
