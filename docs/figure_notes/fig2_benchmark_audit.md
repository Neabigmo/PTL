# Figure 2 Benchmark Audit

- Message: The benchmark figure now reports dataset roles, split sizes, run distributions, context-distance diagnostics, and leakage-audit status.
- Input table: results/tables/all_metrics.csv, split_audit.csv, transfer_decay_summary.csv
- Visual encoding: Bars encode counts and distances; violins encode run-level fidelity; the audit matrix summarizes deterministic split checks.
- Limitation: Some audit entries are summarized from available Phase 06/08 tables rather than re-parsing raw split manifests.
