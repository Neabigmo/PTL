# Phase 00 - Project Decomposition

## Objective
Initialize the project without downloading data, training models, running experiments, or drafting manuscript text. Create the required directory structure, concise task packets, and tracking files.

## Required input files
- PROJECT_RULES.md
- CODEx_MASTER_COMMAND.md

## Output files
- docs/task_packets/00_project_decomposition.md through 11_latex_submission_package.md
- docs/project_status.md
- docs/file_registry.md
- docs/decision_log.md
- docs/manual_intervention_needed.md
- docs/task_logs/phase00_project_decomposition.md
- Required empty directories under data/, src/, results/, docs/, and manuscript/

## Detailed steps
1. Read PROJECT_RULES.md and CODEx_MASTER_COMMAND.md.
2. Create the project directory tree exactly required by the master command.
3. Write one concise, self-contained task packet per phase.
4. Initialize project_status.md with current phase, completed phases, blockers, next action, and timestamp.
5. Initialize file_registry.md with all Phase 00 outputs and expected future placeholder documents.
6. Initialize decision_log.md with Phase 00 setup decisions.
7. Initialize manual_intervention_needed.md with no active issues.
8. Write a short task log for Phase 00.
9. Verify all required paths exist.
10. Verify all task packets are short enough for token-saving execution.

## Commands
PowerShell directory/file checks only:

```powershell
Get-ChildItem docs/task_packets
Get-ChildItem -Recurse data,src,results,docs,manuscript
```

## Logging requirements
- Record Phase 00 actions in docs/task_logs/phase00_project_decomposition.md.
- Do not create download, training, evaluation, or LaTeX logs in this phase.

## Acceptance criteria
- All 12 task packet files exist.
- Tracking files exist.
- Required directory structure exists.
- No data download, model training, experiment, or manuscript drafting has started.
- Task packets are concise and independently executable.

## Next task dependency
Phase 01 may begin only after Phase 00 acceptance criteria are satisfied.
