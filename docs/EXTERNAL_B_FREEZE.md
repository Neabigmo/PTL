# External-B freeze

Date: 2026-09-06

External-B was selected before any PTL-v2 performance evaluation. The decision
uses only the G1 metadata audit, not existing PTL scores or pilot outcomes.

| Candidate | Perturbation/control completeness | Size | Metadata clarity | Context distinctness | Predictor compatibility | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WesselsSatija2023 | 2 | 2 | 2 | 2 | 2 | **10** |
| JoungZhang2023_atlas | 1 | 2 | 1 | 2 | 1 | 7 |
| JoungZhang2023_combinatorial | 2 | 2 | 1 | 2 | 1 | 8 |

The frozen winner is **WesselsSatija2023**. It has a manageable 30,707-cell
surface, 187 perturbation categories, an explicit `Guide.Class` control signal
(`NT`), rich guide/HTO/lane metadata, and a distinct THP-1/monocyte context.
The Joung atlas is much larger but its local perturbation labels are numeric
and the audit did not identify an explicit control label in the primary
metadata; the combinatorial subset is useful but less complete for the
primary External-B role.

From this point, WesselsSatija2023 is excluded from PTL-v2 feature selection,
architecture selection, and threshold tuning. It can be opened only as the
pre-declared external robustness check after the core pipeline is frozen.
