# PTL-v2 configuration boundary

All new configurations must use repository-relative paths resolved from the
project root. Dataset and environment registries are introduced in G1. The
existing `config/` directory is preserved as v1 provenance during G0 and is
not silently rewritten.
