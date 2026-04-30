---
"@platforma-open/milaboratories.spatiotemporal-analysis": patch
---

Force categorical metadata (subject, grouping, timepoint) to String at CSV read so polars never infers Float64 from numeric-looking values, and normalize values via JS-compatible canonicalization so backend output (e.g. `5.0`) matches the UI's `--timepoint-order` (`5`).
