# Technical debt tracking (Phase 1)

GitHub issues are preferred for ongoing tracking. Items below mirror in-code `TODO`/`HACK` markers
until migrated.

| Location | Marker | Summary |
|----------|--------|---------|
| `analyze/analyze.py` | TODO ~220 | `get_ids_object_of_interest` filter/index selection |
| `utilities.py` | TODO ~77 | Split `Image` into base + analysis class |
| `analyze/calibration.py` | TODO | Add filter metadata to transformed magnitudes |
| `analyze/extraction.py` | TODO | Unit check for CCDData |
| `reduce/redu.py` | TODO | Multiple reduction edge cases |
| `reduce/registration.py` | TODO | Registration refinements |
| `analyze/correlate/inter.py` | TODO | Correlation improvements |
| `analyze/plots.py` | TODO | Plot module cleanup |

See `rg 'TODO|FIXME|HACK|dirty' src/` for the full current list.
