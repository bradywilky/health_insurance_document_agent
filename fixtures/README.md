# Synthetic fishing fixtures

All names, rules, product versions, and observations are invented. No external datasets or real product specifications are used.

`fishing_mappings.xlsx` is a deliberately irregular eight-sheet workbook:

| Sheet | Purpose / test difficulty |
| --- | --- |
| About | Free text; an important location-export rule is on row 10 |
| Version History | Narrative change records, including a unit change |
| Attribute Dictionary | Canonical field definitions and units |
| CastLog Mapping | Merged title, header on row 4, blank row, repeated header, footer note |
| RiverTrack Mapping | Structured table with an intentional duplicate mapping |
| TournamentDesk Mapping | Header on row 3, duplicate Notes headings, conflicting statuses and units, unmapped attribute |
| Species Codes | Categorical lookup, including literal code NA |
| Catch Samples | Text IDs with leading zeros, units, missing mass and genuine zero |

`fishing_catches.csv` contains the Catch Samples table for single-file tests. It does not include the mapping/definition sheets; the agent should say when needed information is absent. `fishing_questions.json` supplies ten questions and expected findings for the workbook. Expected answers are independent test evidence and are never given to the answering agent.

## Try individual questions

```powershell
.\.venv\Scripts\python run_local.py fixtures\fishing_mappings.xlsx "What happens to exact fishing locations before export?" --trace runs\location.json
.\.venv\Scripts\python run_local.py fixtures\fishing_mappings.xlsx "List approved mass_g mappings across all products and note duplicates."
.\.venv\Scripts\python run_local.py fixtures\fishing_mappings.xlsx "Which product has no water temperature mapping?"
.\.venv\Scripts\python run_local.py fixtures\fishing_catches.csv "Which catches have missing mass, and which have zero mass?"
```

## Inspect and export preprocessing

```powershell
.\.venv\Scripts\python run_local.py fixtures\fishing_mappings.xlsx Inspect --inspect
.\.venv\Scripts\python preprocess.py fixtures\fishing_mappings.xlsx --output data\preprocessed\fishing_mappings.xlsx
.\.venv\Scripts\python run_local.py data\preprocessed\fishing_mappings.xlsx "What changed in version 2.0?" --preprocessed
```

The export creates one CSV per sheet plus `_metadata.json`. The output directory must be new to prevent stale CSVs. Metadata retains the existing `sheets`, `sheet_type`, `row_count`, `column_count`, and `columns` contract, with added header/source row positions, context rows, warnings, distinct counts, and the eight most frequent values per column. `row_count` for narrative metadata sheets counts nonempty cells, each represented by source row, source column, and text.

Data headers are detected from the first 30 rows. Blank rows and exactly repeated headers are excluded with warnings; duplicate records are retained. Sparse notes prefixed `Note:`, `Source:`, or `Footnote:` are kept as context. Blank and duplicate headers get unique names. Text IDs with leading zeros and literal NA codes survive preprocessing and agent reads.

Classification and header detection are heuristics. Override ambiguous layouts using JSON:

```json
{"Version History": {"sheet_type": "data", "header_row": 1}}
```

Pass the file with `--overrides path\overrides.json` to either preprocessing or `run_local.py`. Header rows use Excel's one-based row numbers. This version handles one table per sheet, not separate table blocks or hierarchical multirow headers. It does not forward-fill merged body cells or calculate Excel formulas; formula values must already be cached in the input. Notes/comments, shapes, images, and formatting are not extracted into CSV. Original input workbooks should be retained separately.

## Evaluate real models

```powershell
.\.venv\Scripts\python evaluate.py --model maverick --case late_note --case identifier --output runs\maverick-check.jsonl
.\.venv\Scripts\python evaluate.py --model scout --output runs\scout-suite.jsonl
```

These are paid Bedrock calls using the configured profile. The JSONL report includes question, expected findings, actual answer, time, calls, and token counts. Successful execution means `needs_human_review`, not a passing answer. Review completeness, source attribution, status distinctions, duplicates, conversions, and refusal to invent facts. The runner stops on an error and preserves completed records. Cross-sheet computations currently rely on several single-sheet tool calls and model synthesis; the conversion case deliberately tests that limitation.

Offline checks: `.\.venv\Scripts\python -m pytest -q`. No Bedrock calls occur during these tests.

## Rebuilding

The workbook is checked in so Python users do not need a JS authoring dependency. `build_fishing.mjs` records how it was authored; rebuilding requires Node and `@oai/artifact-tool`. Invoke `node fixtures/build_fishing.mjs` in an environment providing that package. It writes the workbook and sheet previews under `outputs/fishing/`. No rebuild is needed to test or deploy the Python agent.
