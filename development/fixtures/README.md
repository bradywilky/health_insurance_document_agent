# Health-insurance fixtures

Bingle-Dingle Insurance documents support selected-source claim research and table-ingestion development.evaluation.

## Selected-document research

Click **Load Bingle-Dingle examples** in the app to load eight text documents. The agreement, schedule, and amendment establish provider/product/date-specific rates. The draft tests that later documents do not automatically override executed terms. Benefits, authorization rules, and a claim packet add the facts needed to explain a hypothetical claim calculation.

For claim 0012, the packet supports 2 x USD 88 = USD 176 allowed; USD 50 deductible plus USD 25.20 coinsurance = USD 75.20 member responsibility; USD 100.80 insurer estimate. The supplied out-of-pocket balance does not reduce member responsibility further. These are estimates under the documented case assumptions, not claim approval or payment instructions.

The claim has four authorized sessions available before service and two afterward; annual benefit use goes from eight to ten of twenty. Tests must keep authorization units separate from benefit utilization. Deselecting benefits or the claim packet should produce missing-information explanations.

`documents/questions.json` contains thirteen questions and human-review expectations. Expected answers are never supplied to the answering model. Run `development/evaluation/documents.py --model maverick --output development/runs/insurance-documents.jsonl` after AWS authentication.

## Messy table ingestion

Run `python -m development.scripts.make_samples` to generate `data/insurance_mappings.xlsx` and `data/insurance_claims.csv`. The eight-sheet workbook deliberately includes narrative sheets, title rows above headers, repeated headers, duplicate mappings, duplicate column names, missing and zero billed amounts, and leading-zero identifiers. These parser records are separate from the narrative claim packet.

Known billed total is USD 445 across four non-missing records. Claim 0015 has a missing billed amount; claim 0016 has zero. NA is an intentional literal source code. Meadow's rendering-provider mapping is approved; River's is draft. Billed amounts must not be relabeled as allowed or paid amounts.

`insurance_questions.json` supplies seven table questions. Run `development/evaluation/tables.py --model maverick --output development/runs/insurance-tables.jsonl` after generating the samples. Offline tests create their own temporary workbooks without requiring AWS.

Keep actual input files and traces out of Git. This prototype does not implement clinical necessity rules or automatic claim adjudication.

## VDI extraction-warning regression pack

See [ROBUSTNESS.md](ROBUSTNESS.md) for synthetic upload files, expected warnings/rejections,
question-answering acceptance criteria, and commands for offline and live-model tests.
Generate it with `python -m development.scripts.make_robustness_samples --output data/robustness`.
