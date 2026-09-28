# Health Insurance Document Agent

`health_insurance_document_agent` answers questions across selected health-insurance documents and tables using Llama Maverick or Scout through Amazon Bedrock. It includes a local chat interface, document extraction, source evidence, and deterministic table tools. The existing `run_plan_tables_agent()` entry points and S3 loading paths remain available in `table_agent.py`.

## Multi-file chat prototype

After installing dependencies and configuring your local AWS profile, start the interface:

```powershell
.\.venv\Scripts\python -m streamlit run app.py
```

Open http://127.0.0.1:8501. Upload files, click **Add uploaded files**, and use **Files to answer from** to select one or more sources. Alternatively, click **Load Bingle-Dingle examples**. Changing selected files clears the conversation.

The interface accepts text PDFs, DOCX, XLSX, XLS, CSV, TXT, and Markdown, with at most 10 files and 20 MB per file. PDF references use page numbers; Word references use paragraph/table locations; spreadsheets preserve sheet and row references. Extraction warnings and the retrieved evidence appear alongside answers. Choose Maverick or Scout; `BEDROCK_MODEL_ID`, if set, overrides that choice.

`documents.py` handles extraction and lexical retrieval. `health_insurance_document_agent.py` limits every tool to the selected document IDs and reuses the existing Bedrock wrapper. Table questions use validated filters, aggregates and within-workbook joins. Generated Python execution is disabled in this chat interface. `app.py` keeps files and chat in session memory; selected excerpts and conversation context are sent to Bedrock. Nothing is uploaded to S3 by this prototype.

This is a localhost development application, without shared storage or authentication. File selection is question scope, not a replacement for authorization. A shared deployment needs server-enforced user/document permissions, hardened ingestion, resource limits, and persistent versioned source storage. Automatic bucket discovery, OCR, cross-file table joins, and semantic retrieval are not implemented. PDF layout and Word headers, footers, text boxes, and tracked changes may not extract completely. Citations identify retrieved evidence; they do not independently prove that a model interpreted it correctly.

**Load Bingle-Dingle examples** loads eight Bingle-Dingle health-insurance documents: provider agreement, rate schedule, executed amendment, processing guide, unexecuted draft, benefit summary, authorization rules, and claim packet. Bingle-Dingle Insurance is a regional health insurer offering the Meadow product. The documents cover provider terms, benefits, authorization, and claim-specific records.

Try: "For claim 0012, explain the allowed amount, member responsibility, and insurer estimate." The complete selected packet supports an allowed amount of USD 176.00, member responsibility of USD 75.20, and insurer estimate of USD 100.80 under explicit simplified assumptions. These differ from the billed USD 220.00 and from a final payment decision. Deselect the benefit summary and repeat: the agent should identify missing terms instead of using the previous answer.

Thirteen document evaluation cases cover effective dates, draft precedence, product scope, missing sources, authorization, benefit utilization, and cost sharing:

```powershell
.\.venv\Scripts\python evaluate_documents.py --model maverick --output runs\documents-maverick.jsonl
```

This makes billable model calls. Results require human comparison with the expected answers; offline tests use scripted responses and do not measure model quality.

## Quick start (PowerShell)

Python 3.11+ is required. From this project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
.\.venv\Scripts\python make_samples.py
.\.venv\Scripts\python -m pytest -q
```

If your Python installation cannot bootstrap pip, use `python -m pip --python .\.venv\Scripts\python.exe install -r requirements-dev.txt`.

Inspect preprocessing without credentials or model calls:

```powershell
.\.venv\Scripts\python run_local.py data\insurance_mappings.xlsx "Inspect" --inspect
```

Run the real agent using your existing local AWS profile:

```powershell
.\.venv\Scripts\python run_local.py data\insurance_mappings.xlsx "What is the total known billed amount in Claim Samples?" --profile dev-profile --model maverick --trace runs\maverick.json
.\.venv\Scripts\python run_local.py data\insurance_claims.csv "What is the total known billed amount?" --profile dev-profile --model scout --stream
```

Replace the file path and question to test your own files. CSV options include `--encoding cp1252` and `--delimiter ';'`. `.xlsx` and `.xls` files load all worksheets. The local preprocessor detects headers below title rows and preserves free-text metadata sheets. Explicit header/type overrides handle ambiguous layouts. Multiple tables per sheet and uncached formulas are not supported. See fixtures/README.md for details.

Expected table results: known billed amount **USD 445**, one missing amount (claim **0015**) and one zero amount (claim **0016**). Claim **0012** retains its leading zeros. The workbook includes narrative About and Version History sheets, mappings with duplicates and draft status, repeated headers, and source-row provenance. These independent parser samples are not a combined claim ledger for the narrative packet.

## Personal AWS setup

Local data + personal-account Bedrock closely reproduces the existing LLM API without creating a bucket. Configure a named AWS SDK profile using your account's normal credentials or IAM Identity Center flow (`aws configure sso --profile dev-profile`, then `aws sso login --profile dev-profile`, when applicable). Do not put access keys in Python files. Use synthetic data initially; local loading still sends samples, schemas, and tool results to Bedrock. Calls incur AWS charges.

Set `AWS_PROFILE=dev-profile` in `.env`, or pass `--profile`. The default region is `us-east-1`; use `--region` to override it. The account/role needs `bedrock:InvokeModel` permission for the selected inference profile and underlying models in its destination regions. Organizational policies, regional availability, and account access can also affect invocation. No S3 permissions are needed for local mode.

Defaults:

| Selection | Bedrock inference profile |
| --- | --- |
| maverick | `us.meta.llama4-maverick-17b-instruct-v1:0` |
| scout | `us.meta.llama4-scout-17b-instruct-v1:0` |

`BEDROCK_MODEL_ID` overrides the selection when your deployment uses a specific inference profile ARN or ID. Remove that override when comparing `--model scout` and `--model maverick`. `BEDROCK_MAX_TOKENS` controls the output limit per call. These US profiles may route requests across US regions.

AWS references: [Maverick](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-meta-llama-4-maverick-17b-instruct.html), [Scout](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-meta-llama-4-scout-17b-instruct.html), [using inference profiles](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-use.html).

## Files and AWS migration

- `table_agent.py`: original graph plus JSON retry, recursion budget, streaming, literal/numeric search, empty parallel extraction, attribution, and instruction/data separation fixes.
- `llm.py`: Bedrock Converse wrapper with normal SDK profile/role credentials, configurable region, timeouts, and retries.
- `config_utils.py`: missing configuration module, now supplied with Maverick/Scout defaults and optional business context. Retain or reconcile your existing AWS version of this module; the original business context was not supplied.
- `local_data.py`: in-memory `get_object` adapter and metadata generation; no AWS calls.
- `run_local.py`: CLI with optional full LLM traces and token totals. Trace files contain input data and are ignored by Git.
- `make_samples.py`: deterministic sample files.
- `tests/test_agent.py`: offline regression tests using scripted LLM responses and the real graph/data tools. They do not measure Llama's answer quality.

In AWS, keep calling the existing entry point with the real S3 client:

```python
import boto3
from table_agent import run_plan_tables_agent

answer = run_plan_tables_agent(
    s3_client=boto3.client("s3"),
    s3_bucket="your-bucket",
    s3_prefix="your-prefix/",
    plan_domain="your-domain",
    filename="your-workbook.xlsx",
    query="What is the total known billed amount?",
)
```

Use the deployment's IAM role by leaving `AWS_PROFILE` unset. Do not copy the personal `.env` into AWS. The S3 prefix must include its trailing slash. Existing preprocessing must provide `_metadata.json` plus each sheet's CSV at `{prefix}{domain}/preprocessed/{filename}/`; for CSV inputs the object basename is the original CSV filename. The new preprocess.py exports that contract locally; it does not upload to S3 or validate another preprocessing pipeline.

## Limits of this development harness

The original table CLI retains generated pandas code through in-process `exec`. Restricted builtins and prompt instructions **are not a sandbox**: pandas can access files and networks. Use trusted development files/questions for that CLI. The new multi-file chat does not expose this tool. Isolate generated code in a separate execution service with resource, filesystem, network, and credential restrictions before exposing it to untrusted users.

Search outputs can be truncated, and overview extraction sees only five sample rows. Tool failures and the step limit may yield incomplete answers. Identical table tool calls reuse cached evidence; the document loop stops repeated requests with a limitation. CLI calls are independent questions; existing Python entry points still accept conversation history.

Deterministic `query_table` supports filters, grouping, sorting, pagination, decimal aggregates, and explicit unit conversions. `join_tables` checks relationship constraints and caps output size. Decimal arithmetic preserves loaded numeric values; it cannot restore precision already lost in a source workbook. Missing totals remain missing instead of silently becoming zero. The evidence pipeline bounds model context and retains tool parameters and source references. Its numeric display check is a guard against omitted exact values, not a semantic answer verifier.

For model evaluation, run the same questions with both models and compare answers to known totals. Inspect traces for tool selection, generated code, failures, and token use. Offline passing tests are not evidence that a live AWS profile or model invocation works.


## Publishing to GitHub

Commit source code, tests, dependency files, and `.env.example`. The example uses a placeholder profile name; keep your actual settings in `.env` (ignored by Git). AWS login credentials remain in your user-level AWS configuration/cache outside this project; the source code loads them through the SDK and contains no account-specific credentials.

The ignore rules exclude `.env` variants, `.aws` folders, credential/key files, logs, local data, model traces under `runs/`, spreadsheet inputs, and the virtual environment. Keep traces in `runs/` even when choosing a custom `--trace` path. Ignore rules do not remove files already committed, and do not protect files uploaded manually through the GitHub website. Do not upload a ZIP of the entire working directory.

Before each push, inspect `git status --short` and `git diff --cached`. Add any deliberate sample fixtures explicitly only after verifying they are synthetic. Do not force-add local configuration, credentials, or real spreadsheets.


## Messy workbook development

The [insurance fixtures](fixtures/README.md) cover narrative sheets, cross-system claim attribute mappings, duplicates, draft mappings, missing values, and leading-zero IDs. Run `make_samples.py` to generate the workbook and CSV under ignored `data/`; tests create independent temporary copies.

`preprocess.py` exports each sheet to CSV plus metadata with types, counts, frequent values, source row positions, and parsing warnings. `local_data.py` uses it automatically, or `run_local.py --preprocessed` loads a previously exported directory. `read_sheet` provides paginated access to narrative content and full table rows; search now includes metadata sheets. `evaluate.py` runs known-answer questions through a selected model and saves results for human review under `runs/`.


## Module names

`health_insurance_document_agent.py` contains the selected-document research loop; `table_agent.py` contains the table graph and existing AWS-compatible table entry points. Update integrations to import table functions from `table_agent`. Function signatures and the S3 preprocessing contract are unchanged.
