# Excel/CSV agent development harness

This project exercises the supplied `run_plan_tables_agent()` with local Excel/CSV files and real Llama 4 inference through Amazon Bedrock. The existing S3 loading paths and entry-point arguments are preserved. Your original files in Downloads are unchanged.

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
.\.venv\Scripts\python run_local.py data\sales.xlsx "Inspect" --inspect
```

Run the real agent using your existing local AWS profile:

```powershell
.\.venv\Scripts\python run_local.py data\sales.xlsx "What is total revenue by region in Sales?" --profile dev-profile --model maverick --trace runs\maverick.json
.\.venv\Scripts\python run_local.py data\sales.csv "What is total revenue?" --profile dev-profile --model scout --stream
```

Replace the file path and question to test your own files. CSV options include `--encoding cp1252` and `--delimiter ';'`. `.xlsx` and `.xls` files load all worksheets. The local preprocessor detects headers below title rows and preserves free-text metadata sheets. Explicit header/type overrides handle ambiguous layouts. Multiple tables per sheet and uncached formulas are not supported. See fixtures/README.md for details.

Expected sample results: total revenue **500**, West **250**, East **250**. The Excel workbook also has a Targets sheet: West **300**, East **200**. Try searching for West across both sheets or asking for the purpose of each sheet. Cross-sheet joins are not a dedicated tool yet; verify any multi-sheet arithmetic carefully.

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

- `xlsx_csv_agent.py`: original graph plus JSON retry, recursion budget, streaming, literal/numeric search, empty parallel extraction, attribution, and instruction/data separation fixes.
- `llm.py`: Bedrock Converse wrapper with normal SDK profile/role credentials, configurable region, timeouts, and retries.
- `config_utils.py`: missing configuration module, now supplied with Maverick/Scout defaults and optional business context. Retain or reconcile your existing AWS version of this module; the original business context was not supplied.
- `local_data.py`: in-memory `get_object` adapter and metadata generation; no AWS calls.
- `run_local.py`: CLI with optional full LLM traces and token totals. Trace files contain input data and are ignored by Git.
- `make_samples.py`: deterministic sample files.
- `tests/test_agent.py`: offline regression tests using scripted LLM responses and the real graph/data tools. They do not measure Llama's answer quality.

In AWS, keep calling the existing entry point with the real S3 client:

```python
import boto3
from xlsx_csv_agent import run_plan_tables_agent

answer = run_plan_tables_agent(
    s3_client=boto3.client("s3"),
    s3_bucket="your-bucket",
    s3_prefix="your-prefix/",
    plan_domain="your-domain",
    filename="your-workbook.xlsx",
    query="What is total revenue?",
)
```

Use the deployment's IAM role by leaving `AWS_PROFILE` unset. Do not copy the personal `.env` into AWS. The S3 prefix must include its trailing slash. Existing preprocessing must provide `_metadata.json` plus each sheet's CSV at `{prefix}{domain}/preprocessed/{filename}/`; for CSV inputs the object basename is the original CSV filename. The new preprocess.py exports that contract locally; it does not upload to S3 or validate another preprocessing pipeline.

## Limits of this development harness

Generated pandas code still runs through the original in-process `exec`. Restricted builtins and prompt instructions **are not a sandbox**: pandas can access files and networks. Use trusted development files/questions only. Isolate generated code in a separate execution service with resource, filesystem, network, and credential restrictions before accepting untrusted users. This harness does not add that production isolation.

Search outputs can be truncated, and overview extraction sees only five sample rows. Tool failures and the step limit may yield incomplete answers. The tool loop currently uses prompt guidance to discourage repeated calls, not a duplicate-call cache. CLI calls are independent questions; existing Python entry points still accept conversation history.

For model evaluation, run the same questions with both models and compare answers to known totals. Inspect traces for tool selection, generated code, failures, and token use. Offline passing tests are not evidence that a live AWS profile or model invocation works.


## Publishing to GitHub

Commit source code, tests, dependency files, and `.env.example`. The example uses a placeholder profile name; keep your actual settings in `.env` (ignored by Git). AWS login credentials remain in your user-level AWS configuration/cache outside this project; the source code loads them through the SDK and contains no account-specific credentials.

The ignore rules exclude `.env` variants, `.aws` folders, credential/key files, logs, local data, model traces under `runs/`, spreadsheet inputs, and the virtual environment. Keep traces in `runs/` even when choosing a custom `--trace` path. Ignore rules do not remove files already committed, and do not protect files uploaded manually through the GitHub website. Do not upload a ZIP of the entire working directory.

Before each push, inspect `git status --short` and `git diff --cached`. Add any deliberate sample fixtures explicitly only after verifying they are synthetic. Do not force-add local configuration, credentials, or real spreadsheets.


## Messy workbook development

The [synthetic fishing fixtures](fixtures/README.md) cover free-text About sheets, version history, cross-product attribute mappings, duplicates, conflicting units, missing values, and leading-zero IDs. The reviewed workbook and CSV are committed as specific exceptions to the input-file ignore rules.

`preprocess.py` exports each sheet to CSV plus metadata with types, counts, frequent values, source row positions, and parsing warnings. `local_data.py` uses it automatically, or `run_local.py --preprocessed` loads a previously exported directory. `read_sheet` provides paginated access to narrative content and full table rows; search now includes metadata sheets. `evaluate.py` runs known-answer questions through a selected model and saves results for human review under `runs/`.
