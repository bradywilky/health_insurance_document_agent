# Trace spans

Every span the application records, and every field each one carries. The same spans go to Datadog LLM
Observability (when enabled) and to the trace archive (`archive.py`). The helpers are in `tracing.py`; see the
README's **Audit and tracing** section for setup. Update this page when a span or annotation changes.

Fields marked 🔒 are recorded only with `TRACE_CONTENT=true`, because they can contain PHI. Everything else is
always recorded, so it must never hold question, answer or document text.

## Span trees

One trace per question (`backend/entrypoint.py`, `_run`):

```text
agent  document_agent
├── workflow  init
├── workflow  assess                         always; returns at once when ambiguity is 'off'
│   └── llm  Ambiguity Assessment            ambiguity 'assumptions' or 'ask', no clarification
├── workflow  reason                   ┐     only with PLANNER_REASONING (backend/agents/reasoning.py)
│   └── llm  Planner Reasoning         │
├── workflow  plan                     │
│   └── llm  Native Document Research  │     repeated once per research step
├── workflow  tools                    │
│   └── tool  <tool name> × N          ┘
│       └── llm  Relevance Scan × N          scan tool only: one per batch of 20 text blocks, run in parallel
└── workflow  synthesize
    ├── llm  Document Answer
    └── llm  Document Answer Revision        when the answer check finds issues
```

One trace per document change, each archived when it ends:

```text
workflow  ingest_document                    backend/preprocessing/documents.py, ingest_document
├── llm  Document Enrichment                 when a semantic profile is requested
└── llm  Search Hints × N                    when search hints are requested: one per 15 text blocks, in parallel
workflow  save_profile                       backend/storage/documents.py, save_profile
workflow  save_search_hints                  backend/storage/documents.py, save_search_hints
workflow  delete_document                    backend/storage/documents.py, delete
```

Parents are not set explicitly: ddtrace makes each new span a child of the span open when it starts, so the tree
follows the nesting of `with tracing.span(...)` blocks at run time.

## `agent` · `document_agent`

The root span of each question. Started in `entrypoint._run`; its trace ID is returned in the response, and the
finished trace is archived.

| Field | Type | Value |
| --- | --- | --- |
| `session_id` | span option | Conversation ID; Datadog groups traces by session |
| `request_id` | metadata | Returned in the response |
| `ambiguity_mode` | metadata | `off`, `assumptions` or `ask` |
| `model` | metadata | Model key; omitted when the default is used |
| `group_name` | metadata | Group the documents were read from; `ask_question` only |
| `code_version`, `prompt_version` | metadata | `APP_VERSION` or a source hash; hash of every instruction and tool definition |
| `question_sha256`, `question_chars` | metadata | Hash and length of the question as received |
| `clarified` | metadata | Whether the user chose a reading after `needs_clarification` |
| `documents` | metadata | Added after the documents load: ID, name, kind, manifest key, has profile, warning count |
| `status` | metadata | Final status (see `entrypoint.py`) |
| `error` | metadata | Exception type and message, or the request error; absent on success |
| `totals` | metadata | Model calls, input and output tokens for the question |
| `answer_sha256` | metadata | Hash of the answer, when there is one |
| `protocol` | metadata | Research statistics (steps, retries, prompts); excludes the ambiguity assessment, which quotes the question |
| `coverage`, `limitations_count` | metadata | Files examined, auto-read and not examined; number of limitations |
| `evidence` | metadata | Per evidence item: tool, document, operation, error flag, matched rows, page/sheet/row locations; no text |
| `app_id` | tag | Which application called: the Lambda event's `args.app_id` (default `lambda`), `chat` (chat app), `cli`; `api` and `local` are the entry-point defaults |
| `user_id` | tag | Who asked, from the caller (the Lambda event's `user_id`); omitted when not given |
| `status` | tag | Final status |
| input 🔒 | content | `{question, clarification}` |
| output 🔒 | content | Answer text |
| error | status | Set when the status is not `answered`, `needs_clarification` or `no_evidence` |

## `workflow` · `init`, `assess`, `reason`, `plan`, `tools`, `synthesize`

One span per LangGraph node execution, from `tracing.traced_node` (applied in
`document_agent.build_document_graph`).

| Field | Type | Value |
| --- | --- | --- |
| `step` | metadata | Research step count when the node starts; absent on `init` |
| `finished` | metadata | Whether research is done; reported by `tools` only |
| `returned_result` | metadata | True when the node produced the final result (`synthesize`, or `assess` asking for clarification) |
| error | status | Set when the node raises |

## `llm` · one per Bedrock call

Started in `LLM.converse` (`backend/agents/llm.py`). The span name is the `LLM`'s `tool_name`: `Ambiguity
Assessment`, `Planner Reasoning`, `Native Document Research`, `Document Answer`, `Document Answer Revision`, `Relevance Scan`, `Document Enrichment` or `Search Hints`.

| Field | Type | Value |
| --- | --- | --- |
| `model_name` | span option | Bedrock model ID |
| `model_provider` | span option | `amazon_bedrock` |
| `agent` | metadata | `health_insurance_document_agent` |
| `stop_reason` | metadata | Bedrock stop reason (`end_turn`, `tool_use`, `max_tokens`, ...) |
| `max_tokens` | metadata | Configured output limit |
| `tools_offered` | metadata | Number of tool definitions sent |
| `tool_calls` | metadata | Names of the tools the model requested |
| `input_tokens`, `output_tokens`, `total_tokens` | metrics | Datadog's reserved token metrics; used for cost estimates |
| input 🔒 | content | System prompt and messages, as chat messages (`tracing.bedrock_messages`) |
| output 🔒 | content | Model reply, including requested tool calls |
| error | status | Set when the Bedrock call raises (throttling, timeout, ...) |

## `tool` · one per tool execution

Started in `document_agent.node_tools`. The span name is the tool name: `list_documents` (several documents only), `search`,
`read_document`, `table_tool`, `calculate` or `date_calculate` (tool definitions in
`backend/agents/document_protocol.py`). Table operations such as `get_sheet_schema` are not separate tools; they
are recorded in `operation`.

| Field | Type | Value |
| --- | --- | --- |
| `evidence_id` | metadata | `E1`, `E2`, ...; matches the citations in the answer |
| `operation` | metadata | `table_tool` only: `list_sheets`, `get_sheet_schema`, `read_sheet`, `search_all_sheets`, `query_table` or `join_tables` |
| `document_id`, `document_ids` | metadata | Document(s) the tool targeted |
| `matches` | metadata | Search matches returned |
| `blocks` | metadata | Blocks read |
| `matched_rows` | metadata | Table rows matched |
| input 🔒 | content | Tool parameters |
| output 🔒 | content | Tool result (the evidence) |
| error | status | Set for invalid calculation inputs, an `error` in the tool result, or an exception |

## `workflow` · `ingest_document`

The root span of each upload, started by `ingest_document`. A failure (unsupported type, unreadable file, failed
save) marks it as an error with the exception message.

| Field | Type | Value |
| --- | --- | --- |
| `filename`, `kind` | metadata | Uploaded filename and extension |
| `bytes`, `content_sha256` | metadata | File size and hash |
| `profile_requested` | metadata | Whether a semantic profile was requested |
| `saved`, `group_name` | metadata | Whether a store was given, and its group |
| `document` | metadata | The resulting document: ID, name, kind, manifest key, has profile, warning count |
| `profile_generated`, `profile_model_id` | metadata | Whether a profile was produced, and by which model |

## `workflow` · `save_profile`

Started by `ObjectDocumentStore.save_profile` for a generated or reviewed profile.

| Field | Type | Value |
| --- | --- | --- |
| `manifest_key`, `group_name` | metadata | The document and its group |
| `reviewed`, `profile_model_id` | metadata | Whether a person reviewed it; the model that generated it |
| `glossary_terms` | metadata | Number of glossary terms |
| `terms_added`, `terms_removed`, `terms_changed`, `terms_confirmed` | metadata | Glossary terms compared with the previous profile (terms only, no meanings); confirmed count |

## `workflow` · `delete_document`

Started by `ObjectDocumentStore.delete`.

| Field | Type | Value |
| --- | --- | --- |
| `filename`, `group_name` | metadata | The filename or manifest key given, and the group |

## Example trace

A question asked in the chat app with ambiguity `off`, one spreadsheet selected: "give me some info on this file".
The run made 5 model calls (34,990 input and 481 output tokens, 4.8 s in the model of 5.4 s total; the largest,
call #4, was 10,261 tokens) and produced evidence E1 to E3. Values marked `~` and the order of the tool calls are a
reconstruction consistent with those totals, not recorded values.

```text
agent  document_agent                                                  5.4 s
│  metadata  request_id: 3e00087c7927…   ambiguity_mode: off   documents: [{id: 802f2eedc04f707a, …}]
│            status: answered            (model omitted: default used)
│  tags      app_id: chat   status: answered
│  🔒 input   {question: "give me some info on this file", clarification: null}
│  🔒 output  "The file \"Northstar_Commerce_…xlsx\" is a spreadsheet document with…"
│
├── workflow  init                                                     ~5 ms
├── workflow  assess                                                   <1 ms   ambiguity off: returns {}
│
├── workflow  plan            step: 0                                  ~0.7 s
│   └── llm  Native Document Research          #1
│          model_name: <model ID>   model_provider: amazon_bedrock
│          metadata  agent: health_insurance_document_agent   stop_reason: tool_use
│                    tools_offered: 7   tool_calls: [list_documents]
│          metrics   input_tokens: ~5,300   output_tokens: ~30
│          🔒 input   [system: "You answer questions using ONLY…", user: {question, selected_documents, …}]
│          🔒 output  [assistant: tool call list_documents({})]
├── workflow  tools           step: 1   finished: false
│   └── tool  list_documents
│          metadata  evidence_id: E1                                   (no document fields: none apply)
│
├── workflow  plan            step: 1
│   └── llm  Native Document Research          #2   tool_calls: [read_document]   input_tokens: ~8,250
├── workflow  tools           step: 2   finished: false
│   └── tool  read_document
│          metadata  evidence_id: E2   document_id: 802f2eedc04f707a   blocks: 8
│          🔒 input   {document_id: "802f2eedc04f707a", offset: 0, limit: 8}
│
├── workflow  plan            step: 2
│   └── llm  Native Document Research          #3   tool_calls: [table_tool]      input_tokens: ~8,900
├── workflow  tools           step: 3   finished: false
│   └── tool  table_tool
│          metadata  evidence_id: E3   operation: list_sheets   document_id: 802f2eedc04f707a
│
├── workflow  plan            step: 3
│   └── llm  Native Document Research          #4   tool_calls: [answer]
│          metrics   input_tokens: 10,246   output_tokens: 15   total_tokens: 10,261
├── workflow  tools           step: 4   finished: true                 no tool span: answer is not executed
│
└── workflow  synthesize      step: 4   returned_result: true          ~2.1 s
    └── llm  Document Answer                   #5
           metadata  stop_reason: end_turn   tools_offered: 0   tool_calls: []
           metrics   input_tokens: ~2,294   output_tokens: ~361
           🔒 input   [system: "Write a concise answer using ONLY the evidence…", user: {question, evidence, …}]
           🔒 output  [assistant: answer text]
```

What to read from it:

- Each research step is a `plan` and `tools` pair; the number of `plan` spans is the number of planner turns.
- Planner input tokens grow with each call because every tool result is sent back on the next call, so the call
  that only says `answer` is usually the largest.
- The writer call has few input tokens but most of the output tokens and the longest duration.
- With `TRACE_CONTENT=false` the 🔒 lines are absent and everything else is unchanged. The chat app's local capture
  always records them.

## On every span

Span ID, parent ID (`undefined` on the root), trace ID, start time, duration and status, plus `session_id` on every
span of a question. In Datadog, also `env`, `service` and `version` tags from `DD_ENV`, `DD_SERVICE` and
`DD_VERSION`. Metadata, metric and tag values of `None` are dropped.

## Not traced as separate spans

- In `node_tools`: the `answer` action, duplicate requests and coverage prompts (planner bookkeeping, not tool
  executions).
- Automatic reads when research finishes (`read_document` calls with `auto: True`). They appear in the response
  coverage and the root span's `evidence`, not as tool spans.
- Loading documents from storage in `_run`. Its time is part of the `agent` span.
