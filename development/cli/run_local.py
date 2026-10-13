"""Run the document agent on local data using native Bedrock tool calls."""
import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from backend.config.settings import MODELS


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("question")
    parser.add_argument("--model", choices=list(MODELS))
    parser.add_argument("--profile")
    parser.add_argument("--region")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--inspect", action="store_true", help="Print local metadata without AWS calls")
    parser.add_argument("--trace", type=Path, help="Save every model call with its prompt and response (includes file data)")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--overrides", type=Path, help="JSON sheet_type/header_row overrides")
    parser.add_argument("--preprocessed", action="store_true", help="Input is an exported preprocessing directory")
    args = parser.parse_args()
    for name, value in [("TABLES_MODEL", args.model), ("AWS_PROFILE", args.profile),
                        ("AWS_REGION", args.region)]:
        if value:
            os.environ[name] = value
    logging.basicConfig(level=logging.WARNING)
    from backend.preprocessing.documents import ingest_document
    from backend.storage.local import exported_document
    from backend.agents.document_agent import run_document_agent_stream
    from backend.entrypoint import ask_question_local
    from development.observability import capture
    capture.enable(content=bool(args.trace))
    overrides = json.loads(args.overrides.read_text()) if args.overrides else None
    if args.preprocessed and overrides:
        parser.error("Apply overrides during preprocessing, not when loading its output.")
    doc = (exported_document(args.file) if args.preprocessed else
           ingest_document(args.file.name, args.file.read_bytes(), table_options=dict(
               encoding=args.encoding, delimiter=args.delimiter, overrides=overrides)))
    if args.inspect:
        print(json.dumps(doc.table_metadata, indent=2))
        return
    documents = {doc.id: doc}
    with capture.recording() as events:
        try:
            if args.stream:
                for event in run_document_agent_stream(documents, [doc.id], args.question, model=args.model):
                    print(json.dumps(event, default=str), flush=True)
            else:
                response = ask_question_local(args.question, documents=[doc], app="cli", model=args.model)
                print(response["answer"] or response["message"])
                for warning in response["limitations"]:
                    print(f"Limitation: {warning}")
                print(f"Status: {response['status']}; reference: {response['request_id']}")
        finally:
            summary = capture.summarize(events)
            if args.trace:
                args.trace.parent.mkdir(parents=True, exist_ok=True)
                args.trace.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
            totals = summary["totals"]
            print(f"\nLLM calls: {totals['llm_calls']}; input tokens: {totals['input_tokens']}; "
                  f"output tokens: {totals['output_tokens']}")


if __name__ == "__main__":
    main()
