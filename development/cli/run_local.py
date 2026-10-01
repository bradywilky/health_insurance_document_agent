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
    parser.add_argument("--trace", type=Path, help="Save full prompts/responses (includes file data)")
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
    from backend.storage.local import prepare_local_file, prepare_preprocessed_directory
    from backend.agents.llm import LLMCallLog
    from backend.agents.document_agent import run_document_agent_stream
    from backend.observability.tracing import configure_tracing
    from backend.services.questions import answer_question
    configure_tracing()
    from backend.preprocessing.documents import document_from_table_inputs
    overrides = json.loads(args.overrides.read_text()) if args.overrides else None
    if args.preprocessed and overrides:
        parser.error("Apply overrides during preprocessing, not when loading its output.")
    inputs = (prepare_preprocessed_directory(args.file) if args.preprocessed else
              prepare_local_file(args.file, encoding=args.encoding, delimiter=args.delimiter, overrides=overrides))
    if args.inspect:
        for key, value in inputs["s3_client"].objects.items():
            if key.endswith("/_metadata.json"):
                print(json.dumps(json.loads(value), indent=2))
        return
    doc = document_from_table_inputs(**inputs)
    documents = {doc.id: doc}
    log = LLMCallLog()
    try:
        if args.stream:
            for event in run_document_agent_stream(documents, [doc.id], args.question, call_log=log, model=args.model):
                print(json.dumps(event, default=str), flush=True)
        else:
            result = answer_question(documents, [doc.id], args.question, app="cli",
                                     call_log=log, model=args.model)
            print(result["answer"])
            print(f"Reference: {result['request_id']}")
            for warning in result["limitations"]:
                print(f"Limitation: {warning}")
    finally:
        if args.trace:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            args.trace.write_text(json.dumps(log.records, indent=2, default=str), encoding="utf-8")
        print(f"\nLLM calls: {len(log.records)}; input tokens: "
              f"{sum(r['usage'].get('inputTokens', 0) for r in log.records)}; output tokens: "
              f"{sum(r['usage'].get('outputTokens', 0) for r in log.records)}")


if __name__ == "__main__":
    main()
