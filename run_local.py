"""Run the actual graph on local data, with real Llama inference in Bedrock."""
import argparse
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("question")
    parser.add_argument("--model", choices=["maverick", "scout"])
    parser.add_argument("--profile")
    parser.add_argument("--region")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--inspect", action="store_true", help="Print local metadata without AWS calls")
    parser.add_argument("--trace", type=Path, help="Save full prompts/responses (includes file data)")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--delimiter", default=",")
    args = parser.parse_args()
    for name, value in [("TABLES_MODEL", args.model), ("AWS_PROFILE", args.profile),
                        ("AWS_REGION", args.region)]:
        if value:
            os.environ[name] = value
    logging.basicConfig(level=logging.WARNING)
    from local_data import prepare_local_file
    from llm import LLMCallLog
    from xlsx_csv_agent import run_plan_tables_agent, run_plan_tables_agent_stream
    inputs = prepare_local_file(args.file, encoding=args.encoding, delimiter=args.delimiter)
    if args.inspect:
        for key, value in inputs["s3_client"].objects.items():
            if key.endswith("/_metadata.json"):
                print(json.dumps(json.loads(value), indent=2))
        return
    log = LLMCallLog()
    try:
        if args.stream:
            for event in run_plan_tables_agent_stream(**inputs, query=args.question, call_log=log):
                print(json.dumps(event, default=str), flush=True)
        else:
            print(run_plan_tables_agent(**inputs, query=args.question, call_log=log))
    finally:
        if args.trace:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            args.trace.write_text(json.dumps(log.records, indent=2, default=str), encoding="utf-8")
        print(f"\nLLM calls: {len(log.records)}; input tokens: "
              f"{sum(r['usage'].get('inputTokens', 0) for r in log.records)}; output tokens: "
              f"{sum(r['usage'].get('outputTokens', 0) for r in log.records)}")


if __name__ == "__main__":
    main()
