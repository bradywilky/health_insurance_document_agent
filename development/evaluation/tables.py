"""Run synthetic questions through the real agent; save answers for human review."""
import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from backend.config.settings import MODELS


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', type=Path, default=Path('data/insurance_mappings.xlsx'))
    parser.add_argument('--questions', type=Path, default=Path(__file__).resolve().parents[1] / 'fixtures' / 'insurance_questions.json')
    parser.add_argument('--case', action='append', help='Case ID; repeat to select several. Default: all.')
    parser.add_argument('--model', choices=list(MODELS), default='maverick')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--traces', type=Path, help='Save prompts and evidence per case under this local directory')
    args = parser.parse_args()
    os.environ['TABLES_MODEL'] = args.model
    if os.getenv('BEDROCK_MODEL_ID'):
        raise ValueError('Remove BEDROCK_MODEL_ID override before comparing models.')
    from backend.storage.local import prepare_local_file
    from backend.agents.llm import LLMCallLog
    from backend.agents.document_agent import run_document_agent
    from backend.preprocessing.documents import document_from_table_inputs
    questions = json.loads(args.questions.read_text(encoding='utf-8'))
    if args.case:
        unknown = set(args.case) - {q['id'] for q in questions}
        if unknown:
            raise ValueError(f'Unknown cases: {sorted(unknown)}')
        questions = [q for q in questions if q['id'] in args.case]
    doc = document_from_table_inputs(**prepare_local_file(args.file))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects earlier evaluation evidence.
    with args.output.open('x', encoding='utf-8') as output:
        for q in questions:
            log = LLMCallLog()
            start = time.monotonic()
            record = {'case': q['id'], 'model': args.model, 'planner_mode':'native', 'question': q['question'],
                      'expected': q['expected'], 'expected_sheets': q['sheets']}
            try:
                # Expected answers are deliberately never sent to the agent.
                result = run_document_agent({doc.id: doc}, [doc.id], q['question'], call_log=log, model=args.model)
                record.update(answer=result['answer'], evidence=result['evidence'],
                              limitations=result['limitations'], protocol=result['protocol'])
                record['status'] = 'needs_human_review'
            except Exception as exc:
                record.update(status='error', error=f'{type(exc).__name__}: {exc}')
            record.update(duration_seconds=round(time.monotonic()-start, 2), calls=len(log.records),
                          input_tokens=sum(r['usage'].get('inputTokens', 0) for r in log.records),
                          output_tokens=sum(r['usage'].get('outputTokens', 0) for r in log.records))
            record['invoked_model_ids'] = sorted({r['model_id'] for r in log.records})
            if args.traces:
                args.traces.mkdir(parents=True, exist_ok=True)
                with (args.traces / f"{q['id']}.json").open('x', encoding='utf-8') as trace:
                    json.dump(log.records, trace, default=str, indent=2)
            output.write(json.dumps(record, default=str) + '\n')
            output.flush()
            print(f"{q['id']}: {record['status']} ({record['calls']} calls)", flush=True)
            if record['status'] == 'error':
                raise RuntimeError(f"Evaluation stopped after {q['id']}; see {args.output}")


if __name__ == '__main__':
    main()
