"""Measure the ambiguity step: how often it asks on ambiguous versus clear questions, and what it offers.

Questions: [{"id", "files": [paths relative to --root], "question", "ambiguous": bool, "note"}].
Answers are saved for human review; expected labels are never sent to the model.
"""
import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from backend.agents.document_agent import run_document_agent
from backend.config.settings import MODELS
from backend.preprocessing.documents import ingest_document
from development.observability import capture


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'fixtures' / 'ambiguity_questions.json')
    parser.add_argument('--root', type=Path, help='Folder the question file paths are relative to '
                                                  '(default: the questions file folder)')
    parser.add_argument('--model', choices=list(MODELS), default='maverick')
    parser.add_argument('--mode', choices=['ask', 'assumptions'], default='ask')
    parser.add_argument('--case', action='append')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    capture.enable(content=False)
    if os.getenv('BEDROCK_MODEL_ID'):
        raise ValueError('Remove BEDROCK_MODEL_ID override before comparing models.')
    root = (args.root or args.questions.parent).resolve()
    cases = json.loads(args.questions.read_text(encoding='utf-8'))
    if args.case:
        cases = [c for c in cases if c['id'] in args.case]
    loaded = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as out:
        for case in cases:
            for name in case['files']:
                if name not in loaded:
                    path = (root / name).resolve()
                    loaded[name] = ingest_document(path.name, path.read_bytes())
            docs = {loaded[n].id: loaded[n] for n in case['files']}
            start = time.monotonic()
            record = {'id': case['id'], 'model': args.model, 'mode': args.mode, 'question': case['question'],
                      'labeled_ambiguous': case['ambiguous'], 'note': case.get('note', '')}
            try:
                with capture.recording() as events:
                    result = run_document_agent(docs, list(docs), case['question'], model=args.model,
                                                ambiguity=args.mode)
                record.update(decision=result['protocol'].get('ambiguity_decision'), status=result['status'],
                              assessment=result['protocol'].get('ambiguity_assessment'),
                              clarification=result.get('clarification'), interpretation=result.get('interpretation'),
                              answer=result['answer'])
            except Exception as exc:
                record.update(status='error', error=f'{type(exc).__name__}: {exc}')
            record.update(seconds=round(time.monotonic() - start, 2), calls=len([e for e in events if capture.kind(e) == 'llm']))
            out.write(json.dumps(record, default=str) + '\n')
            out.flush()
            print(f"{case['id']}: {record.get('decision', record['status'])}", flush=True)


if __name__ == '__main__':
    main()
