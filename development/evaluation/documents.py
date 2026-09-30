"""Run the synthetic selected-document evaluation with real Bedrock inference."""
import argparse
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
from backend.preprocessing.documents import ingest_document
from backend.agents.document_agent import run_document_agent
from backend.agents.llm import LLMCallLog
from backend.config.settings import MODELS


def main():
    load_dotenv()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=list(MODELS),default='maverick')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',action='append')
    parser.add_argument('--traces', type=Path, help='Save full local model-call traces')
    parser.add_argument('--fixtures', type=Path,
                        help='Generated fixture directory containing questions.json and upload files')
    args=parser.parse_args()
    if os.getenv('BEDROCK_MODEL_ID'):
        raise ValueError('Remove BEDROCK_MODEL_ID override before comparing models.')
    root=args.fixtures or Path(__file__).resolve().parents[1]/'fixtures'/'documents'
    cases=json.loads((root/'questions.json').read_text())
    if args.case:
        unknown=set(args.case)-{c['id'] for c in cases}
        if unknown:
            raise ValueError(f'Unknown cases: {sorted(unknown)}')
        cases=[c for c in cases if c['id'] in args.case]
    # Only load selected case inputs; rejection fixtures must not abort Q&A evaluation.
    names=sorted({name for case in cases for name in case['files']})
    docs=[]
    for name in names:
        path=(root/name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Fixture path escapes the fixture directory')
        docs.append(ingest_document(path.name,path.read_bytes()))
    documents={d.id:d for d in docs}
    by_name={d.name:d.id for d in docs}
    if args.traces:
        args.traces.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as out:
        for case in cases:
            log=LLMCallLog()
            start=time.monotonic()
            record={**case,'model':args.model,'planner_mode':'native'}
            try:
                result=run_document_agent(documents,[by_name[n] for n in case['files']],
                    case['question'],call_log=log,model=args.model)
                record.update(result=result,status='needs_human_review')
            except Exception as exc:
                record.update(status='error',error=f'{type(exc).__name__}: {exc}')
            record.update(calls=len(log.records),duration_seconds=round(time.monotonic()-start,2),
                input_tokens=sum(r['usage'].get('inputTokens',0) for r in log.records),
                output_tokens=sum(r['usage'].get('outputTokens',0) for r in log.records))
            record['invoked_model_ids'] = sorted({r['model_id'] for r in log.records})
            if args.traces:
                trace_name = case['id']
                if Path(trace_name).name != trace_name or '/' in trace_name or '\\' in trace_name:
                    raise ValueError('Case ID must be a basename')
                with (args.traces / (trace_name + '.json')).open('x', encoding='utf-8') as trace:
                    json.dump(log.records, trace, default=str, indent=2)
            out.write(json.dumps(record,default=str)+'\n')
            out.flush()
            print(f"{case['id']}: {record['status']} ({len(log.records)} calls)",flush=True)
            if record['status']=='error':
                raise RuntimeError(f'Evaluation stopped; see {args.output}')


if __name__=='__main__':
    main()
