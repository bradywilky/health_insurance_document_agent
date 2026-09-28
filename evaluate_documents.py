"""Run the synthetic selected-document evaluation with real Bedrock inference."""
import argparse
import json
from pathlib import Path
import time

from dotenv import load_dotenv
from documents import ingest_document
from health_insurance_document_agent import run_document_agent
from llm import LLMCallLog


def main():
    load_dotenv()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['maverick','scout'],default='maverick')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',action='append')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent/'fixtures'/'documents'
    docs=[ingest_document(p.name,p.read_bytes()) for p in sorted(root.glob('*.txt'))]
    documents={d.id:d for d in docs}
    by_name={d.name:d.id for d in docs}
    cases=json.loads((root/'questions.json').read_text())
    if args.case:
        unknown=set(args.case)-{c['id'] for c in cases}
        if unknown:
            raise ValueError(f'Unknown cases: {sorted(unknown)}')
        cases=[c for c in cases if c['id'] in args.case]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as out:
        for case in cases:
            log=LLMCallLog()
            start=time.monotonic()
            record={**case,'model':args.model}
            try:
                result=run_document_agent(documents,[by_name[n] for n in case['files']],
                    case['question'],call_log=log,model=args.model)
                record.update(result=result,status='needs_human_review')
            except Exception as exc:
                record.update(status='error',error=f'{type(exc).__name__}: {exc}')
            record.update(calls=len(log.records),duration_seconds=round(time.monotonic()-start,2),
                input_tokens=sum(r['usage'].get('inputTokens',0) for r in log.records),
                output_tokens=sum(r['usage'].get('outputTokens',0) for r in log.records))
            out.write(json.dumps(record,default=str)+'\n')
            out.flush()
            print(f"{case['id']}: {record['status']} ({len(log.records)} calls)",flush=True)
            if record['status']=='error':
                raise RuntimeError(f'Evaluation stopped; see {args.output}')


if __name__=='__main__':
    main()
