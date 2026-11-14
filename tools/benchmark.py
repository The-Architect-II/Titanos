"""Bounded HTTP load check; reports observations rather than a performance promise."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import statistics
import time
import httpx


def main():
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:11435');p.add_argument('--model',required=True)
    p.add_argument('--requests',type=int,default=20);p.add_argument('--concurrency',type=int,default=2)
    p.add_argument('--max-tokens',type=int,default=32);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not 1<=args.requests<=100000 or not 1<=args.concurrency<=32 or not 1<=args.max_tokens<=8192:raise ValueError('Invalid load bounds')
    headers={'Authorization':'Bearer '+os.environ['TITANOS_API_KEY']} if os.environ.get('TITANOS_API_KEY') else {}
    with httpx.Client(base_url=args.url,headers=headers,timeout=600,trust_env=False) as client:
        def request(index):
            start=time.monotonic()
            try:
                response=client.post('/v1/chat/completions',json={'model':args.model,'messages':[{'role':'user','content':f'In one sentence explain why test {index} needs a check.'}],'temperature':0,'max_tokens':args.max_tokens})
                response.raise_for_status();data=response.json()
                if not data['choices'][0]['message'].get('content'):raise ValueError('No output content')
                return {'passed':True,'seconds':time.monotonic()-start,'completion_tokens':data['usage']['completion_tokens']}
            except Exception as exc:return {'passed':False,'seconds':time.monotonic()-start,'error':type(exc).__name__}
        started=time.monotonic()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:results=list(pool.map(request,range(args.requests)))
        elapsed=time.monotonic()-started
    latencies=sorted(x['seconds'] for x in results)
    report={'success':all(x['passed'] for x in results),'model':args.model,'requests':args.requests,'concurrency':args.concurrency,
            'seconds':elapsed,'p50_seconds':statistics.median(latencies),'p95_seconds':latencies[min(len(latencies)-1,int(len(latencies)*.95))],
            'output_tokens_per_second':sum(x.get('completion_tokens',0) for x in results)/elapsed,'results':results,
            'scope':'Observed requests on this deployment; not universal throughput or model-quality qualification'}
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
    return 0 if report['success'] else 1


if __name__=='__main__':raise SystemExit(main())
