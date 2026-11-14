"""Run qualification on explicitly listed SSH hosts; no discovery or credentials stored."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--parallel',type=int,default=1);args=p.parse_args()
    plan=json.loads(args.plan.read_text())
    if not 1<=args.parallel<=8:raise ValueError('parallel must be 1–8')
    hosts=plan['hosts'];names=[x['name'] for x in hosts]
    if not hosts or len(names)!=len(set(names)):raise ValueError('List unique host names')
    args.output.mkdir(parents=True,exist_ok=False)
    def run(target):
        import re
        if not re.fullmatch(r'[A-Za-z0-9._-]+',target['name']):raise ValueError('Unsafe report name')
        if target['host'].startswith('-') or any(c.isspace() for c in target['host']):raise ValueError('Invalid SSH host')
        # The encoded program uses subprocess argument arrays on the remote OS.
        program="import json,os,subprocess,sys; p=json.loads(sys.stdin.read()); os.chdir(os.path.expanduser(p['repository'])); sys.exit(subprocess.call([sys.executable,'tools/qualify.py',*p['args']]))"
        code="import base64;exec(base64.b64decode('"+base64.b64encode(program.encode()).decode()+"'))"
        python=target.get('python','python')
        if target.get('shell','posix')=='posix':command=shlex.join([python,'-c',code])
        elif target['shell']=='cmd':
            if any(c in python for c in '"%&|<>\r\n'):raise ValueError('Invalid Windows Python path')
            command='"'+python+'" -c "'+code+'"'
        else:raise ValueError('Remote shell must be posix or cmd (Windows OpenSSH default)')
        argv=['ssh','-oBatchMode=yes','-oConnectTimeout=10','--',target['host'],command]
        started=time.time();log=args.output/(target['name']+'.log')
        try:
            with log.open('w') as stream:
                completed=subprocess.run(argv,input=json.dumps({'repository':target['repository'],'args':target['args']}),text=True,
                                         stdout=stream,stderr=stream,timeout=target.get('timeout',14400))
            status='passed' if completed.returncode==0 else 'blocked' if completed.returncode==255 else 'failed'
            result={'host':target['host'],'status':status,'exit_code':completed.returncode}
        except subprocess.TimeoutExpired:
            result={'host':target['host'],'status':'failed','reason':'SSH timeout; inspect remote processes before retrying'}
        result.update(name=target['name'],seconds=round(time.time()-started,3),log=log.name)
        return result
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:results=list(pool.map(run,hosts))
    report={'checks':results,'success':all(r['status']=='passed' for r in results)}
    (args.output/'cluster.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return 0 if report['success'] else 1


if __name__=='__main__':sys.exit(main())
