"""Focused SDK wait deadline boundaries using fake time and MockTransport only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import httpx
import titanos.client as client_module
from titanos.client import Client,TitanosError

assert Path(client_module.__file__).resolve()==ROOT/'src/titanos/client.py'


def terminal_case(status,finished,transient):
    now=[100.0];calls=[];timeouts=[]
    expected={'id':'same-job','status':status}
    def respond(request):
        calls.append((request.method,request.url.path))
        timeouts.append(request.extensions['timeout']['read'])
        if transient and len(calls)==1:
            now[0]=102.0
            raise httpx.ReadError('inert transient status read',request=request)
        now[0]=100.0+finished
        return httpx.Response(200,json=expected)
    observed=None
    with Client('http://inert.invalid',api_key='',timeout=20,transport=httpx.MockTransport(respond)) as client:
        with patch('titanos.client.time.monotonic',side_effect=lambda:now[0]),patch('titanos.client.time.sleep',side_effect=lambda seconds:now.__setitem__(0,now[0]+seconds)):
            try:
                value=client.wait_job('same-job',timeout=10,poll_interval=1,poll_retries=1 if transient else 0)
                observed={'kind':'returned','record':value}
            except TimeoutError as error:
                observed={'kind':'timeout','message':str(error)}
    assert calls==[('GET','/api/jobs/same-job')]*(2 if transient else 1)
    assert timeouts==([10.0,7.0] if transient else [10.0]),'Original deadline was renewed'
    should_timeout=finished>=10
    passed=(observed['kind']=='timeout' and 'it has not been cancelled' in observed['message']) if should_timeout else observed=={'kind':'returned','record':expected}
    return {'status':status,'response_elapsed_seconds':finished,'transient_read_first':transient,
            'expected_timeout':should_timeout,'observed':observed,'passed':passed,
            'calls':calls,'per_get_timeouts':timeouts}


def retry_policy_case(retries):
    now=[100.0];calls=[]
    def fail(request):
        calls.append((request.method,request.url.path));now[0]+=1
        raise httpx.ReadError('inert repeated read failure',request=request)
    with Client('http://inert.invalid',api_key='',timeout=20,transport=httpx.MockTransport(fail)) as client:
        with patch('titanos.client.time.monotonic',side_effect=lambda:now[0]),patch('titanos.client.time.sleep',side_effect=lambda seconds:now.__setitem__(0,now[0]+seconds)):
            try:client.wait_job('same-job',timeout=50,poll_interval=1,poll_retries=retries)
            except TitanosError as error:assert isinstance(error.__cause__,httpx.ReadError)
            else:raise AssertionError('Configured network retry exhaustion was lost')
    assert calls==[('GET','/api/jobs/same-job')]*(retries+1)
    return {'poll_retries':retries,'calls':calls,'passed':True}


def status_rejection_case():
    calls=[]
    def deny(request):
        calls.append((request.method,request.url.path))
        return httpx.Response(401,json={'error':{'message':'inert denied'}})
    with Client('http://inert.invalid',api_key='',timeout=20,transport=httpx.MockTransport(deny)) as client:
        with patch('titanos.client.time.monotonic',return_value=100.0):
            try:client.wait_job('same-job',timeout=10,poll_retries=3)
            except TitanosError as error:assert error.status_code==401
            else:raise AssertionError('Server rejection was swallowed')
    assert calls==[('GET','/api/jobs/same-job')]
    return {'status':401,'calls':calls,'passed':True}


def check(evidence):
    source=ROOT/'src/titanos/client.py'
    initial=hashlib.sha256(source.read_bytes()).hexdigest()
    cases=[terminal_case(status,finished,transient)
           for status in ('completed','failed','cancelled','interrupted')
           for finished in (9.999,10.0,10.25) for transient in (False,True)]
    retries=[retry_policy_case(value) for value in (0,2)]
    rejection=status_rejection_case()
    failed=[case for case in cases if not case['passed']]
    result={'status':'PASSED_FOCUSED_SDK_WAIT_DEADLINE_CHECKS' if not failed else 'FAILED_REPRODUCED_LATE_TERMINAL_ACCEPTANCE',
            'source_path':str(source),'source_sha256':initial,'source_after_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'test_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'terminal_boundary_cases':cases,'retry_policy_cases':retries,'server_rejection':rejection,
            'terminal_cases_passed':len(cases)-len(failed),'terminal_cases_failed':len(failed),
            'scope':'Fake monotonic time and httpx.MockTransport only; GET same job, no cancellation/submission/server/network/model/host/GPU/install/build.'}
    assert result['source_after_sha256']==initial
    if evidence:
        with evidence.open('x',encoding='utf-8') as output:json.dump(result,output,indent=2);output.write('\n')
    print(json.dumps({key:result[key] for key in ('status','source_sha256','test_sha256','terminal_cases_passed','terminal_cases_failed')},indent=2))
    if failed:
        raise AssertionError(f"Accepted {len(failed)} terminal responses at/after the original deadline")
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path)
    check(parser.parse_args().evidence)
