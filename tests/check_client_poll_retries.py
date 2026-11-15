"""A lost status connection never causes duplicate job submission."""
from unittest.mock import patch
import httpx
from titanos.client import Client, TitanosError

def check():
    calls=[]
    def recover(request):
        calls.append((request.method,request.url.path))
        if len(calls)==1:raise httpx.RemoteProtocolError('keep-alive closed',request=request)
        return httpx.Response(200,json={'id':'same-job','status':'completed'})
    with Client(transport=httpx.MockTransport(recover)) as client, patch('titanos.client.time.sleep'):
        assert client.wait_job('same-job',timeout=2,poll_retries=1)['status']=='completed'
    assert calls==[('GET','/api/jobs/same-job')]*2
    calls.clear()
    def fail(request):
        calls.append(request.method)
        raise httpx.ReadError('connection lost',request=request)
    with Client(transport=httpx.MockTransport(fail)) as client, patch('titanos.client.time.sleep'):
        try:client.wait_job('same-job',timeout=2,poll_retries=2)
        except TitanosError:pass
        else:raise AssertionError('Retry bound not enforced')
        assert calls==['GET']*3
        calls.clear()
        try:client.submit_job(model='fixture',task='image')
        except TitanosError:pass
        else:raise AssertionError('Submission unexpectedly succeeded')
        assert calls==['POST']
    calls.clear()
    def rejected(request):
        calls.append(request.method)
        return httpx.Response(401,json={'error':{'message':'Denied'}})
    with Client(transport=httpx.MockTransport(rejected)) as client, patch('titanos.client.time.sleep'):
        try:client.wait_job('same-job',timeout=2)
        except TitanosError as exc:assert exc.status_code==401
        else:raise AssertionError('Server rejection was ignored')
    assert calls==['GET']
    print('PASS: bounded GET recovery, same job identity, no POST retry, server errors preserved (transport fixtures)')

if __name__=='__main__':check()
