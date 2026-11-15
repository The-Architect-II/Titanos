"""Metadata-only placement checks; no Torch import, weights or device operations."""
from types import SimpleNamespace as NS
from unittest.mock import patch
import copy,os,sys,threading
from titanos.tensor_storage import describe


class Tensor:
    def __init__(self,device,dtype,count):self.device,self.dtype,self.count=device,dtype,count
    def numel(self):return self.count
    def to(self,*args,**kwargs):raise AssertionError('Metadata inspection moved a tensor')
    def cpu(self):raise AssertionError('Metadata inspection copied a tensor')
    def numpy(self):raise AssertionError('Metadata inspection materialized a tensor')


class Module:
    def __init__(self,parameters,buffers=()):self.p,self.b=parameters,buffers
    def parameters(self):return iter(self.p)
    def buffers(self):return iter(self.b)


class BrokenParameters(Module):
    @property
    def parameters(self):raise RuntimeError('Parameter getter unavailable')


class BrokenHook(Module):
    @property
    def _hf_hook(self):raise RuntimeError('Offload getter unavailable')


def main():
    weight=Tensor('mps:0','torch.float16',12);buffer=Tensor('cpu','torch.int64',3)
    module=Module([weight,weight],[buffer]);result=describe(module)
    assert result['components']['model']['groups']==[
        {'kind':'buffer','device':'cpu','dtype':'torch.int64','tensors':1,'elements':3},
        {'kind':'parameter','device':'mps:0','dtype':'torch.float16','tensors':1,'elements':12}]
    offloaded=Module([Tensor('meta','torch.float32',20)]);offloaded._hf_hook=object()
    pipeline=NS(components={'unet':module,'text_encoder':offloaded,'tokenizer':object()})
    result=describe(pipeline);assert set(result['components'])=={'unet','text_encoder'}
    assert result['components']['text_encoder']['has_offload_hook']
    assert result['components']['text_encoder']['groups'][0]['device']=='meta'
    assert 'not an execution trace' in result['scope']
    assert describe(Module([]))['components']['model']['groups']==[]
    assert describe(object())['components']=={}
    class Broken(Module):
        def parameters(self):raise RuntimeError('Metadata unsupported')
    broken=describe(NS(components={'good':module,'bad':Broken([])}))
    assert broken['complete'] is False and broken['components']['good']['complete']
    assert broken['components']['bad']['error_type']=='RuntimeError'
    class NoComponents:
        @property
        def components(self):raise ValueError('Metadata unavailable')
    assert describe(NoComponents())['complete'] is False
    for broken in [BrokenParameters([]),BrokenHook([])]:
        value=describe(NS(components={'good':module,'bad':broken}))
        assert value['components']['good']['complete'] and not value['components']['bad']['complete']
        assert value['components']['bad']['error_type']=='RuntimeError'
    # Exercise the actual LoadedModel.details integration without importing Torch.
    from titanos.backends import LoadedModel
    loaded=LoadedModel.__new__(LoadedModel);loaded.model=pipeline;loaded.spec=NS(task='image',backend='diffusers',parallelism='single',device='auto')
    loaded.device='mps';loaded.tensor_parallel=loaded.sharded=False;loaded._mps_cogvideo_modules=[]
    with patch.dict(sys.modules,{'torch':NS()}):actual=loaded.details()
    assert actual['tensor_storage']==describe(pipeline) and actual['device']=='mps' and actual['configured_device']=='auto'
    # The actual worker loop must publish a successful result despite either getter failure.
    from titanos.config import Config,ModelSpec,WorkerSpec
    from titanos.replicas import worker_main
    class Connection:
        def __init__(self):self.commands=iter([('infer','m','image',{}),('close',None,None,None)]);self.sent=[]
        def recv(self):return next(self.commands)
        def send(self,value):self.sent.append(copy.deepcopy(value))
        def close(self):pass
    class Engine:
        def __init__(self,cfg):self.current=None
        def infer(self,*args):self.current='m';return {'result':'successful'}
        def details(self,name):return loaded.details()
        def unload(self):self.current=None
        def _runtime_fault(self):return None
    cfg=Config(models={'m':ModelSpec(path='.',task='image',backend='diffusers',device='cpu')})
    assignment={**WorkerSpec(models=['m'],threads=1).model_dump(),'resolved_gpus':[]}  # the parent sends WorkerSpec.model_dump() plus resolved_gpus
    for broken in [BrokenParameters([]),BrokenHook([])]:
        loaded.model=NS(components={'good':module,'bad':broken});connection=Connection()
        with patch.dict(sys.modules,{'torch':NS()}),patch.dict(os.environ,dict(os.environ)),patch('titanos.replicas.Engine',Engine):
            worker_main(connection,threading.Event(),cfg.model_dump(),assignment)
        done=[row[1] for row in connection.sent if row[0]=='done'];assert len(done)==1
        assert done[0][0]=={'result':'successful'} and done[0][1]['completed_requests']==1
        assert done[0][1]['runtime']['tensor_storage']['complete'] is False
        assert not any(row[0]=='error' for row in connection.sent)
    print('PASS parameter/buffer device and dtype grouping, alias deduplication, pipeline components, offload/meta honesty and no tensor moves/materialization; no models/Torch')


if __name__=='__main__':main()
