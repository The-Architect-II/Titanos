"""Configured input/output bounds; stub output is validation evidence, not model inference."""
import os
from unittest.mock import patch
import numpy as np
from titanos.compiled import CompiledModel,tensor_payload
from titanos.settings import Settings,_active_settings

def rejected(call):
    try:call()
    except ValueError:return
    raise AssertionError('Expected tensor boundary rejection')

def main():
    body={'dtype':'float32','shape':[3],'data':[1,2,3]}
    with patch.dict(os.environ,{'TITANOS_MAX_TENSOR_ELEMENTS':'2'}):
        rejected(lambda:tensor_payload(body))
    token=_active_settings.set(Settings(max_tensor_elements=3))
    try:
        assert tensor_payload(body).tolist()==[1,2,3]
        rejected(lambda:tensor_payload(dict(body,data=[1,2,float('inf')])))
        rejected(lambda:tensor_payload(dict(body,data=[1,2])))
        model=object.__new__(CompiledModel)
        model.execution={}
        model.run=lambda inputs:{'one':np.ones(2,dtype=np.float32),'two':np.ones(2,dtype=np.float32)}
        rejected(lambda:model.infer('tensor',{'inputs':{}}))
    finally:_active_settings.reset(token)
    token=_active_settings.set(Settings(max_tensor_elements=4))
    try:assert len(model.infer('tensor',{'inputs':{}})['outputs'])==2
    finally:_active_settings.reset(token)
    print('PASS: configurable per-input and aggregate-output element bounds; shape and finite-value validation retained')

if __name__=='__main__':main()
