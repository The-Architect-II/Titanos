"""Exercise native install from a local HTTP feed and reject corrupt/unsafe archives."""
import argparse
from contextlib import chdir
import functools
import hashlib
import http.server
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
from unittest.mock import patch
import zipfile
from titanos.native_install import install,executable_path
from titanos.hardware import discover


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    args=p.parse_args();host=discover()
    with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'TITANOS_HOME':tmp}):
        root=Path(tmp);source=root/'source';source.mkdir()
        shutil.copyfile(args.archive,source/'runtime.tar.gz')
        artifact={'file':'runtime.tar.gz','sha256':hashlib.sha256(args.archive.read_bytes()).hexdigest()}
        data={'version':1,'runtime':'llama.cpp','release':'test-pinned','profiles':{'test':{'os':host['os'],'arch':host['arch'],'assets':[artifact]}}}
        (source/'native.json').write_text(json.dumps(data))
        class Handler(http.server.SimpleHTTPRequestHandler):
            def log_message(self,*args):pass
        server=http.server.ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(source)))
        threading.Thread(target=server.serve_forever,daemon=True).start()
        try:
            url=f'http://127.0.0.1:{server.server_port}/native.json'
            receipt=install(url,'test',root/'installed')
            assert executable_path()==Path(receipt['executable'])
            assert receipt['native_inference_qualified'] is False
            # A relative on-disk catalog must resolve an asset exactly once.
            with chdir(root):
                local_receipt=install('source/native.json','test',root/'local-installed')
            assert local_receipt['binary_sha256']==receipt['binary_sha256']
            receipt=local_receipt
            artifact['sha256']='0'*64;(source/'native.json').write_text(json.dumps(data))
            try:install(url,'test',root/'bad');raise AssertionError('Corrupt archive accepted')
            except ValueError as exc:assert 'checksum' in str(exc)
            assert not (root/'bad').exists()
            with zipfile.ZipFile(source/'unsafe.zip','w') as z:z.writestr('../escape','invalid')
            artifact.update(file='unsafe.zip',sha256=hashlib.sha256((source/'unsafe.zip').read_bytes()).hexdigest())
            (source/'native.json').write_text(json.dumps(data))
            try:install(url,'test',root/'unsafe');raise AssertionError('Unsafe archive accepted')
            except ValueError as exc:assert 'Unsafe' in str(exc)
            assert not (root/'escape').exists() and not (root/'unsafe').exists()
            assert executable_path()==Path(receipt['executable'])
        finally:server.shutdown();server.server_close()
        result={'success':True,'checks':['HTTP native archive installation and executable receipt','relative local catalog resolves and installs the same native archive','checksum mismatch refused before activation','archive traversal refused; existing runtime receipt preserved'],'receipt':receipt}
        args.report.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'success':True,'checks':result['checks']},indent=2))


if __name__=='__main__':main()
