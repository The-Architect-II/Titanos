"""Actual GenAI adapter/job store, fake native pipeline; no model or device load."""
from pathlib import Path
import importlib.util,json,sys,threading,unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('image_test_helpers',ROOT/'tests/check_genai_images.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
from titanos.genai_images import GenAIImageModel
from titanos.jobs import VideoJobs, _native_generation_observer
from titanos.engine import Engine,EngineError
from titanos.config import Config
from titanos.settings import Settings
import titanos.genai_images as adapter
assert Path(adapter.__file__).resolve().is_relative_to(ROOT/'src')

class Checks(helpers.Checks):
    # Reuse only setup/helpers, never replay the inherited suite.
    def test_native_entry_callback_stop_and_return_are_distinct(self):
        self.export();model=GenAIImageModel(self.spec());cancel=threading.Event()
        def hook(callback):
            self.assertFalse(callback(0,3,None));cancel.set();self.assertTrue(callback(1,3,None))
        self.hook=hook
        with self.assertRaises(EngineError):model.infer('image',{'prompt':'fixture'},cancel=cancel)
        evidence=model.last_generation['native_generation'];call=evidence['native_calls'][0]
        self.assertEqual(call['callback_count'],2);self.assertEqual(call['state'],'returned')
        self.assertEqual(call['first_cancel_callback'],{'step':1,'total':3,'callback_index':2})
        self.assertTrue(call['stop_requested']);self.assertEqual(call['callbacks_after_stop'],0)
        self.assertEqual(evidence['output_disposition'],'discarded_after_native_return')

    def test_preentry_postreturn_and_ignored_stop_are_not_conflated(self):
        self.export();model=GenAIImageModel(self.spec());cancel=threading.Event();cancel.set()
        with self.assertRaises(EngineError):model.infer('image',{'prompt':'fixture'},cancel=cancel)
        before=model.details()['last_generation']['native_generation']
        self.assertEqual(before['native_calls'],[]);self.assertEqual(before['cancellation_observed_at'],'before_native_entry')
        cancel.clear();original=model.text.generate
        def returned(*args,**kwargs):
            result=original(*args,**kwargs);cancel.set();return result
        with patch.object(model.text,'generate',side_effect=returned):
            with self.assertRaises(EngineError):model.infer('image',{'prompt':'fixture'},cancel=cancel)
        after=model.last_generation['native_generation'];self.assertEqual(after['native_calls'][0]['state'],'returned')
        self.assertFalse(after['native_calls'][0]['stop_requested']);self.assertEqual(after['output_disposition'],'discarded_after_native_return')
        cancel.clear()
        def ignores(callback):
            callback(0,3,None);cancel.set();self.assertTrue(callback(1,3,None));self.assertTrue(callback(2,3,None))
        self.hook=ignores
        with self.assertRaises(EngineError):model.infer('image',{'prompt':'fixture'},cancel=cancel)
        self.assertEqual(model.last_generation['native_generation']['native_calls'][0]['callbacks_after_stop'],1)
        self.assertNotEqual(before['request_id'],after['request_id'])

    def test_native_exception_and_observer_failure_preserve_behavior(self):
        self.export();model=GenAIImageModel(self.spec());cancel=threading.Event()
        def observer(value):raise OSError('inert observer unavailable')
        binding=_native_generation_observer.set(('b'*32,observer))
        try:
            with patch.object(adapter.logging.getLogger(adapter.__name__),'warning'):
                images=model.infer('image',{'prompt':'fixture'},cancel=cancel)
                self.assertEqual(len(images),1)
                evidence=model.last_generation['native_generation'];self.assertFalse(evidence['observability_complete'])
                self.assertTrue(evidence['observer_errors']);self.assertTrue(model.last_generation['completed'])
                for image in images:image.close()
                error=RuntimeError('inert native failure')
                def failed(callback):callback(0,3,None);raise error
                self.hook=failed
                with self.assertRaises(RuntimeError) as raised:model.infer('image',{'prompt':'fixture'},cancel=cancel)
                self.assertIs(raised.exception,error)
                call=model.last_generation['native_generation']['native_calls'][0]
                self.assertEqual(call['state'],'raised');self.assertEqual(call['native_error_type'],'RuntimeError');self.assertFalse(call['stop_requested'])
                def stopped_error(callback):cancel.set();callback(1,3,None);raise error
                self.hook=stopped_error
                with self.assertRaises(EngineError):model.infer('image',{'prompt':'fixture'},cancel=cancel)
                call=model.last_generation['native_generation']['native_calls'][0]
                self.assertTrue(call['stop_requested']);self.assertEqual(call['state'],'raised');self.assertEqual(call['native_error_type'],'RuntimeError')
        finally:_native_generation_observer.reset(binding)

    def test_masked_serial_calls_and_snapshot_ids_are_independent(self):
        self.export(edit=True);model=GenAIImageModel(self.spec());image=helpers.Image(helpers.np.zeros((64,64,3),dtype=helpers.np.uint8))
        self.hook=lambda callback:callback(0,3,None)
        with patch('titanos.image_mask.validate',return_value=None):
            images=model.infer('image-edit',{'prompt':'fixture','image':image,'mask_image':image,'n':2,'seed':9})
        first=model.details()['last_generation'];calls=first['native_generation']['native_calls']
        self.assertEqual([value['index'] for value in calls],[0,1]);self.assertEqual([value['callback_count'] for value in calls],[1,1])
        self.assertTrue(all(value['state']=='returned' for value in calls));self.assertEqual(first['rng_seeds'],[9,10])
        for result in images:result.close()
        images=model.infer('image',{'prompt':'fixture'})
        self.assertNotEqual(first['native_generation']['request_id'],model.last_generation['native_generation']['request_id'])
        self.assertEqual(len(first['native_generation']['native_calls']),2)
        for result in images:result.close()

    def test_job_get_observes_foreign_thread_callback_while_engine_locked(self):
        self.export();model=GenAIImageModel(self.spec());engine=Engine(Config(models={'image':self.spec()}))
        engine.current='image';engine.runtime=model;engine.states['image']='loaded'
        store=VideoJobs(self.root/'jobs',Settings(job_workers=1));progress=threading.Event();release=threading.Event();thread_errors=[]
        def hook(callback):
            def on_vendor_thread():
                try:
                    self.assertIsNone(_native_generation_observer.get())
                    self.assertFalse(callback(0,3,None));progress.set();self.assertTrue(release.wait(3));self.assertTrue(callback(1,3,None))
                except BaseException as exc:thread_errors.append(exc);progress.set()
            thread=threading.Thread(target=on_vendor_thread);thread.start();thread.join(4)
            self.assertFalse(thread.is_alive())
            if thread_errors:raise thread_errors[0]
        self.hook=hook
        try:
            job=store.submit('image',lambda cancel:engine.infer('image','image',{'prompt':'fixture'},cancel=cancel),task='image')
            self.assertTrue(progress.wait(3));self.assertTrue(engine.lock.locked())
            live=store.get(job['id']);evidence=live['native_generation'];request_id=evidence['request_id']
            self.assertTrue(live['execution_started'])
            self.assertEqual(evidence['job_id'],job['id']);self.assertEqual(evidence['native_calls'][0]['state'],'entered')
            evidence['native_calls'][0]['callback_count']=999
            self.assertEqual(store.get(job['id'])['native_generation']['native_calls'][0]['callback_count'],1)
            store.cancel(job['id']);self.assertEqual(store.get(job['id'])['status'],'cancelling');release.set()
            store.executor.shutdown(wait=True)
            terminal=store.get(job['id']);self.assertEqual(terminal['status'],'cancelled');self.assertNotIn('result',terminal)
            final=terminal['native_generation'];self.assertEqual(final['request_id'],request_id);self.assertEqual(final['job_id'],job['id'])
            self.assertEqual(final['native_calls'][0]['callback_count'],2);self.assertTrue(final['native_calls'][0]['stop_requested'])
            self.assertEqual(final['job_result_disposition'],'no_result');self.assertIsNone(_native_generation_observer.get())
            self.assertEqual(json.loads((store.root/(job['id']+'.json')).read_text())['native_generation'],final)
        finally:release.set();store.close()

    def test_queued_cancel_has_no_native_evidence_or_generation(self):
        store=VideoJobs(self.root/'queued-jobs');store.start();original=store.executor;original.shutdown(wait=True)
        class Queue:
            def submit(self,fn,*args):self.run=lambda:fn(*args)
            def shutdown(self,wait=True):pass
        queue=store.executor=Queue();calls=[]
        try:
            job=store.submit('image',lambda cancel:calls.append(True),task='image');store.cancel(job['id']);queue.run()
            terminal=store.get(job['id']);self.assertEqual(terminal['status'],'cancelled');self.assertEqual(terminal['error'],'Cancelled')
            self.assertEqual(calls,[]);self.assertNotIn('native_generation',terminal);self.assertIsNone(_native_generation_observer.get())
            self.assertFalse(terminal['execution_started'])
        finally:store.close()

    def test_failed_job_publication_does_not_report_published_output(self):
        self.export();model=GenAIImageModel(self.spec());store=VideoJobs(self.root/'failed-publication')
        original=store.write
        def write(record):
            if record['status']=='completed':raise OSError('inert publication failed')
            return original(record)
        def generate(cancel):
            images=model.infer('image',{'prompt':'fixture'},cancel=cancel)
            for image in images:image.close()
            return {'data':[{'url':'inert-artifact'}]}
        try:
            with patch.object(store,'write',side_effect=write):
                job=store.submit('image',generate,task='image');store.executor.shutdown(wait=True)
            terminal=store.get(job['id']);self.assertEqual(terminal['status'],'failed');self.assertNotIn('result',terminal)
            self.assertIn('publication_error',terminal);self.assertEqual(terminal['native_generation']['job_result_disposition'],'discarded')
        finally:store.close()

    def test_snapshot_copy_failure_is_diagnostic_only(self):
        self.export();model=GenAIImageModel(self.spec());seen=[]
        binding=_native_generation_observer.set(('c'*32,seen.append))
        try:
            with patch.object(adapter.copy,'deepcopy',side_effect=RuntimeError('inert copy failed')),patch.object(adapter.logging.getLogger(adapter.__name__),'warning'):
                images=model.infer('image',{'prompt':'fixture'})
            self.assertEqual(len(images),1);self.assertEqual(seen,[])
            self.assertFalse(model.last_generation['native_generation']['observability_complete'])
            self.assertIn('RuntimeError',model.last_generation['native_generation']['observer_errors'])
            for image in images:image.close()
        finally:_native_generation_observer.reset(binding)

    def test_terminal_snapshot_failure_cannot_skip_job_publication_or_release(self):
        self.export();model=GenAIImageModel(self.spec())
        for fails in (False,True):
            store=VideoJobs(self.root/('terminal-copy-'+str(fails)));finished=[];should_fail=[False]
            original=adapter.copy.deepcopy
            def copied(value,*args,**kwargs):
                if should_fail[0]:raise RuntimeError('inert terminal copy failure')
                return original(value,*args,**kwargs)
            def generate(cancel):
                images=model.infer('image',{'prompt':'fixture'},cancel=cancel)
                for image in images:image.close()
                should_fail[0]=True
                if fails:raise ValueError('original generation failure')
                return {'data':['inert result']}
            try:
                with patch.object(adapter.copy,'deepcopy',side_effect=copied):
                    job=store.submit('image',generate,task='image',cleanup=lambda:finished.append(True));store.executor.shutdown(wait=True)
                terminal=store.get(job['id']);self.assertNotIn(job['id'],store.events);self.assertNotIn('native_generation',terminal)
                self.assertEqual(terminal['native_generation_error']['stage'],'terminal_snapshot')
                self.assertEqual(terminal['status'],'failed' if fails else 'completed')
                if fails:self.assertEqual(terminal['error'],'original generation failure');self.assertEqual(finished,[True])
                else:self.assertEqual(terminal['result'],{'data':['inert result']});self.assertEqual(finished,[])
            finally:store.close()

def suite():
    return unittest.TestSuite(Checks(name) for name in Checks.__dict__ if name.startswith('test_'))
if __name__=='__main__':
    outcome=unittest.TextTestRunner(verbosity=2).run(suite());raise SystemExit(not outcome.wasSuccessful())
