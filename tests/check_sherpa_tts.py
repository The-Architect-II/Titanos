"""Sherpa adapter contracts with API-shaped doubles; no native runtime/model."""
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from titanos.config import ModelSpec
from titanos.engine import EngineError
from titanos.sherpa_tts import SherpaTTSModel, inspect_model


class Config:
    def __init__(self, **kwargs):self.__dict__.update(kwargs)
    def validate(self):return True


class Runtime:
    def __init__(self, config):self.config=config;self.calls=[]
    def generate(self, **kwargs):
        self.calls.append(kwargs)
        kwargs["callback"](np.zeros(8,dtype=np.float32),.5)
        return NS(samples=np.sin(np.linspace(0,20,1600,dtype=np.float32))*.2,sample_rate=16000)


class Checks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        (self.root/"model.onnx").write_bytes(b"onnx")
        (self.root/"tokens.txt").write_text("a 0\n")
        (self.root/"espeak-ng-data").mkdir()
        self.spec=ModelSpec(path=str(self.root),task="speech",backend="sherpa-onnx",device="cpu")
        self.fake=NS(OfflineTtsVitsModelConfig=Config,OfflineTtsModelConfig=Config,OfflineTtsConfig=Config,OfflineTts=Runtime)
        self.modules=patch.dict(sys.modules,{"sherpa_onnx":self.fake});self.modules.start();self.addCleanup(self.modules.stop)
        self.version=patch("titanos.sherpa_tts.version",return_value="1.13.8");self.version.start();self.addCleanup(self.version.stop)

    def test_component_identity_wav_details_and_cancellation(self):
        layout=inspect_model(self.root);self.assertEqual(layout["model"].name,"model.onnx")
        model=SherpaTTSModel(self.spec)
        wav=model.infer("speech",{"input":"A controlled speech fixture.","voice":"default"})
        self.assertEqual(wav[:4],b"RIFF");self.assertGreater(len(wav),1000)
        details=model.details();self.assertEqual(details["runtime_versions"],{"sherpa-onnx":"1.13.8","sherpa-onnx-core":"1.13.8"})
        self.assertEqual(details["last_generation"]["speaker_id"],0)
        cancel=threading.Event();cancel.set()
        with self.assertRaises(EngineError):model.infer("speech",{"input":"cancel"},cancel=cancel)

    def test_registration_fails_closed(self):
        for change in ({"device":"cuda:0"},{"task":"chat"},{"trust_remote_code":True},
                       {"allow_pickle":True},{"load_options":{"custom":"x"}}):
            with self.assertRaises(ValueError):ModelSpec.model_validate({**self.spec.model_dump(),**change})


if __name__=="__main__":unittest.main(verbosity=2)
