"""Hub contracts without credentials or large network downloads: python tests/check_hub.py."""
import contextlib
import io
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from titanos import hub
from titanos.cli import main
from titanos.config import read_config


def check():
    info = SimpleNamespace(id='owner/model', pipeline_tag='text-generation', library_name='transformers',
                           downloads=1, likes=0, gated=False, sha='a'*40,
                           siblings=[SimpleNamespace(rfilename='model.safetensors')])
    with tempfile.TemporaryDirectory() as tmp, patch.object(hub, 'HfApi') as api, patch.object(hub, 'snapshot_download') as download, patch.object(hub, 'get_token', return_value=None):
        root = Path(tmp)
        weights = root / 'weights'
        weights.mkdir()
        download.return_value = str(weights)
        api.return_value.model_info.return_value = info
        cfg = root / 'models.json'
        prefix = ['--config', str(cfg)]
        command = prefix + ['pull', 'assistant', 'owner/model', '--task', 'chat']
        with contextlib.redirect_stdout(io.StringIO()):
            main(command)
        assert api.call_args.kwargs['token'] is False
        assert download.call_args.kwargs['revision'] == 'a'*40
        saved = cfg.read_bytes()
        assert read_config(cfg).models['assistant'].source_revision == 'a'*40
        with patch.dict(os.environ, {'TEST_HUB_TOKEN': 'fake-secret'}):
            with contextlib.redirect_stdout(io.StringIO()):
                main(command + ['--replace', '--token-env', 'TEST_HUB_TOKEN'])
            assert download.call_args.kwargs['token'] == 'fake-secret'
            assert b'fake-secret' not in cfg.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                main(command + ['--replace', '--anonymous'])
            assert download.call_args.kwargs['token'] is False
        download.side_effect = OSError('interrupted download')
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                main(command + ['--replace'])
                raise AssertionError('failed pull succeeded')
            except SystemExit as exc:
                assert exc.code == 1
        assert cfg.read_bytes() == saved
        api.return_value.list_models.return_value = iter([info, info])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(['search', '--all', '--anonymous'])
        assert api.return_value.list_models.call_args.kwargs['limit'] is None
        assert len(output.getvalue().splitlines()) == 2
        assert json.loads(output.getvalue().splitlines()[0])['titanos_compatibility'] == 'unverified'
        # Repointing an alias preserves the executable and is picked up by a new registry read.
        other = root / 'other'
        other.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            main(prefix + ['add', 'assistant', str(other), '--task', 'chat', '--replace'])
        spec = read_config(cfg).models['assistant']
        assert Path(spec.path).samefile(other) and spec.source_repo is None
        info.siblings = [SimpleNamespace(rfilename='weights.gguf')]
        download.reset_mock()
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                main(command + ['--replace'])
                raise AssertionError('unsupported-only repository accepted')
            except SystemExit as exc:
                assert exc.code == 1
        download.assert_not_called()
        # Diffusion exports have component graphs, not the LLM root graph.
        required = {'model_index.json', 'scheduler/scheduler_config.json',
                    'tokenizer/openvino_tokenizer.xml', 'tokenizer/openvino_tokenizer.bin'}
        required.update(f'{component}/{name}' for component in ('text_encoder', 'unet', 'vae_decoder')
                        for name in ('config.json', 'openvino_model.xml', 'openvino_model.bin'))
        info.siblings = [SimpleNamespace(rfilename=name) for name in required | {'safety_checker/model.safetensors'}]
        download.side_effect = None
        image_command = prefix + ['pull', 'picture', 'owner/model', '--task', 'image', '--backend', 'openvino-genai', '--anonymous']
        with contextlib.redirect_stdout(io.StringIO()): main(image_command)
        assert download.call_args.kwargs['revision'] == info.sha
        # Whole-repository backends now name every sibling explicitly (literal patterns) instead of passing no filter.
        assert download.call_args.kwargs['allow_patterns'] == sorted(item.rfilename for item in info.siblings)
        assert 'ignore_patterns' not in download.call_args.kwargs
        assert read_config(cfg).models['picture'].backend == 'openvino-genai'
        saved_image = cfg.read_bytes()
        for extra, names in [(['--graph', 'unet/openvino_model.xml'], required),
                             ([], required - {'unet/openvino_model.bin'}),
                             ([], {'model_index.json', 'unet/diffusion_pytorch_model.safetensors'})]:
            info.siblings = [SimpleNamespace(rfilename=name) for name in names]
            download.reset_mock()
            with contextlib.redirect_stderr(io.StringIO()):
                try: main(image_command + ['--replace'] + extra)
                except SystemExit as exc: assert exc.code == 1
                else: raise AssertionError('Incomplete/unprepared image export accepted')
            download.assert_not_called()
            assert cfg.read_bytes() == saved_image
    print('PASS: anonymous/authenticated pulls, pinned revision, no token persistence, failed-pull preservation, streaming catalog, directory swap, unsupported format rejection')


if __name__ == '__main__':
    check()
