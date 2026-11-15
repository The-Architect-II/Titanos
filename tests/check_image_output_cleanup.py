"""All generated outputs close even if one close or an earlier save fails."""
import io
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from fastapi.testclient import TestClient
from titanos.config import Config, ModelSpec
from titanos.server import create_app
from titanos.settings import Settings


def png(mode, color):
    with Image.new(mode, (32, 32), color) as image:
        stream = io.BytesIO(); image.save(stream, format='PNG'); return stream.getvalue()


class Checks(unittest.TestCase):
    def check_case(self, masked, case):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'
            app = create_app(Config(models={'image': ModelSpec(path='unused', backend='diffusers', task='image',
                defaults={'n': 1 if case == 'scalar' else 2})}), output_dir=output, settings=Settings())
            images = [Image.new('RGB', (32, 32), (7, 11, 13)) for _ in range(1 if case == 'scalar' else 2)]
            originals = [image.close for image in images]; closed = []
            for index, image in enumerate(images):
                def close(i=index):
                    closed.append(i); originals[i]()
                    if i == 0 and case != 'scalar': raise OSError('first close failed')
                image.close = close
            if case == 'save':
                def save(*args, **kwargs): raise RuntimeError('primary save failed')
                images[0].save = save
            app.state.engine.target.infer = lambda *args, **kwargs: images[0] if case == 'scalar' else images
            files = {'image': ('source.png', png('RGB', (1, 2, 3)), 'image/png')}
            if masked: files['mask'] = ('mask.png', png('RGBA', (9, 8, 7, 0)), 'image/png')
            try:
                with TestClient(app, base_url='http://127.0.0.1', raise_server_exceptions=case == 'save') as client:
                    if case == 'save':
                        with self.assertRaisesRegex(RuntimeError, 'primary save failed') as error:
                            client.post('/v1/images/edits', data={'model': 'image', 'prompt': 'blue', 'response_format': 'url'}, files=files)
                        self.assertTrue(any('first close failed' in note for note in error.exception.__notes__))
                    else:
                        response = client.post('/v1/images/edits', data={'model': 'image', 'prompt': 'blue', 'response_format': 'url'}, files=files)
                        self.assertEqual(response.status_code, 502 if case == 'scalar' else 500)
                self.assertEqual(closed, list(range(len(images))))
                self.assertFalse(list(output.glob('*.png')))
            finally:
                for close in originals: close()

    def test_masked_multiple_outputs_close(self): self.check_case(True, 'close')
    def test_unmasked_multiple_outputs_close(self): self.check_case(False, 'close')
    def test_primary_save_error_survives_cleanup_error(self): self.check_case(True, 'save')
    def test_malformed_scalar_output_is_closed(self): self.check_case(True, 'scalar')


if __name__ == '__main__': unittest.main(verbosity=2)
