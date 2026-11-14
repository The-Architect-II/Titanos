"""Binary alpha masks and disclosed RGB copying; no resizing or model logic."""
import base64
import hashlib
import io

from .engine import EngineError

CONTRACT = 'alpha-binary-v1'


def inline_mask(value, limit):
    if value.startswith('data:'):
        prefix = 'data:image/png;base64,'
        if not value.startswith(prefix):
            raise EngineError('mask_base64 requires PNG base64')
        value = value[len(prefix):]
    if len(value) > 4 * ((limit + 2) // 3):
        raise EngineError('Mask exceeds the configured image byte limit', 413, 'request_too_large')
    try:
        return base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise EngineError('Invalid mask base64') from exc


def decode(source_bytes, mask_bytes, settings):
    """Return owned source RGB, L white-edit mask and input metadata."""
    from PIL import Image, ImageOps
    if any(not raw or len(raw) > settings.max_input_image_bytes for raw in (source_bytes, mask_bytes)):
        raise EngineError('Source and mask must be nonempty and within the image byte limit')
    # Pillow can normalize PNG storage modes; check the actual IHDR too.
    if (mask_bytes[:8] != b'\x89PNG\r\n\x1a\n' or mask_bytes[12:16] != b'IHDR'
            or len(mask_bytes) < 33 or mask_bytes[24:26] != bytes((8, 6))):
        raise EngineError('Mask must be an 8-bit RGBA PNG with binary alpha')
    source = mask = None
    try:
        with Image.open(io.BytesIO(source_bytes)) as original, Image.open(io.BytesIO(mask_bytes)) as rgba:
            if any(getattr(im, 'n_frames', 1) != 1 for im in (original, rgba)):
                raise EngineError('Animated source images and masks are unsupported')
            if any(im.width * im.height > settings.max_input_image_pixels for im in (original, rgba)):
                raise EngineError('Source or mask exceeds the input pixel limit')
            if original.size != rgba.size:
                raise EngineError('Source and mask dimensions must match exactly; resizing is not performed')
            if rgba.mode != 'RGBA':
                raise EngineError('Mask must have explicit RGBA channels')
            with original.convert('RGBA') as opaque:
                with opaque.getchannel('A') as alpha:
                    if alpha.getextrema() != (255, 255):
                        raise EngineError('Masked edits require an opaque source image')
            with rgba.getchannel('A') as alpha:
                counts = alpha.histogram()
                if sum(counts[1:255]):
                    raise EngineError('Mask alpha must be exactly 0 (edit) or 255 (keep); no thresholding is performed')
                if counts[0] == 0:
                    raise EngineError('Mask has no editable pixels')
                mask = ImageOps.invert(alpha)
            source = original.convert('RGB')
        return source, mask, {'mask_contract': CONTRACT,
            'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
            'mask_sha256': hashlib.sha256(mask_bytes).hexdigest(),
            'width': source.width, 'height': source.height,
            'editable_pixels': counts[0], 'kept_pixels': counts[255],
            'coordinates': 'decoded_raster_no_orientation_or_resizing',
            'preservation_scope': 'decoded_uint8_RGB_pixels'}
    except BaseException as exc:
        for im in (source, mask):
            if im is not None:
                im.close()
        if isinstance(exc, (ValueError, OSError, Image.DecompressionBombError)) and not isinstance(exc, EngineError):
            raise EngineError('Invalid source image or PNG mask: ' + str(exc)) from exc
        raise


def validate(mask, source, size):
    from PIL import Image
    if (not isinstance(mask, Image.Image) or not isinstance(source, Image.Image)
            or mask.mode != 'L' or source.mode != 'RGB' or mask.size != source.size or source.size != size):
        raise EngineError('Inpainting requires exact source/output dimensions and a canonical L mask')
    counts = mask.histogram()
    if sum(counts[1:255]) or not counts[255]:
        raise EngineError('Inpainting mask must be binary white-edit/black-keep with an editable region')


def finish(image, source, mask, preserve):
    """Mutate an owned generated image, never the source or mask."""
    from PIL import Image, ImageOps
    if not isinstance(image, Image.Image) or image.mode != 'RGB' or image.size != source.size:
        raise EngineError('Inpainting returned invalid RGB output dimensions', 502, 'invalid_backend_response')
    if preserve:
        with ImageOps.invert(mask) as keep:
            image.paste(source, (0, 0), keep)
