"""Local relative Depth Anything inference and lossless image-aligned numeric output."""
import base64
import hashlib
import math
import struct
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Annotated, Literal

from .engine import EngineError
from .settings import current_settings


class DepthResult(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    model: str = Field(min_length=1, max_length=128)
    task: Literal['depth-estimation'] = 'depth-estimation'
    semantics: Literal['relative_inverse_depth'] = 'relative_inverse_depth'
    units: None = None
    larger_values: Literal['nearer'] = 'nearer'
    calibrated: Literal[False] = False
    width: int = Field(strict=True, gt=0)
    height: int = Field(strict=True, gt=0)
    encoding: Literal['base64_float32_le'] = 'base64_float32_le'
    layout: Literal['row_major_yx'] = 'row_major_yx'
    data: str
    value_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    input_rgb_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    model_output_shape: list[Annotated[int, Field(strict=True, gt=0)]] = Field(min_length=2, max_length=2)
    alignment: Literal['processor_post_process_to_input_size'] = 'processor_post_process_to_input_size'
    normalization: Literal['none'] = 'none'
    model_class: str
    processor_class: str
    device: str

    @model_validator(mode='after')
    def validate_values(self):
        check_dimensions(self.width, self.height)
        expected = self.width * self.height * 4
        if len(self.data) != 4 * ((expected + 2) // 3):
            raise ValueError('Depth byte count does not match the declared image dimensions')
        raw = base64.b64decode(self.data, validate=True)
        if len(raw) != expected or hashlib.sha256(raw).hexdigest() != self.value_sha256:
            raise ValueError('Depth value bytes do not match their declared shape or identity')
        if any(not math.isfinite(value) for value, in struct.iter_unpack('<f', raw)):
            raise ValueError('Depth output must contain finite float32 values')
        if any(type(value) is not int or value < 1 for value in self.model_output_shape):
            raise ValueError('Depth model output shape must contain two positive integers')
        if len(self.model_dump_json().encode('utf-8')) > current_settings().max_body_bytes:
            raise ValueError('Depth response exceeds TITANOS_MAX_BODY_BYTES')
        return self


def check_dimensions(width, height):
    settings = current_settings()
    if type(width) is not int or type(height) is not int or width < 1 or height < 1:
        raise ValueError('Depth input requires positive integer dimensions')
    if width * height > min(settings.max_input_image_pixels, settings.max_tensor_elements):
        raise ValueError('Depth map exceeds TITANOS_MAX_INPUT_IMAGE_PIXELS or TITANOS_MAX_TENSOR_ELEMENTS')
    if 4 * ((width * height * 4 + 2) // 3) > settings.max_body_bytes:
        raise ValueError('Encoded depth values exceed TITANOS_MAX_BODY_BYTES')


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise EngineError('Depth inference cancelled', 499, 'cancelled')


class DepthModel:
    def __init__(self, spec, device, load_kwargs):
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForDepthEstimation
        self.spec, self.device = spec, device
        config_bytes = (Path(spec.path) / 'config.json').read_bytes()
        config = AutoConfig.from_pretrained(spec.path, local_files_only=True, trust_remote_code=False)
        if config.model_type != 'depth_anything' or getattr(config, 'depth_estimation_type', None) != 'relative':
            raise ValueError('This depth adapter requires a relative Depth Anything checkpoint; metric and other families need their own semantics')
        if (getattr(config, 'backbone', None) is not None or getattr(config, 'use_pretrained_backbone', False)
                or getattr(config, 'use_timm_backbone', False)
                or getattr(config, 'backbone_kwargs', None) not in (None, {})
                or getattr(getattr(config, 'backbone_config', None), 'model_type', None) != 'dinov2'):
            raise ValueError('Relative depth requires an embedded DINOv2 backbone; external or pretrained-backbone resolution is unsupported')
        self.model, info = AutoModelForDepthEstimation.from_pretrained(spec.path,
            **dict(load_kwargs, trust_remote_code=False, output_loading_info=True))
        if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
            raise ValueError('Depth checkpoint has missing or incompatible model weights')
        if ((Path(spec.path) / 'config.json').read_bytes() != config_bytes or
                self.model.config.model_type != 'depth_anything' or self.model.config.depth_estimation_type != 'relative'):
            raise ValueError('Depth checkpoint configuration changed during loading')
        self.processor = AutoImageProcessor.from_pretrained(spec.path, local_files_only=True, trust_remote_code=False, use_fast=False)
        if not callable(getattr(self.processor, 'post_process_depth_estimation', None)):
            raise ValueError('Depth processor does not expose image-aligned depth postprocessing')
        self.model.to(device).eval()
        placement = {str(value.device) for value in self.model.parameters()}
        placement.update(str(value.device) for value in self.model.buffers())
        if len(placement) != 1:
            raise ValueError('Depth inference requires exactly one loaded device')
        self.device = placement.pop()
        if self.device.split(':')[0] != device.split(':')[0] or (':' in device and self.device != device):
            raise ValueError('Depth model did not load on the requested device')

    def details(self):
        return {'family': 'depth_anything', 'semantics': 'relative_inverse_depth', 'units': None,
                'larger_values': 'nearer', 'calibrated': False, 'normalization': 'none',
                'model_class': type(self.model).__name__, 'processor_class': type(self.processor).__name__,
                'device': self.device, 'repository': self.spec.source_repo, 'revision': self.spec.source_revision,
                'qualification': None}

    def infer(self, task, data, cancel=None):
        import numpy as np
        import torch
        if task != 'depth-estimation':
            raise EngineError('Depth checkpoint requires the depth-estimation task')
        if set(data) - {'model', 'image', 'timeout', 'location'}:
            raise EngineError('Unsupported depth controls')
        check_cancel(cancel)
        image = data['image']
        check_dimensions(image.width, image.height)
        rgb = image.convert('RGB')
        try:
            input_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
            inputs = self.processor(images=rgb, return_tensors='pt')
        finally:
            rgb.close()
        check_cancel(cancel)
        inputs = {name: value.to(device=self.device, dtype=self.model.dtype) if value.is_floating_point()
                  else value.to(self.device) for name, value in inputs.items()}
        check_cancel(cancel)
        with torch.inference_mode():
            output = self.model(**inputs)
            check_cancel(cancel)
            raw = output.predicted_depth
            if raw.ndim != 3 or raw.shape[0] != 1 or not raw.is_floating_point():
                raise EngineError('Depth model returned an invalid batch shape', 502, 'invalid_backend_response')
            processed = self.processor.post_process_depth_estimation(output, target_sizes=[(image.height, image.width)])
        check_cancel(cancel)
        if not isinstance(processed, list) or len(processed) != 1:
            raise EngineError('Depth processor returned an invalid batch', 502, 'invalid_backend_response')
        values = processed[0]['predicted_depth'].detach().float().cpu().numpy()
        check_cancel(cancel)
        if values.shape != (image.height, image.width) or not np.isfinite(values).all():
            raise EngineError('Depth processor returned invalid image-aligned values', 502, 'invalid_backend_response')
        # Explicit little-endian float32 preserves numeric values without a display normalization.
        encoded = np.ascontiguousarray(values, dtype='<f4').tobytes()
        result = DepthResult(model=self.spec.name if hasattr(self.spec, 'name') else data.get('model', ''),
            width=image.width, height=image.height, data=base64.b64encode(encoded).decode('ascii'),
            value_sha256=hashlib.sha256(encoded).hexdigest(), input_rgb_sha256=input_hash,
            model_output_shape=list(raw.shape[-2:]), model_class=type(self.model).__name__,
            processor_class=type(self.processor).__name__, device=self.device)
        check_cancel(cancel)
        return result.model_dump()
