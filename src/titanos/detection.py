"""Optional, class-aware suppression without changing model scores or geometry."""
import math

from .engine import EngineError
from .settings import current_settings


def options(data, defaults):
    settings = current_settings()
    resolved = {}
    for name, process_name in [('threshold', 'detection_threshold'), ('nms', 'detection_nms'), ('nms_iou', 'detection_nms_iou')]:
        value = data.get(name)
        if value is None:
            value = defaults.get(name)
        if value is None:
            value = getattr(settings, process_name)
        if name == 'nms':
            if type(value) is not bool:
                raise ValueError('Detection nms must be a boolean')
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f'Detection {name} must be between zero and one')
        resolved[name] = value
    return resolved


def overlap(first, second):
    width = max(0.0, min(first['xmax'], second['xmax']) - max(first['xmin'], second['xmin']))
    height = max(0.0, min(first['ymax'], second['ymax']) - max(first['ymin'], second['ymin']))
    intersection = width * height
    area = lambda box: (box['xmax'] - box['xmin']) * (box['ymax'] - box['ymin'])
    return intersection / (area(first) + area(second) - intersection)


def suppress(objects, iou_threshold, cancel=None):
    """Keep the highest score per overlapping same-class group, in input order."""
    def boundary():
        if cancel is not None and cancel.is_set():
            raise EngineError('Detection postprocessing cancelled', 499, 'cancelled')
    boundary()
    for item in objects:
        boundary()
        score, box = item['score'], item['box']
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
            raise EngineError('Detector returned an invalid score', 502, 'invalid_backend_response')
        if type(item['label_id']) is not int or item['label_id'] < 0:
            raise EngineError('Detector returned an invalid class ID', 502, 'invalid_backend_response')
        values = [box[name] for name in ('xmin', 'ymin', 'xmax', 'ymax')]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
            raise EngineError('Detector returned non-finite geometry', 502, 'invalid_backend_response')
        area = (box['xmax'] - box['xmin']) * (box['ymax'] - box['ymin'])
        if box['xmax'] <= box['xmin'] or box['ymax'] <= box['ymin'] or area <= 0 or not math.isfinite(area):
            raise EngineError('Detector returned invalid box extents', 502, 'invalid_backend_response')
    selected = []
    # Model query sets are usually small; cancellation bounds this quadratic pass.
    for index in sorted(range(len(objects)), key=lambda index: (-objects[index]['score'], index)):
        boundary()
        for kept in selected:
            boundary()
            if objects[index]['label_id'] == objects[kept]['label_id'] and overlap(objects[index]['box'], objects[kept]['box']) > iou_threshold:
                break
        else:
            selected.append(index)
    retained = sorted(selected)
    retained_set = set(retained)
    return [objects[index] for index in retained], {
        'method': 'class_aware_nms', 'iou_threshold': iou_threshold, 'original_count': len(objects),
        'retained_indices': retained, 'removed_indices': [index for index in range(len(objects)) if index not in retained_set],
        'index_space': 'model_postprocess_output', 'output_order': 'model', 'tie_break': 'lowest_original_index',
        'scores_modified': False, 'boxes_clipped': False,
    }
