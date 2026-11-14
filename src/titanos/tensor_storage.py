"""Read current tensor metadata without moving, materializing or copying weights."""
from collections import defaultdict


def describe(owner):
    scope = 'Current exposed parameter/buffer storage; not an execution trace or complete memory accounting. Offloaded/meta tensors may differ from the compute device.'
    try:
        components = getattr(owner, 'components', None)
    except Exception as exc:
        return {'scope': scope, 'components': {}, 'complete': False, 'error_type': type(exc).__name__}
    if not isinstance(components, dict):
        components = {'model': owner}
    result = {}
    for name, component in components.items():
        groups = defaultdict(lambda: {'tensors': 0, 'elements': 0})
        error = None
        offload = None
        try:
            parameters = getattr(component, 'parameters', None)
            if not callable(parameters):
                continue
            offload = hasattr(component, '_hf_hook')
            for kind, method in [('parameter', parameters), ('buffer', getattr(component, 'buffers', None))]:
                if not callable(method):
                    continue
                seen = set()
                for tensor in method():
                    if id(tensor) in seen:
                        continue
                    seen.add(id(tensor))
                    row = groups[(kind, str(tensor.device), str(tensor.dtype))]
                    row['tensors'] += 1
                    row['elements'] += int(tensor.numel())
        except Exception as exc:
            error = type(exc).__name__
        result[str(name)] = {'class': type(component).__name__,
            'has_offload_hook': offload,
            'complete': error is None,
            'groups': [{'kind': kind, 'device': device, 'dtype': dtype, **counts}
                       for (kind, device, dtype), counts in sorted(groups.items())]}
        if error is not None:
            result[str(name)]['error_type'] = error
    return {'scope': scope, 'components': result, 'complete': all(row['complete'] for row in result.values())}
