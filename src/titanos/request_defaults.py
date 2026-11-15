"""One model-default merge used by public requests and internal native completion."""
from .engine import EngineError


def model_defaults(payload, spec, exclude=frozenset()):
    """payload with spec.defaults filled into its unset fields; fields named in exclude are never filled."""
    if spec is None:
        return payload
    values = payload.model_dump(exclude_unset=True)
    request_tuning = dict(values.get('pipeline_options', {})) if 'pipeline_options' in type(payload).model_fields else {}
    fields = type(payload).model_fields
    token_field = 'max_output_tokens' if 'max_output_tokens' in fields and 'max_tokens' not in fields else 'max_tokens'
    aliases = {'max_tokens':token_field, 'max_new_tokens':token_field, 'max_output_tokens':token_field,
               'num_inference_steps':'steps', 'num_frames':'frames'}
    noninherited = {'model', 'worker', 'task', 'require_loaded'}
    # These defaults belong to Titanos's composition layer, not llama.cpp's wire request.
    native_metadata = {'min_score', 'min_margin', 'sentiment_prompt_version',
                       'sentiment_decoding', 'retrieval_prompt_version'}
    for key, value in spec.defaults.items():
        key = aliases.get(key, key)
        if key in exclude:
            continue
        if key == 'max_completion_tokens' and key not in fields and token_field in fields:
            if value is None:
                continue  # Preserve the optional alias's existing unset/null behavior.
            key = token_field
        if (getattr(spec, 'backend', None) == 'llama-cpp'
                and key not in fields and key not in native_metadata | noninherited):
            raise EngineError(f'Unsupported llama-cpp model default: {key}; use a documented Titanos request or native option',
                              400, 'unsupported_capability')
        if key in fields and key not in values and key not in noninherited:
            if key in request_tuning:
                continue
            if key == 'max_tokens' and 'max_completion_tokens' in values:
                continue
            if key == 'max_completion_tokens' and 'max_tokens' in payload.model_fields_set:
                continue
            values[key] = value
    try:
        return type(payload).model_validate(values)
    except ValueError as exc:
        raise EngineError(f'Model defaults exceed current request limits: {exc}') from exc
