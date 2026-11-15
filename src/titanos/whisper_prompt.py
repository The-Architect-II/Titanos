"""Whisper context IDs and decoder budgets; never rewrite recognized text."""
from copy import deepcopy
import hashlib
import math
from types import SimpleNamespace

from .engine import EngineError
from .settings import current_settings
from .text_inputs import check_cancel


def prepare(pipeline,spec,prompt,generate,return_timestamps,cancel=None):
    if (getattr(pipeline,'type',None)!='seq2seq_whisper' or getattr(pipeline.model.config,'model_type',None)!='whisper'
            or spec.backend!='transformers' or spec.trust_remote_code):
        raise EngineError('Transcription prompt requires builtin Whisper',400,'unsupported_capability')
    if not isinstance(prompt,str) or not prompt.strip() or len(prompt)>current_settings().max_prompt_chars:
        raise EngineError('ASR prompt must be nonblank and within TITANOS_MAX_PROMPT_CHARS')
    conflicts={'prompt_ids','decoder_input_ids','decoder_start_token_id','forced_decoder_ids','generation_config'} & generate.keys()
    if conflicts or generate.get('prompt_condition_type','first-segment')!='first-segment':
        raise EngineError('Public ASR prompt cannot combine with decoder-prefix overrides or all-segments conditioning')
    check_cancel(cancel)
    import torch
    from transformers import WhisperProcessor
    from transformers.models.whisper.generation_whisper import WhisperGenerationMixin
    processor=WhisperProcessor(feature_extractor=pipeline.feature_extractor,tokenizer=pipeline.tokenizer)
    ids=processor.get_prompt_ids(prompt,return_tensors='pt')
    check_cancel(cancel)
    if ids.ndim!=1 or ids.numel()<2 or ids.dtype!=torch.long:
        raise EngineError('Whisper processor returned invalid context IDs',502,'invalid_backend_response')
    vocabulary=pipeline.model.config.vocab_size
    if any(type(value) is not int or not 0 <= value < vocabulary for value in ids.tolist()):
        raise EngineError('Whisper context IDs exceed the model vocabulary',502,'invalid_backend_response')
    capacity=pipeline.model.config.max_target_positions
    if type(capacity) is not int or capacity<4:
        raise EngineError('Whisper decoder capacity is unavailable')
    context_limit=capacity//2-1
    if ids.numel()-1>context_limit:
        raise EngineError(f'ASR prompt exceeds Whisper context limit of {context_limit} text tokens; no truncation was applied')
    # The ASR pipeline applies its own generation defaults (for example beams)
    # after model loading. Preserve those, rather than replacing them with the
    # model's earlier config when adding prompt IDs.
    generation=deepcopy(pipeline.generation_config)
    if getattr(generation,'prompt_condition_type',None) not in (None,'first-segment'):
        raise EngineError('Public ASR prompt supports first-segment conditioning only')
    first=int(ids[0].item())
    previous=getattr(generation,'prev_sot_token_id',None)
    if previous is not None and previous!=first:
        raise EngineError('Whisper tokenizer and generation config disagree on the previous-text sentinel')
    # Older prepared configs may lack this tokenizer-defined marker. Fill it only
    # in the request copy, so native first-segment handling strips it correctly.
    generation.prev_sot_token_id=first
    generation.prompt_condition_type='first-segment'
    WhisperGenerationMixin._set_language_and_task(generate.get('language'),generate.get('task'),generate.get('is_multilingual'),generation)
    generation.return_timestamps=bool(return_timestamps or getattr(generation,'return_timestamps',False))
    # Reuse the builtin prefix builder without invoking language detection/model
    # work. Its language IDs have identical width; these stand-ins are never sent
    # to generate. Both pinned 4.57.3 and 4.57.6 helpers use only these two members.
    def language_width(**kwargs):
        mapping=getattr(generation,'lang_to_id',None)
        if not isinstance(mapping,dict) or not mapping:
            raise EngineError('Whisper language-prefix metadata is unavailable')
        return torch.tensor([next(iter(mapping.values()))],dtype=torch.long)
    prefix=WhisperGenerationMixin._retrieve_init_tokens(SimpleNamespace(device='cpu',detect_language=language_width),
        None,1,generation,pipeline.model.config,0,{})
    prefix_tokens=prefix.shape[-1]
    maximum=generate.get('max_new_tokens',getattr(generation,'max_new_tokens',None))
    if maximum is not None and (type(maximum) is not int or maximum<1 or maximum>current_settings().max_tokens):
        raise EngineError('Prompted Whisper max_new_tokens must be a positive integer within TITANOS_MAX_TOKENS')
    trace={'characters':len(prompt),'sha256':hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
        'text_tokens':ids.numel()-1,'prompt_id_tokens':ids.numel(),'context_token_limit':context_limit,
        'decoder_prefix_budget':prefix_tokens,'decoder_capacity':capacity,'max_new_tokens':maximum,
        'conditioning':'first-segment','truncated':False}
    check_cancel(cancel)
    generate.update(prompt_ids=ids.to(pipeline.model.device),generation_config=generation,prompt_condition_type='first-segment')
    return trace


def check_budget(trace,pipeline,generate,seconds):
    chunk=getattr(pipeline.feature_extractor,'chunk_length',None)
    if type(chunk) not in (int,float) or not math.isfinite(chunk) or chunk<=0:
        raise EngineError('Whisper feature-extractor window is unavailable')
    generation=generate['generation_config']
    previous=generate.get('condition_on_prev_tokens')
    if previous is None:previous=getattr(generation,'condition_on_prev_tokens',False)
    if previous is None:previous=False
    if type(previous) is not bool:
        raise EngineError('Whisper condition_on_prev_tokens must be a boolean or None')
    static=generate.get('cache_implementation',getattr(generation,'cache_implementation',None))=='static'
    # Later long-form segments and static caches may occupy the entire previous
    # token window. Reserve it instead of silently reducing max_new_tokens later.
    reserve=trace['context_token_limit']+1 if previous and (seconds>chunk or static) else trace['prompt_id_tokens']
    remaining=trace['decoder_capacity']-trace['decoder_prefix_budget']-reserve
    if remaining<1 or trace['max_new_tokens'] is not None and trace['max_new_tokens']>remaining:
        raise EngineError(f'ASR prompt/prefix leaves {max(remaining,0)} new decoder tokens; reduce prompt or max_tokens')
    trace.update(previous_token_reserve=reserve,available_new_tokens=remaining)
