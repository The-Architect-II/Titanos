"""Small policies for builtin-only checkpoint loading; no runtime imports."""
import re


def validate_weight_variant(value):
    """A local filename suffix, never a path or executable loader option."""
    if not isinstance(value,str) or re.fullmatch(r'[A-Za-z0-9_-]+',value) is None:
        raise ValueError('Weight variant must be a nonempty suffix of ASCII letters, numbers, underscores or hyphens, such as fp16 or non_ema; omit it for ordinary filenames')


def validate_attention_selectors(value):
    """Check raw dictionaries and parsed config objects, including private aliases.

    trust_remote_code=False does not prohibit Transformers Hub-kernel attention
    selectors. Keep this policy explicit before any model loader is entered.
    """
    aliases={'attn_implementation','attn_implementation_internal',
             'attention_implementation','attention_implementation_internal'}
    seen=set()
    def visit(item):
        if id(item) in seen:return
        seen.add(id(item))
        if isinstance(item,dict):
            for key,selection in item.items():
                if isinstance(key,str) and key.lstrip('_') in aliases:
                    if selection is not None and (type(selection) is not str or selection not in ('eager','sdpa')):
                        raise ValueError('Checkpoint attention selectors must be None, eager or sdpa; external kernels and selector maps are unsupported')
                visit(selection)
        elif isinstance(item,(list,tuple)):
            for child in item:visit(child)
        elif hasattr(item,'__dict__'):
            visit(vars(item))
    visit(value)
