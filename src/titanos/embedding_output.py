"""The existing public embedding envelope, shared by direct and composed calls."""
import base64
import struct


def format_embeddings(payload, result):
    vectors = result['vectors']
    if payload.encoding_format == 'base64':
        vectors = [base64.b64encode(struct.pack(f'<{len(vector)}f', *vector)).decode() for vector in vectors]
    response = {'object':'list','model':payload.model,
                'data':[{'object':'embedding','index':index,'embedding':vector} for index,vector in enumerate(vectors)],
                'usage':result['usage']}
    for name in ('input_details','truncation','provenance'):
        if result.get(name) is not None:
            response[name] = result[name]
    return response
