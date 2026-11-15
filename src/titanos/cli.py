"""Small command-line interface; the server never installs packages or weights."""

import argparse
import contextlib
import json
import os
import re
import shutil
import sys
import subprocess
from pathlib import Path

from . import __version__
from .config import Config, ModelSpec, NativeOptions, WorkerSpec, config_path, read_config, update_config


def client_request(args, method, route, **kwargs):
    from .client import Client
    with Client(args.url, timeout=args.timeout) as client:
        return client.request(method, route, **kwargs).json()


def remote_options(parser):
    parser.add_argument("--url", default=os.environ.get("TITANOS_URL", "http://127.0.0.1:11435"))
    parser.add_argument("--timeout", type=float, help="HTTP timeout in seconds; defaults to TITANOS_CLIENT_TIMEOUT")


def inference_options(parser):
    """Routing hints of an inference subcommand (sent as X-Titanos-Location and X-Titanos-Target) and --verbose."""
    parser.add_argument("--location", choices=["local_only", "prefer_local", "auto", "cloud_only"],
                        help="Where the request may run (X-Titanos-Location)")
    parser.add_argument("--target", metavar="REF", help="Pin one target, e.g. model@upstream (X-Titanos-Target; needs route:explicit)")
    parser.add_argument("--verbose", action="store_true", help="Print the X-Titanos-* provenance to stderr; stdout is unchanged")


def request_options(args):
    options = json.loads(args.options.read_text(encoding="utf-8")) if getattr(args, "options", None) else {}
    if not isinstance(options, dict):
        raise ValueError("--options must contain a JSON object")
    if set(options) & {"model", "messages", "stream", "input", "prompt", "task"}:
        raise ValueError("--options cannot replace model, messages, stream, input, prompt or task")
    return options


def run_chat(args, client):
    from .client import image_part
    messages = []
    if args.messages and args.history:
        raise ValueError("Use --messages or --history, not both")
    if args.messages and args.image and args.prompt is None:
        raise ValueError("Pass a prompt when adding --image to --messages")
    source = args.messages or (args.history if args.history and args.history.exists() else None)
    if source:
        messages = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
            raise ValueError("Chat history/messages must contain a JSON array of messages")
    if args.system and not messages:
        messages.append({"role": "system", "content": args.system})
    options = request_options(args)
    options.update({k: v for k, v in {"worker": args.worker, "max_tokens": args.max_tokens,
                                    "temperature": args.temperature, "seed": args.seed}.items() if v is not None})
    if args.stream:
        options.setdefault("stream_options", {"include_usage": True})

    def run(prompt):
        outgoing = list(messages)
        if prompt is not None:
            content = ([{"type": "text", "text": prompt}] + [image_part(p) for p in args.image]) if args.image else prompt
            outgoing.append({"role": "user", "content": content})
        if not outgoing:
            raise ValueError("A prompt or messages file is required")
        result = client.chat(args.name, outgoing, stream=args.stream, **options)
        if args.stream:
            message, calls = {"role": "assistant", "content": ""}, {}
            for event in result:
                if args.json:
                    print(json.dumps(event), flush=True)
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    text = delta.get("content") or ""
                    message["content"] += text
                    if not args.json:
                        print(text, end="", flush=True)
                    for call in delta.get("tool_calls", []):
                        target = calls.setdefault(call["index"], {"type": "function", "function": {"name": "", "arguments": ""}})
                        if call.get("id"):
                            target["id"] = call["id"]
                        for key in ("name", "arguments"):
                            target["function"][key] += call.get("function", {}).get(key, "")
            if calls:
                message["tool_calls"] = [calls[i] for i in sorted(calls)]
            if not args.json:
                print()
                if calls:
                    print(json.dumps({"tool_calls": message["tool_calls"]}))
        else:
            message = result["choices"][0]["message"]
            print(json.dumps(result, indent=2) if args.json else (message.get("content") or json.dumps(message)))
        messages[:] = outgoing + [message]
        if args.history:
            client.save_asset((json.dumps(messages, indent=2) + "\n").encode("utf-8"), args.history)
        if args.output:
            client.save_asset((message.get("content") or json.dumps(message)).encode("utf-8"), args.output)

    if args.prompt is not None:
        run(sys.stdin.read() if args.prompt == "-" else args.prompt)
    elif args.messages:
        run(None)
    elif not sys.stdin.isatty():
        prompt = sys.stdin.read()
        if not prompt.strip():
            raise ValueError("Pass a prompt, pipe text to stdin, or run in an interactive terminal")
        run(prompt)
    else:
        while True:
            # Keep machine-readable stdout free of interactive prompt characters.
            print("> ", end="", file=sys.stderr, flush=True)
            prompt = input().strip()
            if prompt in {"/exit", "/quit"}:
                break
            if prompt:
                run(prompt)


@contextlib.contextmanager
def provenance_report(args, client):
    """--verbose: the X-Titanos-* provenance of the last call goes to stderr, so stdout stays the unchanged result."""
    try:
        yield
    finally:
        if getattr(args, "verbose", False) and client.last_provenance:
            print(json.dumps({"provenance": client.last_provenance}), file=sys.stderr)


def remote_command(args):
    from .client import Client, TitanosError
    with Client(args.url, timeout=args.timeout, location=getattr(args, "location", None),
                target=getattr(args, "target", None)) as client, provenance_report(args, client):
        if args.command == "run":
            return run_chat(args, client)
        if args.command=='providers':
            result=client.providers();print(json.dumps(result,indent=2));return
        if args.command=='discover':
            result=client.discover_models(args.provider,args.task,args.query,args.limit);print(json.dumps(result,indent=2));return
        if args.command=='source-inspect':
            result=client.inspect_model_source(args.provider,args.identifier,args.revision);print(json.dumps(result,indent=2));return
        if args.command=='import-link':
            result=client.import_model_link(args.link);print(json.dumps(result,indent=2));return
        if args.command=='acquire':
            request={key:value for key,value in {
                'provider':args.provider,'identifier':args.identifier,'revision':args.revision,
                'registry_name':args.name,'task':args.task,'backend':args.backend,'device':args.device,
                'dtype':args.dtype,'graph':args.graph,'model_file':args.model_file,'mmproj':args.mmproj,
                'variant':args.variant,'replace':args.replace}.items() if value is not None}
            plan=client.plan_acquisition(**request)
            value=client.submit_acquisition(plan,idempotency_key=args.idempotency_key)
            if args.wait or args.activate:
                value=client.wait_acquisition(value['id'],timeout=args.wait_timeout)
            if args.activate:
                if value['status']!='prepared':
                    raise TitanosError(f"Acquisition {value['id']} {value['status']}: {value.get('error') or 'not prepared'}")
                value=client.activate_acquisition(value['id'],accepted_license_sha256=args.accept_license_sha256,load=not args.no_load,
                    **({'runtime_profile':args.runtime_profile} if args.runtime_profile else {}))
            print(json.dumps({'plan':plan,'acquisition':value},indent=2));return
        if args.command=='acquisitions':
            if args.action!='list' and not args.id:raise ValueError('Acquisition action requires an ID')
            if args.action=='list':result=client.acquisitions()
            elif args.action=='get':result=client.acquisition(args.id)
            elif args.action=='cancel':result=client.cancel_acquisition(args.id)
            elif args.action=='retry':result=client.retry_acquisition(args.id)
            elif args.action=='wait':result=client.wait_acquisition(args.id,timeout=args.wait_timeout)
            elif args.action=='activate':result=client.activate_acquisition(args.id,accepted_license_sha256=args.accept_license_sha256,load=not args.no_load,
                **({'runtime_profile':args.runtime_profile} if args.runtime_profile else {}))
            else:result=client.delete_acquired_weights(args.id)
            print(json.dumps(result,indent=2))
            if result.get('status') in {'failed','cancelled','interrupted'}:
                raise TitanosError(f"Acquisition {result['id']} {result['status']}: {result.get('error') or 'not prepared'}")
            return
        if args.command=='runtime-prepare':
            if args.plan:result=client.plan_runtime_setup(args.profile)
            else:
                result=client.submit_runtime_setup(args.profile)
                if args.wait:result=client.wait_runtime_setup(result['id'],timeout=args.wait_timeout)
            print(json.dumps(result,indent=2));return
        if args.command=='runtime-profiles':
            print(json.dumps(client.runtime_profiles(),indent=2));return
        if args.command=='runtime-setups':
            if args.action!='list' and not args.id:raise ValueError('Runtime setup action requires an ID')
            if args.action=='list':result=client.runtime_setups()
            elif args.action=='get':result=client.runtime_setup(args.id)
            elif args.action=='cancel':result=client.cancel_runtime_setup(args.id)
            elif args.action=='retry':result=client.retry_runtime_setup(args.id)
            else:result=client.wait_runtime_setup(args.id,timeout=args.wait_timeout)
            print(json.dumps(result,indent=2))
            if result.get('status') in {'failed','cancelled','interrupted'}:
                raise TitanosError(f"Runtime setup {result['id']} {result['status']}: {result.get('error') or 'not prepared'}")
            return
        if args.command=='assignment-devices':
            options={**({'model':args.model} if args.model else {}),**({'refresh':True} if args.refresh else {})}
            print(json.dumps(client.assignment_devices(**options),indent=2));return
        if args.command=='assignment':
            if args.action=='get':result=client.assignment(args.model)
            else:
                groups=[[item.strip() for item in value.split(',') if item.strip()] for value in (args.group or [])]
                result=client.save_assignment(model=args.model,mode=args.mode,groups=groups,
                    tensor_parallel_size=args.tensor_parallel_size,
                    **({'threads':args.threads} if args.threads is not None else {}),
                    **({'timeout':args.worker_timeout} if args.worker_timeout is not None else {}),
                    **({'policy':args.policy} if args.policy is not None else {}))
            print(json.dumps(result,indent=2));return
        options = request_options(args)
        if args.command in {'transcribe', 'analyze-audio'} and args.prompt is not None:
            options['prompt']=args.prompt
        if getattr(args, "worker", None):
            options["worker"] = args.worker
        if getattr(args, 'deadline_ms', None) is not None:
            options['timeout'] = args.deadline_ms
        for key in ("size", "steps", "seed", "frames", "fps", "strength", "voice", "language", "n", "source_language", "translation_model", "threshold", "nms", "nms_iou", "max_tokens", "temperature", "top_p", "top_k", "min_p", "repeat_penalty", 'input_modality', 'asr_uncertain', 'min_score', 'min_margin', 'max_input_tokens', 'prompt_version', 'decoding', 'llm_worker', 'truncate', 'top_n', 'return_documents', 'response_format', 'timestamp_granularities', 'sample_rate', 'min_speech_ms', 'min_silence_ms', 'speech_pad_ms', 'return_frames', 'vad_enabled', 'vad_model'):
            if getattr(args, key, None) is not None:
                options[key] = getattr(args, key)
        if args.command=='retrieval-answer':
            for key in ['embedding_worker','rerank_worker','retrieve_top_k','context_top_n','embedding_max_tokens','rerank_max_input_tokens','context_max_tokens']:
                if getattr(args,key,None) is not None:options[key]=getattr(args,key)
        if args.command == 'ocr':
            for key in ['psm', 'dpi', 'preserve_interword_spaces', 'char_whitelist', 'char_blacklist', 'include_geometry', 'max_output_bytes']:
                if getattr(args, key, None) is not None:options[key] = getattr(args, key)
        if args.command in {'translate', 'translate-audio'} and args.include_provenance is not None:
            options['include_provenance'] = args.include_provenance
        if args.command in {'diarize', 'analyze-audio'}:
            knobs = {key: getattr(args, key) for key in ('num_speakers', 'cluster_threshold', 'min_duration_on', 'min_duration_off') if getattr(args, key, None) is not None}
            if args.command == 'diarize':
                options.update(knobs)
            else:
                for key in ('diarization_worker', 'text_analysis_model', 'text_analysis_worker'):
                    if getattr(args, key, None) is not None:options[key] = getattr(args, key)
                for key in ('diarization_options', 'text_analysis_options'):
                    if getattr(args, key, None) is not None:
                        options[key] = json.loads(getattr(args, key).read_text(encoding='utf-8-sig'))
                if knobs:options['diarization_options'] = {**options.get('diarization_options', {}), **knobs}
        if args.command == 'translate-audio':
            for key in ['target_language', 'translation_worker']:
                if getattr(args, key, None) is not None:options[key] = getattr(args, key)
        if args.command == "models":
            result = client.models()
        elif args.command == "info":
            result = client.info()
        elif args.command == "embed":
            value = args.text
            if args.input_file:
                value = json.loads(args.input_file.read_text(encoding="utf-8"))
            elif not value or value == ["-"]:
                value = sys.stdin.read()
            result = client.embeddings(args.name, value, **options)
        elif args.command == 'rerank':
            if args.documents_file and args.documents:
                raise ValueError('Use positional documents or --documents-file, not both')
            documents = json.loads(args.documents_file.read_text(encoding='utf-8')) if args.documents_file else args.documents
            if not documents:
                documents = json.loads(sys.stdin.read())
            if args.deadline_ms is not None:
                options['timeout'] = args.deadline_ms
            result = client.rerank(args.name, args.query, documents, **options).model_dump(mode='json', exclude_none=True)
        elif args.command == 'classify-pairs':
            pairs = json.loads(args.pairs_file.read_text(encoding='utf-8-sig')) if args.pairs_file else json.loads(sys.stdin.read())
            result = client.classify_pairs(args.name,pairs,**options).model_dump(mode='json')
        elif args.command == 'diarize':
            result = client.diarize(args.name, args.file, **options).model_dump(mode='json')
        elif args.command == 'analyze-audio':
            options['diarization_model'] = args.diarization_model
            result = client.analyze_audio(args.name, args.file, **options).model_dump(mode='json')
        elif args.command == 'vad':
            result = client.vad(args.name, args.file, **options).model_dump(mode='json')
        elif args.command == "transcribe":
            result = client.transcribe(args.name, args.file, **options)
        elif args.command == "translate-audio":
            result = client.translate_audio(args.name, args.file, **options)
            if options.get('include_provenance'):result = result.model_dump(mode='json')
        elif args.command == "translate":
            text = sys.stdin.read() if args.text == "-" else args.text
            result = client.translate(args.name, text, args.target_language, **options)
            if options.get('include_provenance'):result = result.model_dump(mode='json')
        elif args.command == 'sentiment':
            text = sys.stdin.read() if args.text == '-' else args.text
            result = client.sentiment(args.name, text, **options)
        elif args.command == 'sentiment-compose':
            text = sys.stdin.read() if args.text == '-' else args.text
            result = client.compose_sentiment(args.name, args.llm_model, text, **options)
        elif args.command == 'retrieval-answer':
            documents=json.loads(args.documents_file.read_text(encoding='utf-8-sig'))
            result=client.retrieval_answer(args.name,args.query,documents,embedding_model=args.embedding_model,
                rerank_model=args.rerank_model,**options).model_dump(mode='json',exclude_none=True)
        elif args.command in {"ocr", "detect", "depth"}:
            result = getattr(client, args.command)(args.name, args.image, **options)
            if args.command == "depth": result = result.model_dump(mode="json")
        elif args.command == "segment":
            if bool(args.points) != bool(args.point_labels):
                raise ValueError("--points and --point-labels must be provided together")
            options.update({key: value for key, value in {
                "box": args.box,
                "points": json.loads(args.points.read_text(encoding="utf-8-sig")) if args.points else None,
                "point_labels": json.loads(args.point_labels.read_text(encoding="utf-8-sig")) if args.point_labels else None,
                "multimask": args.multimask, "mask_threshold": args.mask_threshold}.items() if value is not None})
            result = client.segment(args.name, args.image, **options).model_dump(mode="json")
        elif args.command == "speak":
            text = sys.stdin.read() if args.text == "-" else args.text
            result = {"output": str(client.save_asset(client.speech(args.name, text, **options), args.output))}
        elif args.command in {"image", "edit"}:
            if args.command == "edit":
                options.update({k: v for k, v in {"mask": args.mask,
                                "preserve_unmasked": args.preserve_unmasked,
                                "image_guidance_scale":args.image_guidance_scale}.items() if v is not None})
            result = (client.image(args.name, args.prompt, **options) if args.command == "image"
                      else client.edit(args.name, args.image, args.prompt, **options))
            if args.output:
                result = {**result, "saved": []}
                for i, asset in enumerate(result["data"]):
                    dest = args.output if len(result["data"]) == 1 else args.output.with_name(f"{args.output.stem}-{i + 1}{args.output.suffix}")
                    result["saved"].append(str(client.save_asset(asset, dest)))
                # Base64 is useful for API calls, but wasteful after saving files.
                result["data"] = [{k: v for k, v in item.items() if k != "b64_json"} for item in result["data"]]
        elif args.command == "music":
            lyrics=args.lyrics_file.read_text(encoding='utf-8') if args.lyrics_file else None
            music={key:value for key,value in {'mode':args.mode,'lyrics':lyrics,'duration':args.duration,
                'vocal_language':args.vocal_language,'bpm':args.bpm,'key_scale':args.key_scale,
                'seed':args.seed,'guidance_scale':args.guidance_scale,'steps':args.steps}.items() if value is not None}
            if args.reference:
                import base64,mimetypes
                raw=args.reference.read_bytes();music['reference_audio_base64']='data:'+(mimetypes.guess_type(args.reference.name)[0] or 'audio/wav')+';base64,'+base64.b64encode(raw).decode()
            result=client.submit_music(args.name,args.prompt,idempotency_key=args.idempotency_key,**music)
            if not args.async_job:
                print(f"Waiting for job {result['id']}",file=sys.stderr)
                result=client.wait_job(result['id'],timeout=args.wait_timeout,poll_retries=args.poll_retries)
        elif args.command == "video":
            if args.async_job and args.output:
                raise ValueError("--output requires waiting; omit --async or save with jobs wait")
            if args.image:
                from .client import image_part
                options["input_reference"] = image_part(args.image)["image_url"]["url"]
            result = client.submit_video(args.name, args.prompt, idempotency_key=args.idempotency_key, **options)
            if not args.async_job:
                print(f"Waiting for job {result['id']}", file=sys.stderr)
                result = client.wait_job(result["id"], timeout=args.wait_timeout, poll_retries=args.poll_retries)
        elif args.command == "jobs":
            if args.action == "list":
                result = client.jobs()
            elif args.action == "submit":
                payload = json.loads(args.request.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Job request must be a JSON object")
                if not payload.get("task") or not payload.get("model"):
                    raise ValueError("Job request requires task and model")
                result = client.submit_job(idempotency_key=args.idempotency_key, **payload)
            elif args.action == "get":
                result = client.job(args.id)
            elif args.action == "cancel":
                result = client.cancel_job(args.id)
            else:
                result = client.wait_job(args.id, timeout=args.wait_timeout, poll_interval=args.poll_interval, poll_retries=args.poll_retries)
        if args.command in {"jobs", "video", "music"} and getattr(args, "output", None):
            if result.get("status") == "completed":
                asset = result["result"]
                assets = asset.get("data", [asset])
                result["saved"] = []
                for i, item in enumerate(assets):
                    dest = args.output if len(assets) == 1 else args.output.with_name(f"{args.output.stem}-{i + 1}{args.output.suffix}")
                    result["saved"].append(str(client.save_asset(item, dest)))
        print(json.dumps(result, indent=2))
        if args.command in {"jobs", "video", "music"} and result.get("status") in {"failed", "cancelled", "interrupted"}:
            raise TitanosError(f"Job {result['id']} {result['status']}: {result.get('error') or 'no asset produced'}")


def register_with_gateway(url, job_type, port, health_url, *, attempts=300, key=None):
    """Managed role, explicit operator configuration (serve --register-with-gateway, hivemind register-service): POST
    {gateway}/register-service {job_type, port} once, after this server answers its readiness route. The only
    parent-directed outbound call; it is never retried after an answer. key: the gateway bearer (Authorization: Bearer),
    needed by a gateway on another host; without one the request carries no credential. The key never reaches a log line
    or the printed outcome."""
    import time
    import httpx
    from .broker.cli import gateway_url
    target = gateway_url(url) + "/register-service"
    for _ in range(attempts):
        try:
            # our own loopback readiness route; a self-signed certificate is not verified here
            if httpx.get(health_url, timeout=1, trust_env=False, verify=False).is_success:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    else:
        print("titanos: gateway registration skipped: the server never became ready", file=sys.stderr)
        return None
    headers = {"Authorization": "Bearer " + key} if key else {}
    try:
        response = httpx.post(target, json={"job_type": job_type, "port": port}, headers=headers, timeout=10,
                              trust_env=False, follow_redirects=False)
    except httpx.HTTPError as exc:
        print(f"titanos: gateway registration failed: {type(exc).__name__}", file=sys.stderr)
        return None
    print(f"titanos: gateway registration answered {response.status_code}"
          + (" (bearer sent)" if headers else " (no bearer: set TITANOS_API_KEY or --key-env)"), file=sys.stderr)
    return response.status_code


def hivemind_register_service(args):
    """`titanos hivemind register-service GATEWAY --job-type JT [--port P] [--url U] [--key-env VAR] [--attempts N]`:
    register a running Titanos with a HiveMind gateway, on this host or another. The bearer is read from the environment
    variable --key-env names (default TITANOS_API_KEY) and sent when set; the option takes the variable's name, never a
    secret value. Readiness is probed on the Titanos server --url names (default TITANOS_URL, then
    http://127.0.0.1:11435); --port defaults to that URL's port (the port the gateway will call)."""
    from urllib.parse import urlsplit
    from .broker.cli import gateway_url
    base = (args.url or os.environ.get("TITANOS_URL") or "http://127.0.0.1:11435").rstrip("/")
    parts = urlsplit(base)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("--url takes the http(s) URL of the Titanos server whose readiness is probed")
    port = args.port if args.port is not None else (parts.port or (443 if parts.scheme == "https" else 80))
    if not 1 <= port <= 65535:
        raise ValueError("--port must be in 1..65535")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", args.job_type or ""):
        raise ValueError("--job-type must be 1-64 characters [A-Za-z0-9._-]")
    if args.key_env:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", args.key_env):
            raise ValueError("--key-env takes the name of an environment variable, never a secret value")
        key = os.environ.get(args.key_env)
        if not key:
            raise ValueError(f"--key-env names {args.key_env}, which is not set; export the gateway key in that variable")
    else:
        key = os.environ.get("TITANOS_API_KEY") or None
    status = register_with_gateway(gateway_url(args.gateway), args.job_type, port, base + "/v1/health/ready",
                                   attempts=args.attempts, key=key)
    result = {"gateway": gateway_url(args.gateway), "job_type": args.job_type, "port": port, "bearer_sent": bool(key),
              "status": status}
    print(json.dumps(result, indent=2))
    if status is None or not 200 <= status < 300:
        raise ValueError("gateway registration did not succeed" + (f" (HTTP {status})" if status else ""))


def option_json(path, option):
    """Read a JSON option file, naming the option and file in any failure instead of a bare parser message."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"{option} file could not be read: {path}: {exc.strerror or exc}") from exc
    except ValueError as exc:
        raise ValueError(f"{option} must be a JSON file: {path}: {exc}") from exc


def require_torch(device):
    """Device pinning for CUDA/XPU needs the PyTorch runtime profile; say so instead of a bare import failure."""
    try:
        import torch
    except ImportError as exc:
        raise ValueError(f"Device {device!r} needs the PyTorch runtime profile; this package or environment has no torch "
                         "(use --device cpu, or the full runtime package)") from exc
    return torch


USERNAME_VARIABLES = ('LOGNAME', 'USER', 'LNAME', 'USERNAME')


def account_name():
    """The operating-system account name, or None when the platform cannot supply one (no controlling terminal, unmapped uid)."""
    try:
        name = os.getlogin()
    except OSError:
        name = None
    if not name and os.name == 'nt':
        import ctypes
        buffer = ctypes.create_unicode_buffer(257)
        size = ctypes.c_uint32(len(buffer))
        if ctypes.windll.advapi32.GetUserNameW(buffer, ctypes.byref(size)):
            name = buffer.value
    return name or None


def ensure_username_environment():
    """Runtime libraries ask getpass.getuser() for a user name (PyTorch's compiled-kernel cache does so while importing
    torch._inductor, which the Qwen3-TTS speech path reaches through torch.compiler.cudagraph_mark_step_begin). Without
    LOGNAME/USER/LNAME/USERNAME that lookup falls back to the Unix-only pwd module, so on Windows the first such request
    fails with "No module named 'pwd'". Provide the account name when every variable is absent; never change a set one."""
    if any(os.environ.get(name) for name in USERNAME_VARIABLES):
        return None
    os.environ['USERNAME'] = account_name() or 'titanos'
    return os.environ['USERNAME']


def main(argv=None):
    if argv is None and len(sys.argv)==1:
        argv=['serve','--open']
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    from .broker.cli import RedactingArgumentParser, gateway_url_arg; parser = RedactingArgumentParser(prog="titanos", description="Standalone multimodal inference")  # error() never echoes a typed value
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", help="Registry path (default: TITANOS_HOME/models.json)")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser('native-install', help='Install a checksum-pinned llama.cpp runtime')
    p.add_argument('--catalog',help='Trusted catalog override; defaults to TITANOS_NATIVE_CATALOG then packaged data')
    p.add_argument('--profile', required=True)
    p.add_argument('--target', type=Path)
    p.add_argument('--plan',action='store_true',help='Plan without downloading or installing')
    p = sub.add_parser('upgrade',help='Install a new versioned runtime; preserve the existing runtime and model registry')
    p.add_argument('--catalog',required=True)
    p.add_argument('--profile',default='auto')
    p.add_argument('--target',type=Path)
    p.add_argument('--gpu',type=int)
    p.add_argument('--plan',action='store_true')
    sub.add_parser("init", help="Create an empty local model registry")
    p = sub.add_parser("settings", help="Inspect effective settings, defaults and environment names")
    p.add_argument("--json", action="store_true")
    for command in ("add", "pull"):
        p = sub.add_parser(command, help="Register a local directory" if command == "add" else "Explicitly download a Hugging Face repository and register it")
        p.add_argument("name")
        p.add_argument("source", help="Local model directory" if command == "add" else "Hugging Face repository ID")
        p.add_argument("--task", required=True, choices=["chat", "vision", "embeddings", "transcription", "speech", "image", "video", "music", "tensor", "translation", "image-to-text", "object-detection", 'sentiment', 'rerank', 'vad', 'diarization', 'pair-classification', 'depth-estimation', 'image-segmentation'])
        p.add_argument("--backend", choices=["transformers", "diffusers", "xtts", "onnx", "openvino", "openvino-genai", "llama-cpp", "qwen3-tts", "sherpa-onnx", "ace-step", "tesseract", "onnx-diffusers"])
        p.add_argument("--device", default="auto")
        p.add_argument("--model-file", help="Relative GGUF filename for llama-cpp")
        p.add_argument("--mmproj", help="Matching relative GGUF vision projector")
        p.add_argument("--runtime", help="Path to llama-server; otherwise use the installed native runtime")
        p.add_argument("--native-options", type=Path, help="JSON NativeOptions, including slots, context_size and gpu_layers")
        p.add_argument("--graph", help="Relative prepared graph filename")
        p.add_argument("--allow-cpu-fallback", action="store_true")
        p.add_argument("--provider-options", type=Path, help="JSON file of vendor runtime options")
        p.add_argument("--defaults", type=Path, help="JSON generation defaults for this model; explicit request fields take precedence")
        p.add_argument("--voices", type=Path, help="JSON voice-name to local reference-audio path mapping")
        p.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
        p.add_argument("--parallelism", default="single", choices=["single", "model_sharding", "tensor_parallel"])
        p.add_argument("--tensor-parallel-size", type=int, default=2)
        p.add_argument("--allow-pickle", action="store_true", help="Explicitly permit trusted legacy checkpoints")
        p.add_argument("--trust-remote-code", action="store_true", help="Permit model repository Python code")
        p.add_argument("--replace", action="store_true", help="Replace an existing model registration")
        if command == "pull":
            p.add_argument("--revision", default="main")
            p.add_argument("--cache-dir", help="Directory that receives the plain per-revision model directory (default: TITANOS_HOME/models); no Hugging Face cache symlinks")
            auth = p.add_mutually_exclusive_group()
            auth.add_argument("--anonymous", action="store_true")
            auth.add_argument("--token-env", help="Read token from this environment variable; default: HF_TOKEN or saved login")
    for command in ("search", "inspect"):
        p = sub.add_parser(command, help="Browse the Hugging Face catalog" if command == "search" else "Inspect a Hugging Face model repository")
        if command == "search":
            p.add_argument("query", nargs="?", default="")
            p.add_argument("--author")
            p.add_argument("--task", help="Hugging Face pipeline tag, e.g. text-generation")
            p.add_argument("--library")
            p.add_argument("--sort", default="downloads", choices=["downloads", "likes", "last_modified"])
            p.add_argument("--limit", type=int, default=20)
            p.add_argument("--all", action="store_true", help="Stream every matching accessible model as JSON Lines")
        else:
            p.add_argument("repo")
            p.add_argument("--revision", default="main")
        auth = p.add_mutually_exclusive_group()
        auth.add_argument("--anonymous", action="store_true", help="Do not send any Hugging Face token")
        auth.add_argument("--token-env", help="Read an optional token from this environment variable")
    sub.add_parser("login", help="Save an optional Hugging Face token using a hidden prompt")
    sub.add_parser("logout", help="Remove locally saved Hugging Face tokens")
    p = sub.add_parser("prepare", help="Export a local embedding model, compare outputs, then register")
    p.add_argument("source", help="Existing Transformers embedding registration")
    p.add_argument("name", help="New model registration")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--backend", required=True, choices=["onnx", "openvino"])
    p.add_argument("--device", default="auto", help="Device on which export must pass numerical validation")
    p.add_argument("--sequence-length", type=int, default=128)
    sub.add_parser("list", help="List local model registrations")
    p = sub.add_parser('devices', help='Inspect the local hardware inventory; --refresh queries the selected runtime')
    p.add_argument('--backend',default='all',choices=['all','nvidia','pytorch','openvino','onnx','system'])
    p.add_argument('--refresh', action='store_true')
    sub.add_parser('pin-devices', help='Explicitly migrate existing numeric NVIDIA assignments to current GPU UUIDs')
    p = sub.add_parser("capabilities", help="Inspect a local model's execution modes without loading its weights")
    p.add_argument("name")
    p = sub.add_parser("worker", help="Assign models to an independent local worker; restart serve after changes")
    p.add_argument("name")
    p.add_argument("--models", nargs='+', required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--gpus", nargs='+', default=[], help="Full GPU-/HIP- UUIDs or current CUDA-visible indices; never driver indices")
    p.add_argument("--mode", default="single", choices=["single", "model_sharding", "tensor_parallel"])
    p.add_argument("--tensor-parallel-size", type=int, default=2, help="CPU/XPU rank count; CUDA count comes from --gpus")
    p.add_argument("--threads", type=int, help="Worker CPU threads; defaults to TITANOS_WORKER_THREADS")
    p.add_argument("--timeout", type=float, help="Worker execution timeout in seconds; defaults to TITANOS_WORKER_TIMEOUT")
    p.add_argument("--profile", action="store_true", help="Record actual GPU kernel devices per inference; adds overhead")
    p.add_argument("--replace", action="store_true")
    p = sub.add_parser("remove-worker", help="Remove a worker assignment; restart serve after changes")
    p.add_argument("name")
    p = sub.add_parser("remove", help="Remove a registration; leave its weights on disk")
    p.add_argument("name")
    p = sub.add_parser("serve", help="Run the independent HTTP server")
    p.add_argument("--open",action="store_true",help="Open the local browser application after startup")
    p.add_argument("--native-catalog",help="Trusted pinned runtime catalog for in-app setup; also TITANOS_NATIVE_CATALOG")
    p.add_argument("--host", help="Bind host; overrides TITANOS_HOST")
    p.add_argument("--port", type=int, help="Bind port; overrides TITANOS_PORT")
    p.add_argument("--threads", type=int, help="PyTorch CPU threads (default: up to 8)")
    p.add_argument("--limit-concurrency", type=int, help="HTTP concurrency limit; overrides TITANOS_LIMIT_CONCURRENCY")
    p.add_argument("--keep-alive-timeout", type=int, help="Idle HTTP connection timeout; overrides TITANOS_KEEP_ALIVE_TIMEOUT")
    p.add_argument("--log-level", choices=["critical", "error", "warning", "info", "debug", "trace"])
    p.add_argument("--role", choices=["standalone", "broker", "managed"], help="Deployment role; overrides TITANOS_ROLE; fixed until restart")
    p.add_argument("--register-with-gateway", metavar="URL", type=gateway_url_arg, help="Managed role: POST /register-service {job_type, port} to this gateway once at startup")
    p.add_argument("--job-type", help="The gateway job type for --register-with-gateway (it must already exist there)")
    p.add_argument("--pid-file", type=Path, metavar="PATH", help="Write this server's process id to PATH while it runs and remove it on exit (titanos service uses it to stop the server)")
    p = sub.add_parser("doctor", help="Report hardware and optionally execute a tensor operation")
    p.add_argument("--backend", default="pytorch", choices=["engine", "pytorch", "onnx", "openvino", "openvino-genai"])
    p.add_argument("--device", default="auto")
    p.add_argument("--check", action="store_true")
    p.add_argument('--provider-options',type=Path,help='Compiled-runtime options JSON, e.g. DirectML device_id')
    p.add_argument("--output", type=Path)
    for command in ("load", "unload", "run", "ps", "details", "workers"):
        p = sub.add_parser(command)
        remote_options(p)
        if command not in {"ps", "workers"}:
            p.add_argument("name")
            p.add_argument("--worker", help="Target a specific replica assignment")
        if command == "run":
            inference_options(p)
            p.add_argument("prompt", nargs="?")
            p.add_argument("--stream", action="store_true", help="Print incremental tokens; --json emits JSON Lines")
            p.add_argument("--json", action="store_true", help="Print the full response/event objects for agents")
            p.add_argument("--system", help="System prompt for a new conversation")
            p.add_argument("--history", type=Path, help="Read and atomically save a JSON conversation")
            p.add_argument("--messages", type=Path, help="Read a JSON message array for a single request")
            p.add_argument("--image", type=Path, action="append", default=[], help="Attach a local image to the prompt; repeat for multiple images")
            p.add_argument("--output", type=Path, help="Save the final assistant text")
            p.add_argument("--options", type=Path, help="Additional API options as a JSON object, including native tool/format options")
            p.add_argument("--max-tokens", type=int)
            p.add_argument("--temperature", type=float)
            p.add_argument("--seed", type=int)
    for command in ("models", "info"):
        remote_options(sub.add_parser(command, help="Inspect the running server's " + command))
    for command in ("embed", "transcribe", "speak", "image", "edit", "video", "music", "translate", "translate-audio", "ocr", "detect", "depth", "segment", 'sentiment', 'sentiment-compose', 'rerank', 'vad', 'diarize', 'analyze-audio', 'retrieval-answer', 'classify-pairs'):
        p = sub.add_parser(command, help="Run " + command + " inference on the standalone server")
        remote_options(p)
        inference_options(p)
        p.add_argument("name", help="Registered model name")
        if command!='retrieval-answer':p.add_argument("--worker")
        p.add_argument("--options", type=Path, help="Additional API options as a JSON object")
        if command in {"embed","transcribe","analyze-audio","translate-audio","translate","ocr",'sentiment','sentiment-compose'}:
            p.add_argument("--max-tokens",type=int)
        if command == "embed":
            p.add_argument("text", nargs="*", help="One or more texts; '-' reads stdin")
            p.add_argument("--input-file", type=Path, help="JSON string or array of strings")
            p.add_argument('--truncate', action=argparse.BooleanOptionalAction, default=None, help='Explicitly allow or forbid input truncation')
        elif command=='retrieval-answer':
            p.add_argument('query',help='Question answered only from the supplied passages')
            p.add_argument('--documents-file',type=Path,required=True,help='UTF-8 JSON array of {id,text,title?,source_uri?}')
            p.add_argument('--embedding-model',required=True)
            p.add_argument('--rerank-model',required=True)
            p.add_argument('--prompt-version',choices=['retrieval-answer-v1','retrieval-answer-v2'],help='Source-reading policy; omitted inherits the answer-model/server default')
            for name in ['embedding-worker','rerank-worker','llm-worker']:p.add_argument('--'+name)
            for name in ['retrieve-top-k','context-top-n','embedding-max-tokens','rerank-max-input-tokens','context-max-tokens','max-tokens','deadline-ms']:p.add_argument('--'+name,type=int)
            p.add_argument('--temperature',type=float)
            p.add_argument('--top-p',type=float)
            p.add_argument('--return-documents',action=argparse.BooleanOptionalAction,default=None)
        elif command == 'rerank':
            p.add_argument('query', help='Query text')
            p.add_argument('documents', nargs='*', help='Documents in original order; absent reads a JSON array from stdin')
            p.add_argument('--documents-file', type=Path, help='JSON array of document strings')
            p.add_argument('--top-n', type=int, help='Return at most this many ranked documents')
            p.add_argument('--return-documents', action=argparse.BooleanOptionalAction, default=None)
            p.add_argument('--max-input-tokens', type=int, help='Maximum tokens per query/document pair')
            p.add_argument('--deadline-ms', type=int, help='Server deadline including model-queue wait, in milliseconds')
            p.add_argument('--truncate', action=argparse.BooleanOptionalAction, default=None, help='Permit document truncation while preserving the query')
        elif command == 'classify-pairs':
            p.add_argument('--pairs-file',type=Path,help='UTF-8 JSON pair array; omission reads stdin')
            p.add_argument('--max-input-tokens',type=int,help='Maximum tokens per complete premise/hypothesis pair')
            p.add_argument('--truncate',action=argparse.BooleanOptionalAction,default=None,help='Permit premise-only truncation; hypothesis stays complete')
            p.add_argument('--deadline-ms',type=int,help='Shared request deadline in milliseconds')
        elif command in {'diarize', 'analyze-audio'}:
            p.add_argument('file', type=Path)
            p.add_argument('--deadline-ms', type=int, help='One server deadline for every selected stage, in milliseconds')
            p.add_argument('--num-speakers', type=int, help='Anonymous speaker count; zero selects automatic clustering')
            p.add_argument('--cluster-threshold', type=float, help='Native clustering threshold, not a probability')
            p.add_argument('--min-duration-on', type=float)
            p.add_argument('--min-duration-off', type=float)
            if command == 'analyze-audio':
                p.add_argument('--diarization-model', required=True)
                p.add_argument('--diarization-worker')
                p.add_argument('--diarization-options', type=Path, help='JSON object of diarization controls; explicit flags override it')
                p.add_argument('--text-analysis-model', help='Optional registered sentiment classifier or local chat model')
                p.add_argument('--text-analysis-worker')
                p.add_argument('--text-analysis-options', type=Path, help='JSON transcript-text sentiment controls')
                p.add_argument('--language')
                p.add_argument('--prompt', help='ASR vocabulary/context, never transcript replacement')
                p.add_argument('--timestamp-granularities', nargs='+', choices=['segment', 'word'])
                p.add_argument('--vad-model')
                p.add_argument('--vad', dest='vad_enabled', action=argparse.BooleanOptionalAction, default=None)
        elif command == 'vad':
            p.add_argument('file', type=Path)
            p.add_argument('--sample-rate', type=int, choices=[8000, 16000])
            p.add_argument('--threshold', type=float)
            p.add_argument('--min-speech-ms', type=float)
            p.add_argument('--min-silence-ms', type=float)
            p.add_argument('--speech-pad-ms', type=float)
            p.add_argument('--return-frames', action=argparse.BooleanOptionalAction, default=None)
            p.add_argument('--deadline-ms', type=int)
        elif command in {"transcribe", "translate-audio"}:
            p.add_argument("file", type=Path)
            p.add_argument("--language")
            p.add_argument('--response-format', choices=['json', 'text', 'verbose_json'])
            p.add_argument('--deadline-ms', type=int)
            p.add_argument('--vad-model', help='Registered VAD model to check before ASR')
            p.add_argument('--vad', dest='vad_enabled', action=argparse.BooleanOptionalAction, default=None, help='Enable the configured gate, or explicitly disable it with --no-vad')
            if command == 'transcribe':
                p.add_argument('--timestamp-granularities', nargs='+', choices=['segment', 'word'])
                p.add_argument('--prompt',help='Whisper transcription vocabulary/context; never a transcript replacement')
            if command == "translate-audio":
                p.add_argument("--translation-model", help="Optional registered translation model")
                p.add_argument('--translation-worker')
                p.add_argument('--source-language', help='NMT tokenizer source ID, distinct from the ASR --language hint')
                p.add_argument('--target-language', help='English NMT tokenizer target ID, for example en for a fixed Marian fr-en model')
                p.add_argument('--include-provenance', action=argparse.BooleanOptionalAction, default=None)
        elif command == "translate":
            p.add_argument("text", help="Text to translate; '-' reads stdin")
            p.add_argument("--target-language", required=True)
            p.add_argument("--source-language")
            p.add_argument('--include-provenance', action=argparse.BooleanOptionalAction, default=None)
            p.add_argument('--deadline-ms', type=int)
        elif command in {'sentiment', 'sentiment-compose'}:
            if command == 'sentiment-compose':
                p.add_argument('llm_model', help='Registered LLM to compare with the classifier')
                p.add_argument('--llm-worker')
            p.add_argument('text', help="Text to classify; '-' reads stdin")
            p.add_argument('--input-modality', choices=['text', 'asr_transcript'])
            p.add_argument('--asr-uncertain', action='store_true', default=None)
            p.add_argument('--language', help='Caller-declared text language; not a language-detection result')
            p.add_argument('--min-score', type=float)
            p.add_argument('--min-margin', type=float)
            p.add_argument('--max-input-tokens', type=int)
            p.add_argument('--temperature', type=float)
            p.add_argument('--top-p', type=float)
            p.add_argument('--prompt-version', choices=['weights-v1', 'label-v2', 'label-v3', 'facets-v1'])
            p.add_argument('--decoding', choices=['auto', 'validate', 'label_choices', 'facet_choices', 'native_json_schema'])
            p.add_argument('--deadline-ms', type=int, help='Shared inference deadline in milliseconds, distinct from HTTP --timeout seconds')
        elif command in {"ocr", "detect", "depth"}:
            p.add_argument("image", type=Path)
            if command == 'ocr':
                p.add_argument('--language', help='Tesseract traineddata language IDs, for example eng or eng+deu')
                p.add_argument('--psm', type=int, choices=[3,4,5,6,7,8,9,10,11,13], help='Tesseract page segmentation mode; 7 is a single line')
                p.add_argument('--dpi', type=int)
                p.add_argument('--max-output-bytes', type=int)
                p.add_argument('--deadline-ms', type=int, help='Shared inference deadline in milliseconds')
                for flag in ['preserve-interword-spaces', 'include-geometry']:
                    p.add_argument('--' + flag, action=argparse.BooleanOptionalAction, default=None)
                p.add_argument('--char-whitelist')
                p.add_argument('--char-blacklist')
            if command == "depth":
                p.add_argument("--deadline-ms", type=int, help="Shared inference deadline in milliseconds")
            if command == "detect":
                p.add_argument("--threshold", type=float)
                p.add_argument('--nms', action=argparse.BooleanOptionalAction, default=None, help='Filter overlapping boxes within each native class')
                p.add_argument('--nms-iou', type=float, help='IoU threshold for enabled NMS; omitted inherits model/process policy')
        elif command == "segment":
            p.add_argument("image", type=Path)
            prompt = p.add_mutually_exclusive_group(required=True)
            prompt.add_argument("--box", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"),
                                help="One box in original image pixel coordinates")
            prompt.add_argument("--points", type=Path, help="UTF-8 JSON array of [x,y] original image coordinates")
            p.add_argument("--point-labels", type=Path, help="UTF-8 JSON array: 1 selects foreground, 0 selects background")
            p.add_argument("--multimask", action=argparse.BooleanOptionalAction, default=None,
                           help="Return multiple candidate masks; omission inherits model/process policy")
            p.add_argument("--mask-threshold", type=float, help="Mask logit threshold; omission inherits model/process policy")
            p.add_argument("--deadline-ms", type=int, help="Shared inference deadline in milliseconds")
        elif command == "music":
            p.add_argument("prompt");p.add_argument("--mode",choices=["instrumental","song"],default="instrumental")
            p.add_argument("--lyrics-file",type=Path);p.add_argument("--duration",type=float);p.add_argument("--vocal-language")
            p.add_argument("--bpm",type=int);p.add_argument("--key-scale");p.add_argument("--seed",type=int)
            p.add_argument("--guidance-scale",type=float);p.add_argument("--steps",type=int);p.add_argument("--reference",type=Path)
            p.add_argument("--async",dest="async_job",action="store_true");p.add_argument("--wait-timeout",type=float)
            p.add_argument("--poll-retries",type=int,default=3);p.add_argument("--idempotency-key");p.add_argument("--output",type=Path)
        elif command == "speak":
            p.add_argument("text", help="Text to speak; '-' reads stdin")
            p.add_argument("--voice")
            p.add_argument("--language")
            p.add_argument("--output", type=Path, required=True, help="Output WAV file")
        else:
            p.add_argument("prompt")
            p.add_argument("--size", help="WIDTHxHEIGHT")
            p.add_argument("--steps", type=int)
            p.add_argument("--seed", type=int)
            p.add_argument("--output", type=Path, help="Save the generated PNG or MP4")
            if command == "edit":
                p.add_argument("--image", type=Path, required=True)
                p.add_argument("--strength", type=float)
                p.add_argument('--image-guidance-scale',type=float,help='InstructPix2Pix source-image guidance (at least one)')
                p.add_argument("--mask", type=Path, help="RGBA PNG mask; transparent pixels are editable")
                p.add_argument("--preserve-unmasked", action=argparse.BooleanOptionalAction, default=None,
                               help="Copy original RGB pixels outside the mask after generation (server default: enabled)")
            if command in {"image", "edit"}:
                p.add_argument("--n", type=int, help="Image count; multiple files receive numeric suffixes")
            elif command == "video":
                p.add_argument("--image", type=Path, help="Input image for an image-conditioned video model")
                p.add_argument("--frames", type=int)
                p.add_argument("--fps", type=int)
                p.add_argument("--async", dest="async_job", action="store_true", help="Return the job immediately")
                p.add_argument("--wait-timeout", type=float, help="Job wait deadline; defaults to TITANOS_CLIENT_TIMEOUT and does not cancel the job")
                p.add_argument("--poll-retries", type=int, default=3, help="Consecutive transient status-read retries; never resubmits inference")
                p.add_argument("--idempotency-key")
    p = sub.add_parser("jobs", help="Submit, list, inspect, wait for or cancel durable asset jobs")
    jobs = p.add_subparsers(dest="action", required=True)
    for action in ("list", "submit", "get", "wait", "cancel"):
        p = jobs.add_parser(action)
        remote_options(p)
        if action in {"get", "wait", "cancel"}:
            p.add_argument("id")
        if action == "submit":
            p.add_argument("request", type=Path, help="JSON request containing task, model and modality options")
            p.add_argument("--idempotency-key")
        elif action == "wait":
            p.add_argument("--wait-timeout", type=float, help="Job wait deadline; defaults to TITANOS_CLIENT_TIMEOUT")
            p.add_argument("--poll-interval", type=float, default=1)
            p.add_argument("--poll-retries", type=int, default=3, help="Consecutive transient status-read retries")
            p.add_argument("--output", type=Path, help="Save the completed asset")
    p=sub.add_parser('providers',help='List task-first model discovery/acquisition providers')
    remote_options(p)
    p=sub.add_parser('discover',help='Search one model provider by Titanos task')
    remote_options(p);p.add_argument('provider');p.add_argument('--task',required=True);p.add_argument('--query',default='');p.add_argument('--limit',type=int,default=20)
    p=sub.add_parser('source-inspect',help='Inspect one provider model/release without downloading')
    remote_options(p);p.add_argument('provider');p.add_argument('identifier');p.add_argument('--revision')
    p=sub.add_parser('import-link',help='Normalize a permitted model page/link and inspect recognized providers')
    remote_options(p);p.add_argument('link')
    p=sub.add_parser('acquire',help='Plan and start a durable model acquisition')
    remote_options(p);p.add_argument('name');p.add_argument('provider');p.add_argument('identifier')
    p.add_argument('--task',required=True);p.add_argument('--backend',required=True);p.add_argument('--revision');p.add_argument('--device',default='auto')
    p.add_argument('--dtype',default='auto',choices=['auto','float32','float16','bfloat16']);p.add_argument('--graph');p.add_argument('--model-file');p.add_argument('--mmproj');p.add_argument('--variant')
    p.add_argument('--replace',action='store_true');p.add_argument('--idempotency-key');p.add_argument('--wait',action='store_true');p.add_argument('--wait-timeout',type=float)
    p.add_argument('--activate',action='store_true');p.add_argument('--no-load',action='store_true');p.add_argument('--accept-license-sha256')
    p.add_argument('--runtime-profile',help='Require this prepared managed native profile when activating')
    p=sub.add_parser('acquisitions',help='List, inspect, cancel, retry, wait for or activate model acquisitions')
    remote_options(p);p.add_argument('action',choices=['list','get','cancel','retry','wait','activate','delete-weights']);p.add_argument('id',nargs='?')
    p.add_argument('--wait-timeout',type=float);p.add_argument('--accept-license-sha256');p.add_argument('--no-load',action='store_true')
    p.add_argument('--runtime-profile',help='Require this prepared managed native profile when activating')
    p=sub.add_parser('runtime-prepare',help='Plan or start a server-owned native runtime setup')
    remote_options(p);p.add_argument('profile');p.add_argument('--plan',action='store_true');p.add_argument('--wait',action='store_true');p.add_argument('--wait-timeout',type=float)
    p=sub.add_parser('runtime-setups',help='List, inspect, cancel, retry or wait for native runtime setup')
    remote_options(p);p.add_argument('action',choices=['list','get','cancel','retry','wait']);p.add_argument('id',nargs='?');p.add_argument('--wait-timeout',type=float)
    p=sub.add_parser('runtime-profiles',help='List packaged runtime profiles compatible with this server host');remote_options(p)
    p=sub.add_parser('assignment-devices',help='List stable local device identities available for assignments');remote_options(p)
    p.add_argument('--model');p.add_argument('--refresh',action='store_true')
    p=sub.add_parser('assignment',help='Read or save one model local device/parallelism assignment');remote_options(p)
    p.add_argument('action',choices=['get','save']);p.add_argument('model');p.add_argument('--mode',choices=['single','model_sharding','tensor_parallel'],default='single')
    p.add_argument('--group',action='append',help='Ordered stable GPU-/HIP- groups or one XPU-/OV- UUID; repeat for replicas')
    p.add_argument('--policy',choices=['auto','cpu'],help='Explicitly change automatic/CPU intent with no pinned groups')
    p.add_argument('--tensor-parallel-size',type=int,default=2);p.add_argument('--threads',type=int);p.add_argument('--worker-timeout',type=float)
    p = sub.add_parser("mcp", help="Expose the standalone server to an MCP agent over stdio")
    remote_options(p)
    p.add_argument("--transport", choices=["stdio"], default="stdio")
    p = sub.add_parser("hivemind", help="HiveMind gateway operations for a running Titanos (register-service)")
    hive = p.add_subparsers(dest="hivemind_action", required=True)
    # allow_abbrev=False: '--key VALUE' must not be read as '--key-env VALUE' (a secret typed there would be treated as a name)
    p = hive.add_parser("register-service", allow_abbrev=False, help="POST /register-service {job_type, port} to a HiveMind gateway, with a bearer when a key is set")
    p.add_argument("gateway", type=gateway_url_arg, help="Gateway URL (http(s), no credentials, query or fragment)")
    p.add_argument("--job-type", required=True, help="An existing gateway job type (1-64 characters [A-Za-z0-9._-])")
    p.add_argument("--port", type=int, help="The Titanos port the gateway will call (default: the port of --url)")
    p.add_argument("--url", metavar="URL", help="The Titanos server whose readiness is probed first (default TITANOS_URL, then http://127.0.0.1:11435)")
    p.add_argument("--key-env", metavar="VAR", help="Environment variable holding the gateway bearer (default: TITANOS_API_KEY when set); the option names the variable, never a secret value")
    p.add_argument("--attempts", type=int, default=300, help="Readiness probes (0.1 s apart) to wait for before registering (default 300)")
    p = sub.add_parser("service", help="Install, remove or inspect the host service that runs titanos serve (Scheduled Task, systemd, launchd)")
    svc = p.add_subparsers(dest="service_action", required=True)
    for action in ("install", "start", "stop", "uninstall", "status"):
        q = svc.add_parser(action, help={"install": "Write the unit and launcher under TITANOS_HOME/service, register and start it, wait for GET /healthz",
                                         "start": "Start the registered unit and wait for GET /healthz",
                                         "stop": "Stop the server (Windows: the task and the server process recorded in the pid file)",
                                         "uninstall": "Stop the server and unregister the unit (TITANOS_HOME files stay)",
                                         "status": "Report whether the unit is registered and the server is running"}[action])
        q.add_argument("--name", default="titanos", help="Unit / task name (default titanos)")
        q.add_argument("--system", action="store_true", help="System-wide unit (Linux /etc/systemd/system, macOS LaunchDaemons, Windows ONSTART as SYSTEM); needs elevation")
        q.add_argument("--dry-run", action="store_true", help="Print the files and commands without writing or running anything")
        q.add_argument("--json", action="store_true", help="Print the plan or status as JSON")
        if action in ("install", "start"):
            q.add_argument("--wait", type=float, default=None, metavar="SECONDS", help="Wait this long for GET /healthz to answer after starting (default 30; 0 skips the probe); exit 3 when it does not answer")
        if action == "install":
            q.add_argument("--host", help="TITANOS_HOST for the service (default: the current environment)")
            q.add_argument("--port", type=int, help="TITANOS_PORT for the service (default: the current environment)")
            q.add_argument("--role", choices=["standalone", "broker", "managed"], help="TITANOS_ROLE for the service")
            q.add_argument("--executable", help="The titanos executable the unit runs (default: the one beside this interpreter)")
    from .broker.cli import COMMANDS as broker_commands, register_cli
    register_cli(sub)
    args = parser.parse_args(argv)
    path = config_path(args.config)
    try:
        if args.command == "settings":
            from .settings import load_settings
            result = load_settings().describe()
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                for row in result["settings"].values():
                    print(f"{row['env']}={row['value']} (default: {row['default']})")
        elif args.command == "mcp":
            from .mcp_server import run_stdio
            run_stdio(args.url)
        elif args.command == "hivemind":
            hivemind_register_service(args)
        elif args.command == "service":
            from .service import run_service_command
            code = run_service_command(args, config_path=path)
            if code:
                return sys.exit(code)
        elif args.command == "broker" and args.broker_action == "live-smoke":
            from .broker.cli import live_smoke_command
            code = subprocess.run(live_smoke_command(args), check=False).returncode
            if code:
                return sys.exit(code)
        elif args.command in broker_commands:
            from .broker.cli import run_cli
            code = run_cli(args, path)
            if code:
                return sys.exit(code)
        elif args.command in {"run", "models", "info", "embed", "transcribe", "speak", "image", "edit", "video", "music", "jobs", "translate", "translate-audio", "ocr", "detect", "depth", "segment", 'sentiment', 'sentiment-compose', 'rerank', 'vad', 'diarize', 'analyze-audio', 'retrieval-answer', 'classify-pairs',
                              'providers','discover','source-inspect','import-link','acquire','acquisitions',
                              'runtime-prepare','runtime-setups','runtime-profiles','assignment-devices','assignment'}:
            remote_command(args)
        elif args.command in {"search", "inspect"}:
            from .hub import search, inspect
            if args.command == "search":
                for item in search(args):
                    print(json.dumps(item), flush=True)
            else:
                print(json.dumps(inspect(args), indent=2))
        elif args.command in {"login", "logout"}:
            from huggingface_hub import login, logout
            from .hub import hub_errors
            with hub_errors():
                if args.command == "login":
                    from getpass import getpass
                    token = getpass("Hugging Face token (optional; blank cancels): ").strip()
                    if token:
                        login(token=token, add_to_git_credential=False)
                else:
                    logout()
        elif args.command == 'devices':
            from .devices import inventory
            print(json.dumps(inventory(args.refresh,args.backend), indent=2))
        elif args.command == 'pin-devices':
            from .devices import inventory, pin, cuda_uuid
            torch = require_torch('cuda')
            if torch.version.hip:
                raise ValueError('pin-devices currently supports NVIDIA identities only')
            cfg = read_config(path)
            original = cfg.model_dump()
            devices = [d for d in inventory(True)['devices'] if d['present']]
            for worker in cfg.workers.values():
                if worker.gpus:
                    worker.gpus = pin(worker.gpus, devices)
                elif worker.device.startswith('cuda'):
                    import torch
                    from .hardware import select_device
                    worker.gpus = [cuda_uuid(torch, select_device(worker.device))]
                    worker.device = 'cpu'
            for spec in cfg.models.values():
                if spec.device.startswith('cuda'):
                    from .hardware import select_device
                    import torch
                    spec.device = cuda_uuid(torch, select_device(spec.device))
            def save_pins(current):
                if current.model_dump() != original:
                    raise ValueError('Registry changed during device discovery; run pin-devices again')
                current.models, current.workers = cfg.models, cfg.workers
            update_config(path, save_pins)
            print(f'Pinned current NVIDIA identities in {path}; restart serve')
        elif args.command == "prepare":
            from .prepare import prepare
            print(json.dumps(prepare(path, args), indent=2))
        elif args.command == 'capabilities':
            from .capabilities import describe
            cfg = read_config(path)
            if args.name not in cfg.models:
                raise ValueError(f'Unknown model {args.name!r}')
            print(json.dumps(describe(cfg.models[args.name]), indent=2))
        elif args.command in {"worker", "remove-worker"}:
            cfg = read_config(path)
            if args.command == "remove-worker":
                if args.name not in cfg.workers:
                    raise ValueError(f"Unknown worker {args.name!r}")
            else:
                if args.name in cfg.workers and not args.replace:
                    raise ValueError("Worker already exists; use --replace to change its assignment")
                gpus = args.gpus
                if gpus:
                    from .devices import scan,admit_gpu_uuids,probe_visible_cuda,hip_records,require_homogeneous,normalized_uuid
                    if all(item.startswith('HIP-') for item in gpus):
                        records=hip_records();gpus=admit_gpu_uuids(gpus,family='hip',visible=records)
                    else:
                        if not (all(item.startswith('GPU-') for item in gpus) or all(item.isdecimal() for item in gpus)):
                            raise ValueError('Use full GPU-/HIP- UUIDs from one family, or current CUDA-visible indices')
                        records=scan();visible=None;library_dir=None
                        mask=os.environ.get('CUDA_VISIBLE_DEVICES')
                        needs_probe=all(item.isdecimal() for item in gpus) or mask is not None and mask.strip() not in {'','-1'} and not all(item.strip().startswith('GPU-') and normalized_uuid(item.strip()) for item in mask.split(','))
                        if needs_probe:
                            native_models=[cfg.models[name] for name in args.models if name in cfg.models and cfg.models[name].backend=='llama-cpp']
                            if native_models:
                                if len(args.models)!=1:raise ValueError('A native worker names exactly one model')
                                from .native_install import executable_path
                                library_dir=executable_path(native_models[0].native.executable).parent
                            visible=probe_visible_cuda(library_dir)
                            if all(item.isdecimal() for item in gpus):
                                if any(int(item)>=len(visible) for item in gpus):raise ValueError('CUDA-visible index is unavailable; no driver-index fallback was used')
                                gpus=[visible[int(item)]['uuid'] for item in gpus]
                        gpus=admit_gpu_uuids(gpus,physical=records,visible=visible)
                    if len(gpus)>1:require_homogeneous([row for identity in gpus for row in records if row.get('uuid')==identity])
                    args.device='cuda'
                elif args.device.startswith('cuda'):
                    raise ValueError('Use --gpus for NVIDIA workers so assignments can be pinned to GPU UUIDs')
                if args.device.startswith('xpu') and args.mode!='tensor_parallel':
                    from .devices import pin_runtime
                    args.device=pin_runtime(args.device,'pytorch')
                assignment = WorkerSpec(models=args.models, device=args.device, gpus=gpus,
                    mode=args.mode, tensor_parallel_size=args.tensor_parallel_size, profile=args.profile,
                    **{key: value for key, value in {'threads': args.threads, 'timeout': args.timeout}.items() if value is not None})
            def save_worker(current):
                if args.command == "remove-worker":
                    if args.name not in current.workers:
                        raise ValueError(f"Unknown worker {args.name!r}")
                    del current.workers[args.name]
                else:
                    if args.name in current.workers and not args.replace:
                        raise ValueError("Worker already exists; use --replace to change its assignment")
                    current.workers[args.name] = assignment
            update_config(path, save_worker)
            print(f"Saved {path}; restart serve to apply worker assignments")
        elif args.command == 'native-install':
            from .native_install import install,plan
            from .runtime_catalog import resolve_catalog
            print(json.dumps((plan if args.plan else install)(resolve_catalog(args.catalog),args.profile,args.target),indent=2))
        elif args.command == 'upgrade':
            from .hardware import install_main
            options=['--catalog',args.catalog,'--profile',args.profile]
            if args.target:options+=['--target',str(args.target)]
            if args.gpu is not None:options+=['--gpu',str(args.gpu)]
            if args.plan:options+=['--plan']
            install_main(options)
        elif args.command == "init":
            if path.exists():
                read_config(path)
                print(f"Registry already exists: {path}")
            else:
                update_config(path, lambda current: None)
                print(f"Created {path}")
        elif args.command in {"add", "pull", "remove"}:
            cfg = read_config(path) if path.exists() else Config()
            if args.command == "remove":
                if args.name not in cfg.models:
                    raise ValueError(f"Unknown model {args.name!r}")
                if any(args.name in worker.models for worker in cfg.workers.values()):
                    raise ValueError("Remove this model from its worker assignments before removing its registration")
            else:
                if args.name in cfg.models and not args.replace:
                    raise ValueError("Model already exists; use --replace to change its path")
                native = option_json(args.native_options, "--native-options") if args.native_options else {}
                native.update({k:v for k,v in {"model_file":args.model_file,"mmproj":args.mmproj,"executable":args.runtime}.items() if v is not None})
                spec = ModelSpec(path=args.source, task=args.task, native=NativeOptions(**native) if native else None,
                                 backend=args.backend or ("diffusers" if args.task in {"image", "video"} else "transformers"),
                                 device=args.device, dtype=args.dtype, allow_pickle=args.allow_pickle,
                                 parallelism=args.parallelism, tensor_parallel_size=args.tensor_parallel_size,
                                 trust_remote_code=args.trust_remote_code, graph=args.graph,
                                 allow_cpu_fallback=args.allow_cpu_fallback,
                                 load_options=option_json(args.provider_options, "--provider-options") if args.provider_options else {},
                                 defaults=option_json(args.defaults, "--defaults") if args.defaults else {},
                                 voices=option_json(args.voices, "--voices") if args.voices else {})
                if spec.device.startswith('cuda'):
                    torch = require_torch(spec.device)
                    if not torch.version.hip:
                        from .hardware import select_device
                        from .devices import cuda_uuid
                        spec.device = cuda_uuid(torch, select_device(spec.device))
                    else:
                        from .devices import pin_runtime
                        spec.device=pin_runtime(spec.device,'pytorch')
                elif spec.device.startswith('xpu'):
                    require_torch(spec.device)
                    from .devices import pin_runtime
                    spec.device=pin_runtime(spec.device,'pytorch')
                elif spec.backend in {'openvino','openvino-genai'} and spec.device.upper().startswith(('GPU','NPU')):
                    from .devices import pin_runtime
                    spec.device=pin_runtime(spec.device.upper(),'openvino')
                # Validate identifiers before starting a potentially large download.
                Config(models={args.name: spec})
                if args.command == "pull":
                    from .hub import pull
                    spec.path, spec.source_revision = pull(args)
                    spec.source_repo = args.source
                    print(f"Downloaded {args.source}@{spec.source_revision}")
                else:
                    spec.path = str(Path(args.source).expanduser().resolve())
                    if not Path(spec.path).is_dir():
                        raise ValueError("Model source must be an existing local checkpoint directory")
            def save_model(current):
                if args.command == "remove":
                    if args.name not in current.models:
                        raise ValueError(f"Unknown model {args.name!r}")
                    if any(args.name in worker.models for worker in current.workers.values()):
                        raise ValueError("Remove this model from its worker assignments before removing its registration")
                    del current.models[args.name]
                else:
                    if args.name in current.models and not args.replace:
                        raise ValueError("Model already exists; use --replace to change its path")
                    current.models[args.name] = spec
            update_config(path, save_model)
            print(f"Saved {path}. Reload via POST /api/models/reload or restart the server.")
        elif args.command == "list":
            cfg = read_config(path)
            print(json.dumps({name: {"task": m.task, "backend": m.backend, "path": m.path, "device": m.device} for name, m in cfg.models.items()}, indent=2))
        elif args.command == "serve":
            import uvicorn; from . import health_listener, log_rotation
            from .settings import load_settings
            settings = load_settings(host=args.host, port=args.port, threads=args.threads,
                                     limit_concurrency=args.limit_concurrency, keep_alive_timeout=args.keep_alive_timeout,
                                     log_level=args.log_level, role=args.role)
            key = os.environ.get("TITANOS_API_KEY")
            # role preconditions (broker.role.check_serve_preconditions); a keyless network bind keeps the message
            # "Set TITANOS_API_KEY before binding to a network interface"
            from .broker.cli import serve_preconditions
            for note in serve_preconditions(settings, bootstrap_key=key,
                                            explicit_port=args.port is not None or "TITANOS_PORT" in os.environ):
                print("titanos: " + note, file=sys.stderr)
            if args.job_type and not args.register_with_gateway:
                raise ValueError("--job-type is used only with --register-with-gateway")
            if args.register_with_gateway:
                if settings.role != "managed":
                    raise ValueError("--register-with-gateway is available only with --role managed")
                if not args.job_type:
                    raise ValueError("--register-with-gateway needs --job-type (an existing gateway job type)")
            count = settings.threads
            os.environ.setdefault("OMP_NUM_THREADS", str(count))
            os.environ.setdefault("MKL_NUM_THREADS", str(count))
            os.environ['TITANOS_THREADS'] = str(count)
            ensure_username_environment()
            from .server import create_app
            cfg = read_config(path)
            if any(m.device.startswith('cuda') for m in cfg.models.values()):
                import torch
                if not torch.version.hip:
                    raise ValueError('Saved CUDA indices require migration: run titanos pin-devices before serve')
            tls = {'ssl_certfile': settings.tls_certfile, 'ssl_keyfile': settings.tls_keyfile} if settings.tls_certfile and settings.tls_keyfile else {}
            local_host='127.0.0.1' if settings.host in {'0.0.0.0','::'} else settings.host
            if ':' in local_host:local_host='['+local_host+']'
            local_url=('https' if tls else 'http')+f'://{local_host}:{settings.port}'
            if args.open:
                import threading,time,webbrowser
                def open_browser():
                    import httpx
                    for _ in range(100):
                        try:
                            # the local health probe of our own server; a self-signed certificate is not verified here
                            if httpx.get(local_url+'/healthz',timeout=1,trust_env=False,verify=not tls).is_success:
                                webbrowser.open(local_url);return
                        except httpx.HTTPError:pass
                        time.sleep(.1)
                threading.Thread(target=open_browser,daemon=True).start()
            if args.register_with_gateway:
                import threading
                # the bearer is this server's bootstrap key when set (a gateway on another host needs one)
                threading.Thread(target=register_with_gateway, args=(args.register_with_gateway, args.job_type, settings.port,
                                 local_url+'/v1/health/ready'), kwargs={'key': key}, daemon=True).start()
            # proxy headers are pinned: honoured only when the operator attests a TLS terminator on a loopback bind, and
            # only from 127.0.0.1, so FORWARDED_ALLOW_IPS cannot let a LAN client spoof scheme or peer
            pid_file = getattr(args, 'pid_file', None); uvicorn = health_listener.with_health(uvicorn, settings.health_port, **log_rotation.uvicorn_options(settings))  # service.py: pid file
            if pid_file is not None:
                pid_file.parent.mkdir(parents=True, exist_ok=True)
                pid_file.write_text(str(os.getpid()), encoding='utf-8')
                import signal
                # after its graceful shutdown uvicorn re-raises the stop signal with the handler it found installed; the
                # default action of SIGBREAK (Windows) and SIGTERM ends the process before the finally below could remove
                # the pid file, so those become a KeyboardInterrupt, which uvicorn.run absorbs
                for signal_name in ('SIGBREAK', 'SIGTERM'):
                    if hasattr(signal, signal_name):
                        signal.signal(getattr(signal, signal_name), signal.default_int_handler)
            try:
                uvicorn.run(create_app(cfg, key, settings=settings, registry_path=path,
                                       runtime_catalog=args.native_catalog), host=settings.host, port=settings.port, workers=1,
                            limit_concurrency=settings.limit_concurrency, timeout_keep_alive=settings.keep_alive_timeout,
                            proxy_headers=settings.behind_tls_proxy, forwarded_allow_ips='127.0.0.1', **tls,
                            log_level=settings.log_level)
            finally:
                if pid_file is not None:
                    pid_file.unlink(missing_ok=True)
        elif args.command == "doctor":
            from .hardware import doctor
            options = json.loads(args.provider_options.read_text(encoding='utf-8')) if args.provider_options else None
            if options is not None and not isinstance(options,dict):
                raise ValueError('Provider options must be a JSON object')
            result = doctor(args.device, args.check, args.backend, options)
            text = json.dumps(result, indent=2)
            if args.output:
                args.output.write_text(text + "\n", encoding="utf-8")
            print(text)
            if args.check and not result.get("check", {}).get("passed"):
                return sys.exit(1)
        elif args.command == "ps":
            print(json.dumps(client_request(args, "GET", "/api/models/loaded"), indent=2))
        elif args.command == "workers":
            print(json.dumps(client_request(args, "GET", "/api/workers"), indent=2))
        elif args.command == "details":
            print(json.dumps(client_request(args, "POST", "/api/models/details", json={"model": args.name, "worker": args.worker}), indent=2))
        elif args.command in {"load", "unload"}:
            print(json.dumps(client_request(args, "POST", f"/api/models/{args.command}", json={"model": args.name, "worker": args.worker}), indent=2))
    except (ValueError, OSError, ImportError, KeyError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"titanos: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "titanos: interrupted\n")
    except EOFError:
        return
