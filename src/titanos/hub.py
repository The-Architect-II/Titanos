"""Hub discovery and pinned downloads; inference remains entirely local."""

import fnmatch
import os
import json
import shutil
from contextlib import contextmanager, suppress
from pathlib import Path

from huggingface_hub import HfApi, get_token, snapshot_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError, RevisionNotFoundError
from requests.exceptions import RequestException

from .link_policy import is_link_path
from .providers import check_case_insensitive_collisions, literal_pattern, validate_acquisition_path

# Metadata siblings that accompany a chosen GGUF; matched case-insensitively (the lowercased name against the lowercased
# pattern, on every platform, so Readme.md, License.txt and CONFIG.JSON are included as the previous Windows-only folding
# did) against the sibling list, so the resulting plan is an explicit list of existing names, never a glob handed to
# the downloader; every resolved name still passes the acquisition rule and the collision checks.
METADATA_PATTERNS = ('*.json', '*.jinja', '*LICENSE*', 'README*')
# huggingface_hub keeps its own download metadata (.metadata/.lock files that steer resume and skip decisions) under
# <local_dir>/.cache; the post-download walk leaves that subtree alone, so no repository file may be planned into it.
HF_RESERVED_ROOT = '.cache'
# Weight formats the default (transformers) pull never fetches; pickle formats are added unless --allow-pickle.
UNSUPPORTED_WEIGHT_PATTERNS = ('*.h5', '*.msgpack', '*.onnx', '*.tflite', '*.ot', '*.gguf')
PICKLE_WEIGHT_PATTERNS = ('*.bin', '*.pt', '*.pth', '*.ckpt')


def credential(args):
    if args.anonymous:
        return False
    if args.token_env:
        value = os.environ.get(args.token_env)
        if not value:
            raise ValueError(f"Token environment variable {args.token_env!r} is empty")
        return value
    return get_token() or False


@contextmanager
def hub_errors():
    # Do not echo provider exceptions: requests and credentials are external data.
    try:
        yield
    except GatedRepoError:
        raise ValueError("Gated model: obtain access on Hugging Face, then use login or HF_TOKEN.") from None
    except RevisionNotFoundError:
        raise ValueError("Hugging Face revision does not exist.") from None
    except RepositoryNotFoundError:
        raise ValueError("Repository unavailable: check its ID and optional token's access.") from None
    except HfHubHTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        raise ValueError(f"Hugging Face HTTP {status}; check access, rate limits, or retry later.") from None
    except RequestException:
        raise ValueError("Hugging Face connection failed; retry when connectivity is restored.") from None


def destination(args, info):
    """Plain per-revision directory for a pinned download. The Hugging Face cache layout links snapshots to blobs with
    symlinks, which a stock Windows account cannot create (WinError 1314 without Developer Mode); a plain directory needs
    no privilege, stays independently replaceable and is what the registry records."""
    from .config import home
    base = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else home() / "models"
    return base / f"{args.source.replace('/', '--')}@{info.sha[:12]}"


def _bounded(text):
    shown = repr(text)
    return shown if len(shown) <= 64 else shown[:61] + "..."


def sibling_names(info):
    return [item.rfilename for item in info.siblings or []]


def matching_siblings(files, patterns):
    """Sibling names that match any pattern case-insensitively on every platform (fnmatchcase over the lowercased name
    and pattern: explicit folding, no host-dependent ntpath/posixpath normcase)."""
    lowered = [pattern.lower() for pattern in patterns]
    return {name for name in files if any(fnmatch.fnmatchcase(name.lower(), pattern) for pattern in lowered)}


def downloader_fold(name):
    """The folding huggingface_hub's allow_patterns filter applies to both name and pattern on Windows (fnmatch.fnmatch ->
    ntpath.normcase): case, and '/' and '\\' as the same separator. Two sibling names with one folded form are
    selected by one literal pattern there."""
    return name.replace("\\", "/").lower()


def plan_selection(files, chosen):
    """Explicit, validated download plan: the chosen sibling names, sorted.

    Every name passes the shared Windows-safe acquisition rule before any downloader runs (drive-relative, stream,
    device, traversal, glob-metacharacter and case-hostile names are refused, naming one bounded component), and no
    name may lie in huggingface_hub's reserved '.cache' directory. Names that fold to a selected name are refused both
    among the selection and against unselected siblings: huggingface_hub filters allow_patterns with fnmatch, which on
    Windows folds case and treats '/' and '\\' alike, so such a sibling could otherwise be fetched over the planned file."""
    selected = sorted(set(chosen))
    if not selected:
        raise ValueError("Nothing to download: the repository selection is empty")
    for name in selected:
        validate_acquisition_path(name)
        if name.split("/", 1)[0] == HF_RESERVED_ROOT:
            raise ValueError("Repository file lies in the downloader's reserved .cache directory: " + _bounded(name))
    check_case_insensitive_collisions(selected)
    folded = {downloader_fold(name) for name in selected}
    unselected = set(files) - set(selected)
    for other in sorted(unselected):
        if downloader_fold(other) in folded:
            raise ValueError("Repository has an unselected file that differs only in case or path separator from a selected one: "
                             + _bounded(other))
    return selected


def literal_patterns(names):
    """allow_patterns that each match exactly one validated name (fnmatch metacharacters escaped)."""
    return [literal_pattern(name) for name in names]


def _is_link(path):
    return is_link_path(path)  # symlink or junction, on Python 3.11 too


def snapshot_files(target):
    """Relative POSIX names of regular files under target (huggingface_hub's .cache excluded), refusing any symlink or
    junction and any entry that resolves outside target."""
    target = Path(target)
    if not target.exists():
        return set()
    if _is_link(target):
        raise ValueError("Download directory is a link: " + _bounded(target.name))
    resolved = target.resolve()
    found = set()
    for dirpath, dirnames, filenames in os.walk(target, followlinks=False):
        here = Path(dirpath)
        if here == target:
            dirnames[:] = [name for name in dirnames if name != ".cache"]
        for name in [*dirnames, *filenames]:
            entry = here / name
            relative = entry.relative_to(target).as_posix()
            if _is_link(entry):
                raise ValueError("Downloaded snapshot contains a symbolic link or junction: " + _bounded(relative))
            if not entry.resolve().is_relative_to(resolved):
                raise ValueError("Downloaded snapshot entry resolves outside the download directory: " + _bounded(relative))
        found.update((here / name).relative_to(target).as_posix() for name in filenames)
    return found


def _inventory(target):
    """Every entry under target as it stands (files and directories, huggingface_hub's .cache included, links listed but
    never followed), as relative POSIX names: the record that tells one pull's writes apart from what was there before."""
    files, directories = set(), set()
    for dirpath, dirnames, filenames in os.walk(target, followlinks=False):
        here = Path(dirpath)
        # os.walk(followlinks=False) still descends into a junction on Windows: prune links, list them as entries.
        dirnames[:] = [name for name in dirnames if not _is_link(here / name)]
        directories.update((here / name).relative_to(target).as_posix() for name in os.listdir(here)
                           if (here / name).is_dir())
        files.update((here / name).relative_to(target).as_posix() for name in filenames)
    return files, directories


def _remove_introduced(target, before):
    """Remove what one refused pull wrote into a pre-existing directory: every file or link that was not there before and
    every directory it created (emptied ones pruned deepest-first). Links are unlinked in place and never walked into;
    nothing that existed before the pull is touched, so a pre-existing snapshot keeps its accepted contents."""
    files, directories = before
    created = []
    for dirpath, dirnames, filenames in os.walk(target, followlinks=False):
        here = Path(dirpath)
        for name in list(dirnames):
            entry = here / name
            if _is_link(entry):
                dirnames.remove(name)  # never descended: the link's contents are not this directory's
                if entry.relative_to(target).as_posix() not in directories:
                    with suppress(OSError):
                        os.unlink(entry)  # removes the link itself (directory symlink or junction included on Windows)
            elif entry.relative_to(target).as_posix() not in directories:
                created.append(entry)
        for name in filenames:
            entry = here / name
            if entry.relative_to(target).as_posix() not in files:
                with suppress(OSError):
                    entry.unlink()
    for entry in reversed(created):
        with suppress(OSError):
            entry.rmdir()  # only an emptied directory this pull created goes; a populated one stays


def fetch_snapshot(repo_id, revision, token, target, selected):
    """Fetch exactly the planned sibling names into the plain directory `target` and verify the result.

    The plan is passed as literal-escaped allow_patterns; afterwards the directory is walked and any file the plan did
    not name (other than files that already existed there before this pull, e.g. another quantization of the same
    revision), any link, or any entry outside the directory is refused. On refusal, a directory this pull created is
    removed and, in a pre-existing directory, every file, link and directory this pull introduced is removed, so a
    refused write neither stays on disk nor becomes 'pre-existing' for the next pull of the same revision."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    existed = target.exists()
    before = snapshot_files(target) if existed else set()
    inventory = _inventory(target) if existed else (set(), set())
    snapshot_download(repo_id, revision=revision, token=token, local_dir=str(target),
                      allow_patterns=literal_patterns(selected))
    try:
        planned = set(selected) | before
        unplanned = sorted(name for name in snapshot_files(target) if name not in planned)
        if unplanned:
            raise ValueError("Downloaded snapshot contains a file the plan did not name: " + _bounded(unplanned[0]))
    except ValueError:
        if not existed:
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
        else:
            _remove_introduced(target, inventory)
        raise
    return str(target)


def download(args, info, token, selected):
    """`titanos pull`'s fetch: the planned names into the plain per-revision directory (see fetch_snapshot)."""
    return fetch_snapshot(args.source, info.sha, token, destination(args, info), selected)


def describe(info):
    return {"id": info.id, "task": info.pipeline_tag, "library": info.library_name,
            "downloads": info.downloads, "likes": info.likes, "gated": info.gated,
            "revision": info.sha, "titanos_compatibility": "unverified"}


def search(args):
    if args.limit < 1:
        raise ValueError("limit must be positive")
    token = credential(args)
    with hub_errors():
        # SDK follows Hub pagination lazily; --all never materializes the catalog.
        for info in HfApi(token=token).list_models(
            search=args.query or None, author=args.author, pipeline_tag=args.task,
            filter=args.library, sort=args.sort, direction=-1,
            limit=None if args.all else args.limit, full=True,
        ):
            yield describe(info)


def inspect(args):
    with hub_errors():
        info = HfApi(token=credential(args)).model_info(args.repo, revision=args.revision)
        return {**describe(info), "files": sibling_names(info)}


def pull(args):
    token = credential(args)
    with hub_errors():
        info = HfApi(token=token).model_info(args.source, revision=args.revision)
        if not info.sha:
            raise ValueError("Hub did not resolve an immutable model revision")
        files = sibling_names(info)
        if args.backend == 'llama-cpp':
            options = json.loads(args.native_options.read_text()) if args.native_options else {}
            selected = args.model_file or options.get('model_file')
            projector = args.mmproj or options.get('mmproj')
            if not selected or selected not in files or (projector and projector not in files):
                raise ValueError('Requested GGUF model/projector is absent; inspect the repository and choose --model-file')
            # Pull only the chosen quantization, its split shards, and the projector.
            import re
            shard = re.fullmatch(r'(.+)-\d{5}-of-(\d{5})\.gguf', selected)
            chosen = {selected, *([projector] if projector else [])}
            if shard:
                chosen.update(f'{shard[1]}-{i:05d}-of-{shard[2]}.gguf' for i in range(1, int(shard[2])+1))
            if not chosen <= set(files):
                raise ValueError('Incomplete split GGUF checkpoint')
            # Metadata siblings resolved against the repository, so the plan names only existing files.
            plan = plan_selection(files, chosen | matching_siblings(files, METADATA_PATTERNS))
            path = download(args, info, token, plan)
            return path, info.sha
        if args.backend == "onnx-diffusers":
            if args.task != "image" or args.graph is not None:
                raise ValueError("ONNX images require task=image and a component directory, not --graph")
            from .onnx_images import COMPONENTS
            required = {name + "/model.onnx" for name in COMPONENTS} | {"model_index.json",
                "scheduler/scheduler_config.json", "tokenizer/tokenizer_config.json",
                "tokenizer/merges.txt", "tokenizer/vocab.json", "feature_extractor/preprocessor_config.json"}
            if required - set(files):
                raise ValueError("Prepared ONNX image export is incomplete: " + ", ".join(sorted(required - set(files))))
            # Every sibling, explicitly: the required set and metadata plus the graphs' external tensor data.
            path = download(args, info, token, plan_selection(files, files))
            return path, info.sha
        if args.backend in {"onnx", "openvino", "openvino-genai"}:
            filenames = set(files)
            if args.backend == 'openvino-genai' and args.task == 'image':
                if args.graph is not None:
                    raise ValueError('GenAI images use a complete exported component directory; omit --graph')
                required = {'model_index.json', 'scheduler/scheduler_config.json',
                            'tokenizer/openvino_tokenizer.xml', 'tokenizer/openvino_tokenizer.bin'}
                required.update(f'{component}/{name}' for component in ('text_encoder', 'unet', 'vae_decoder')
                                for name in ('config.json', 'openvino_model.xml', 'openvino_model.bin'))
                missing = required - filenames
                if missing:
                    raise ValueError('Prepared SD/SDXL image export is incomplete: ' + ', '.join(sorted(missing)))
            else:
                graph = args.graph or ('model.onnx' if args.backend == 'onnx' else 'openvino_model.xml')
                if graph not in filenames:
                    raise ValueError(f"Prepared graph {graph!r} is absent; inspect files and use --graph, or prepare the model first")
            # Full prepared snapshots, named explicitly, so external tensors and tokenizers stay together.
            path = download(args, info, token, plan_selection(files, files))
            return path, info.sha
        ignored = UNSUPPORTED_WEIGHT_PATTERNS if args.allow_pickle else UNSUPPORTED_WEIGHT_PATTERNS + PICKLE_WEIGHT_PATTERNS
        extensions = (".safetensors",) if not args.allow_pickle else (".safetensors", ".bin", ".pt", ".pth", ".ckpt")
        if not any(name.endswith(extensions) for name in files):
            raise ValueError("No supported checkpoint weights found; inspect the repository format or explicitly allow trusted legacy weights.")
        # Every sibling except the excluded weight formats (matched on the lowercased name, so no host-dependent folding).
        kept = [name for name in files if not any(fnmatch.fnmatchcase(name.lower(), pattern) for pattern in ignored)]
        path = download(args, info, token, plan_selection(files, kept))
        return path, info.sha
