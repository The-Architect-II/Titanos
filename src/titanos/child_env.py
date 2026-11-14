"""Child-process environments without provider credentials. Standard library only: titanos.native_process imports this
under `python -I -S`, where no third-party package (and so no titanos.broker module) can be imported.

titanos.broker.redact re-exports PROVIDER_ENV_DENY, PROVIDER_ENV_DENY_SUFFIXES and child_env from here, so the deny
list has one definition."""
import json
import os
from pathlib import Path

# Provider credentials and base-URL overrides a child process must never inherit (native.py, native_process.py and
# devices.py child environments). Names ending in PROVIDER_ENV_DENY_SUFFIXES are dropped too.
PROVIDER_ENV_DENY = frozenset({
    'OPENAI_API_KEY', 'OPENAI_ORG_ID', 'OPENAI_ORGANIZATION', 'OPENAI_PROJECT_ID', 'OPENAI_BASE_URL', 'OPENAI_API_BASE',
    'AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_ENDPOINT', 'AZURE_OPENAI_AD_TOKEN', 'AZURE_API_KEY',
    'ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL',
    'GEMINI_API_KEY', 'GOOGLE_API_KEY', 'GOOGLE_GENAI_API_KEY',
    'XAI_API_KEY', 'GROQ_API_KEY', 'TOGETHER_API_KEY', 'TOGETHERAI_API_KEY', 'MISTRAL_API_KEY', 'DEEPSEEK_API_KEY',
    'FIREWORKS_API_KEY', 'PERPLEXITY_API_KEY', 'PPLX_API_KEY', 'NVIDIA_API_KEY', 'NGC_API_KEY', 'NIM_API_KEY',
    'CEREBRAS_API_KEY', 'SAMBANOVA_API_KEY', 'OPENROUTER_API_KEY', 'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN',
    'HUGGINGFACEHUB_API_TOKEN', 'ELEVENLABS_API_KEY', 'XI_API_KEY', 'OLLAMA_API_KEY', 'LITELLM_API_KEY',
    'LITELLM_MASTER_KEY', 'TITANOS_API_KEY',
})
PROVIDER_ENV_DENY_SUFFIXES = ('_BASE_URL', '_API_KEY')


def child_env(environ, *, extra_deny=()):
    """A copy of environ without PROVIDER_ENV_DENY names, names ending in _BASE_URL or _API_KEY, and extra_deny
    (compared case-insensitively, as Windows environment names are)."""
    deny = {name.upper() for name in PROVIDER_ENV_DENY} | {name.upper() for name in extra_deny}
    return {name: value for name, value in environ.items()
            if name.upper() not in deny and not name.upper().endswith(PROVIDER_ENV_DENY_SUFFIXES)}


def vault_env_vars(home=None):
    """Variables named by env: credential records in TITANOS_HOME/broker/vault/index.json, read as plain JSON (the
    same file titanos.broker.vault.store.Vault keeps). An absent or unreadable index names nothing. Any record that
    says source 'env' counts, whatever the index version: denying an extra name is the safe direction."""
    try:
        root = Path(os.environ.get('TITANOS_HOME', '~/.titanos') if home is None else home).expanduser().resolve()
        document = json.loads((root / 'broker' / 'vault' / 'index.json').read_text(encoding='utf-8'))
        records = document.get('credentials', {})
        return frozenset(record['env_var'] for record in records.values()
                         if isinstance(record, dict) and record.get('source') == 'env'
                         and isinstance(record.get('env_var'), str) and record['env_var'])
    except Exception:
        return frozenset()


# Names the engine's own children never get even when an allowlist or TITANOS_CHILD_ENV_ALLOW admits them: credential-shaped
# suffixes beyond PROVIDER_ENV_DENY_SUFFIXES (HF_AUTH_TOKEN, GITHUB_TOKEN, AWS_SECRET_ACCESS_KEY and the like).
CREDENTIAL_SUFFIXES = (*PROVIDER_ENV_DENY_SUFFIXES, '_TOKEN', '_SECRET', '_PASSWORD', '_SECRET_KEY', '_SECRET_ACCESS_KEY',
                       '_ACCESS_KEY_ID')
# What a native runtime or helper child may inherit: operating-system and locale basics, temporary and home directories,
# accelerator driver and visibility controls, thread-count knobs, Python and PyInstaller plumbing, offline-cache locations
# and Titanos's own TITANOS_* settings. TITANOS_CHILD_ENV_ALLOW adds names or NAME* prefixes (comma separated).
CHILD_ENV_ALLOW = frozenset({
    'PATH', 'PATHEXT', 'SYSTEMROOT', 'WINDIR', 'SYSTEMDRIVE', 'COMSPEC', 'OS', 'DRIVERDATA', 'NUMBER_OF_PROCESSORS',
    'PROCESSOR_ARCHITECTURE', 'PROCESSOR_ARCHITEW6432', 'PROCESSOR_IDENTIFIER', 'PROCESSOR_LEVEL', 'PROCESSOR_REVISION',
    'TEMP', 'TMP', 'TMPDIR', 'HOME', 'USERPROFILE', 'HOMEDRIVE', 'HOMEPATH', 'APPDATA', 'LOCALAPPDATA', 'PROGRAMDATA',
    'ALLUSERSPROFILE', 'PUBLIC', 'PROGRAMFILES', 'PROGRAMFILES(X86)', 'PROGRAMW6432', 'COMMONPROGRAMFILES',
    'COMMONPROGRAMFILES(X86)', 'COMMONPROGRAMW6432', 'USERNAME', 'USER', 'LOGNAME', 'USERDOMAIN', 'COMPUTERNAME',
    'HOSTNAME', 'SHELL', 'LANG', 'LANGUAGE', 'TZ', 'TERM', 'LD_LIBRARY_PATH', 'LD_LIBRARY_PATH_ORIG', 'DYLD_LIBRARY_PATH',
    'DYLD_FALLBACK_LIBRARY_PATH', 'VIRTUAL_ENV', 'SSL_CERT_FILE', 'SSL_CERT_DIR', 'HF_HOME', 'HF_HUB_CACHE',
    'HF_HUB_OFFLINE', 'HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_OFFLINE', 'TRANSFORMERS_CACHE', 'HF_DATASETS_OFFLINE',
    'TOKENIZERS_PARALLELISM', 'GPU_DEVICE_ORDINAL'})
CHILD_ENV_ALLOW_PREFIXES = ('LC_', 'XDG_', 'CUDA_', 'NVIDIA_', 'CUBLAS', 'CUDNN_', 'NCCL_', 'GGML_', 'HIP_', 'ROCR_',
                            'ROCM_', 'HSA_', 'ZE_', 'ONEAPI_', 'SYCL_', 'VK_', 'MESA_', 'OMP_', 'MKL_', 'KMP_', 'OPENBLAS_',
                            'NUMEXPR_', 'TORCH_', 'PYTORCH_', 'TRITON_', 'OV_', 'QNN_', 'ADSP_', 'PYTHON', '_PYI',
                            'IMAGEIO_', 'TITANOS_')


def scrubbed_env(environ, *, extra_deny=()):
    """child_env without CREDENTIAL_SUFFIXES names: the floor every child of the engine gets, including processes that
    may need variables no allowlist can know (a spawned replica worker, FFmpeg)."""
    return {name: value for name, value in child_env(environ, extra_deny=extra_deny).items()
            if not name.upper().endswith(CREDENTIAL_SUFFIXES)}


def allowlisted_env(environ, *, extra_deny=(), extra_allow=None):
    """Only CHILD_ENV_ALLOW names, CHILD_ENV_ALLOW_PREFIXES and the TITANOS_CHILD_ENV_ALLOW entries of environ (or
    extra_allow: names, or prefixes ending in *), then scrubbed_env, so a credential name is dropped even when allowed.
    Names compare case-insensitively, as Windows environment names do."""
    entries = [item.strip().upper() for item in (environ.get('TITANOS_CHILD_ENV_ALLOW', '').split(',')
                                                 if extra_allow is None else extra_allow) if item.strip()]
    names = {item for item in entries if not item.endswith('*')}
    prefixes = CHILD_ENV_ALLOW_PREFIXES + tuple(item[:-1] for item in entries if item.endswith('*') and len(item) > 1)
    kept = {name: value for name, value in environ.items()
            if name.upper() in CHILD_ENV_ALLOW or name.upper() in names or name.upper().startswith(prefixes)}
    return scrubbed_env(kept, extra_deny=extra_deny)


__all__ = ['CHILD_ENV_ALLOW', 'CHILD_ENV_ALLOW_PREFIXES', 'CREDENTIAL_SUFFIXES', 'PROVIDER_ENV_DENY',
           'PROVIDER_ENV_DENY_SUFFIXES', 'allowlisted_env', 'child_env', 'scrubbed_env', 'vault_env_vars']
