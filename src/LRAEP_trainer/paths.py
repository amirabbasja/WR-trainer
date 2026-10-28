import os
from pathlib import Path
import platform
DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / 'LRAEP_helpers'

def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / '.config'

def bundled_xray() -> Path | None:
    if platform.system() != 'Linux':
        return None
    machine = platform.machine().lower()
    machine = {'amd64': 'x86_64', 'arm64': 'aarch64'}.get(machine, machine)
    if machine not in {'x86_64', 'aarch64'}:
        return None
    executable = Path(__file__).resolve().parent / 'bin' / f'linux-{machine}' / 'xray'
    if executable.is_file() and os.access(executable, os.X_OK):
        return executable
    return None

def cache_dir() -> Path:
    override = os.environ.get('LRAEP_CACHE_DIR')
    if override:
        return Path(override).expanduser()
    base = Path(os.environ.get('XDG_CACHE_HOME') or Path.home() / '.cache')
    return base / 'LRAEP'

def artifacts_dir() -> Path:
    override = os.environ.get('LRAEP_DATA_DIR')
    if override:
        return Path(override).expanduser() / 'artifacts'
    base = Path(os.environ.get('XDG_DATA_HOME') or Path.home() / '.local' / 'share')
    return base / 'LRAEP' / 'artifacts'
