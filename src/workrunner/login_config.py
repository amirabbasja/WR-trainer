"""Read WR login data in a disposable process."""

from contextlib import contextmanager
import json
from pathlib import Path
import select
import subprocess
import sys
import time

from .config import ConfigError


_HELPER_TIMEOUT = 10
# Allow for JSON escaping and the WR frame's hex encoding.
_MAX_LOGIN_FRAME = 16 * 1024


def _endpoint(settings):
    return [settings['host'], settings['port'], settings.get('tls', False), settings.get('ca_file')]


def _command(action, path, *args):
    return [
        sys.executable, '-B', '-c',
        'import sys; sys.path.insert(0, sys.argv.pop(1)); '
        'from workrunner.login_config import _main; raise SystemExit(_main(sys.argv[1:]))',
        str(Path(__file__).resolve().parents[1]), action, str(Path(path).resolve()), *args,
    ]


def load_training_config(path):
    """Load validated settings without returning the login identities."""
    try:
        result = subprocess.run(_command('config', path), capture_output=True,
                                timeout=_HELPER_TIMEOUT, check=True)
    except subprocess.CalledProcessError as error:
        raise ConfigError(error.stderr.decode(errors='replace').strip() or
                          'Could not load training configuration') from None
    except subprocess.TimeoutExpired:
        raise ConfigError('Training configuration helper timed out') from None
    return json.loads(result.stdout)


@contextmanager
def login_frame(path, request_id, settings, worker, agent):
    """Yield a wire frame and overwrite its buffer on success or failure."""
    buffer = bytearray(_MAX_LOGIN_FRAME + 1)
    try:
        command = _command('login', path, str(request_id), worker, agent, json.dumps(_endpoint(settings)))
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              bufsize=0) as child:
            deadline = time.monotonic() + _HELPER_TIMEOUT
            try:
                count = 0
                with memoryview(buffer) as view:
                    while count < len(buffer):
                        remaining = max(0, deadline - time.monotonic())
                        if not select.select([child.stdout], [], [], remaining)[0]:
                            raise subprocess.TimeoutExpired(child.args, _HELPER_TIMEOUT)
                        with view[count:] as tail:
                            size = child.stdout.readinto(tail)
                        if not size:
                            break
                        count += size
                if count > _MAX_LOGIN_FRAME:
                    raise ConfigError('WR login frame exceeds its size limit')
                status = child.wait(timeout=max(0, deadline - time.monotonic()))
                if status == 2:
                    raise ConfigError('WR endpoint changed in the private config. Restart training to use it.')
                if status or not count:
                    raise ConfigError('Could not prepare WR login. Check S3.WR_S2 in the private config.')
            except subprocess.TimeoutExpired:
                raise ConfigError('WR login helper timed out') from None
            finally:
                if child.poll() is None:
                    child.kill()
        with memoryview(buffer)[:count] as frame:
            yield frame
    finally:
        buffer[:] = bytes(len(buffer))


def _main(args):
    import resource

    from .config import load_config, resolve_S3_settings

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    action, path, *login = args
    try:
        config = load_config(path)
        settings = config['S3']
        if action == 'config':
            settings.pop('WR_S2')
            settings.pop('KEX_uname')
            json.dump(config, sys.stdout)
        else:
            from .fixed_constants import neutral_protocol_text
            from .bundle_verify import verify_bundle
            from .training_constants import DEVICE_ID
            from .training_identity import validate_login_identity
            from .training_protocol import TrainingDialect
            from .wire_io import dumps_buffer

            request_id, worker, agent, endpoint = login
            settings = resolve_S3_settings(settings)
            if _endpoint(settings) != json.loads(endpoint):
                return 2
            settings[DEVICE_ID] = worker
            source = verify_bundle(config['official_source'])
            sys.path.insert(0, str(source / 'python'))
            validate_login_identity(settings, 'WR')
            dialect = TrainingDialect(settings['WR_S2'], worker, agent)
            with neutral_protocol_text():
                request, = dialect.auth_requests(int(request_id))
                with dumps_buffer(request, 'wr', framed=True) as frame:
                    sys.stdout.buffer.write(frame)
                    sys.stdout.buffer.flush()
    except (OSError, RuntimeError, ValueError) as error:
        if action == 'config' and isinstance(error, ConfigError):
            print(error, file=sys.stderr)
        return 1
    return 0
