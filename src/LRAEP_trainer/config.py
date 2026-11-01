import copy
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit
from .config_export import ExportError, decode_config_document, encode_config_document, read_open_text
from .effort import normalize_job_sequence
from .paths import artifacts_dir
from .wire_constants import S7_LEGACY_ALIAS

class ConfigError(RuntimeError):
    pass
_TOP_LEVEL = {'version', 'mode', 'official_source', 'device', 'training', 'S3', 'node', 'vpn', 'side_job', 'trainer_name', 'telegram'}
_TELEGRAM_DEFAULTS = {
    'enabled': False, 'tgBot_key': '', 'tgChatID': '',
    'update_interval_seconds': 300,
}
_TRAINING_DEFAULTS = {'m': 256, 'n': 256, 'k': 2048, 'tile_h': 8, 'tile_w': 8, 'max_batches': 0, 'max_seconds': 0, 'job_sequence': [['main', 60, 80], ['main', 120, 70], ['main', 180, 10]], 'output_dir': str(artifacts_dir())}
_S3_DEFAULTS = {'host': '', 'port': 7048, 'tls': False, 'ca_file': None, 'WR_S2': '', 'KEX_uname': '', 'worker': 'rig-01', 'password': 'x', 'connect_timeout': 15, 'read_timeout': 60, 'reconnect_delay': 3, 'protocol': 'WR', 'compression': 'none', 'certificate_version': None, 'agent': 'LRAEP-trainer/0.1', 'stop_on_unknown_request': True, 'unknown_request_dir': None, 'protocol_documentation': None}
_S3_PROFILES = ('KEX', 'WR')
_S3_CONNECTION_FIELDS = {'host', 'port', 'tls', 'ca_file', 'compression', 'certificate_version', 'protocol_documentation'}
_NODE_DEFAULTS = {'rpc_url': 'https://127.0.0.1:44107', 'username': '', 'password': '', 'ca_file': None, 'poll_interval': 2, 'refresh_interval': 15, 'rpc_timeout': 15, 'address': ''}
_VPN_DEFAULTS = {'enabled': True, 'link': '', 'core_path': None, 'startup_timeout': 10, 'links_file': 'links.txt', 'check_url': 'https://api.ipify.org'}
_SIDE_JOB_DEFAULTS = {'repo_url': '', 'channels': [64, 128, 256, 512], 'blocks_per_stage': [2, 2, 2, 2], 'hidden_size': 512, 'image_size': 224, 'batch_size': 16, 'prefetch_batches': 2, 'passes_per_shard': 1, 'cache_limit_mb': 2048, 'learning_rate': 0.001, 'weight_decay': 0.0001, 'epochs': 0, 'use_amp': True, 'seed': 42, 'vpn': {'enabled': False, 'link': '', 'core_path': None, 'startup_timeout': 10}}
DEFAULT_CONFIG = {'version': 1, 'mode': 'S3', 'official_source': 'src/LRAEP_helpers', 'device': 'cuda:0', 'training': copy.deepcopy(_TRAINING_DEFAULTS), 'S3': dict(_S3_DEFAULTS), 'node': dict(_NODE_DEFAULTS), 'vpn': dict(_VPN_DEFAULTS), 'side_job': copy.deepcopy(_SIDE_JOB_DEFAULTS), 'trainer_name': 'LRAEP-WR-Trainer', 'telegram': dict(_TELEGRAM_DEFAULTS)}

def _object_without_duplicates(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ConfigError('configuration contains a duplicate field')
        result[name] = value
    return result

def _reject_constant(_):
    raise ConfigError('configuration numbers must be finite')

def _section(value, allowed, section):
    if not isinstance(value, dict):
        raise ConfigError(f'{section} must be an object')
    if value.keys() - allowed:
        raise ConfigError(f'{section} contains an unknown field')
    return value

def _string(value, field, *, nonempty=False, controls=False):
    if not isinstance(value, str):
        raise ConfigError(f'{field} must be a string')
    if nonempty and (not value.strip()):
        raise ConfigError(f'{field} must be a nonempty string')
    if not controls and any((ord(char) < 32 or ord(char) == 127 for char in value)):
        raise ConfigError(f'{field} must not contain control characters')
    return value

def _integer(value, field, minimum, maximum=None):
    if type(value) is not int:
        raise ConfigError(f'{field} must be an integer')
    if value < minimum or (maximum is not None and value > maximum):
        raise ConfigError(f'{field} is outside the supported range')
    return value

def _number(value, field, *, zero=False):
    if type(value) not in (int, float):
        raise ConfigError(f'{field} must be a number')
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < 0 or (not zero and value == 0):
        adjective = 'nonnegative' if zero else 'positive'
        raise ConfigError(f'{field} must be {adjective} and finite')
    return value

def _path(value, field, directory, *, optional=False):
    if optional and value is None:
        return None
    value = _string(value, field, nonempty=True)
    try:
        resolved = Path(value).expanduser()
        if not resolved.is_absolute():
            resolved = directory / resolved
        return str(resolved.resolve())
    except (OSError, RuntimeError, ValueError):
        raise ConfigError(f'{field} is not a valid filesystem path') from None

def _telegram_config(config):
    name = _string(config.get("trainer_name", DEFAULT_CONFIG["trainer_name"]),
                   "trainer_name", nonempty=True)
    if len(name) > 128:
        raise ConfigError("trainer_name must contain at most 128 characters")
    settings = dict(_TELEGRAM_DEFAULTS)
    settings.update(_section(config.get("telegram", {}), set(settings), "telegram"))
    if type(settings["enabled"]) is not bool:
        raise ConfigError("telegram.enabled must be true or false")
    token = _string(settings["tgBot_key"], "telegram.tgBot_key")
    if token and re.fullmatch(r"[0-9]+:[A-Za-z0-9_-]+", token) is None:
        raise ConfigError("telegram.tgBot_key must be a valid bot token")
    chat_id = settings["tgChatID"]
    if type(chat_id) is not int:
        _string(chat_id, "telegram.tgChatID")
        if chat_id and re.fullmatch(r"-?[0-9]+|@[A-Za-z0-9_]+", chat_id) is None:
            raise ConfigError("telegram.tgChatID must be a chat ID or channel username")
    if settings["enabled"] and (not token or not chat_id):
        raise ConfigError("telegram.enabled requires telegram.tgBot_key and telegram.tgChatID")
    _number(settings["update_interval_seconds"], "telegram.update_interval_seconds")
    return name, settings


def _http_url(value, field, *, allow_fragment=False):
    value = _string(value, field, nonempty=True)
    try:
        parsed = urlsplit(value)
        valid = parsed.scheme in {'http', 'https'} and bool(parsed.hostname) and (parsed.username is None) and (parsed.password is None) and (allow_fragment or not parsed.fragment) and (not any((char.isspace() for char in value)))
        port = parsed.port
        if port is not None and (not 1 <= port <= 65535):
            valid = False
    except ValueError:
        valid = False
    if not valid:
        restrictions = 'credentials' if allow_fragment else 'credentials or a fragment'
        raise ConfigError(f'{field} must be an HTTP or HTTPS URL without {restrictions}')
    return value

def _side_job_config(raw, directory, *, required=False):
    raw = _section(raw, set(_SIDE_JOB_DEFAULTS), 'side_job')
    settings = copy.deepcopy(_SIDE_JOB_DEFAULTS)
    settings.update(copy.deepcopy({field: value for field, value in raw.items() if field != 'vpn'}))
    _string(settings['repo_url'], 'side_job.repo_url', nonempty=required)
    if settings['repo_url']:
        url = _http_url(settings['repo_url'], 'side_job.repo_url')
        parsed = urlsplit(url)
        if parsed.scheme != 'https' or parsed.hostname != 'huggingface.co' or parsed.port not in {None, 443} or parsed.query or (re.fullmatch('/datasets/[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+(?:/tree/[^/?#]+(?:/[^?#]*)?)?/?', parsed.path) is None):
            raise ConfigError('side_job.repo_url must be a Hugging Face HTTPS dataset or dataset tree URL')
    for field, maximum in (('channels', 2048), ('blocks_per_stage', 8)):
        values = settings[field]
        if not isinstance(values, list) or not 1 <= len(values) <= 6:
            raise ConfigError(f'side_job.{field} must be a list containing 1 to 6 integers')
        for value in values:
            _integer(value, f'side_job.{field}', 1, maximum)
    if len(settings['channels']) != len(settings['blocks_per_stage']):
        raise ConfigError('side_job.channels and side_job.blocks_per_stage must have equal lengths')
    for field, minimum, maximum in (('hidden_size', 1, 8192), ('image_size', 32, 1024), ('batch_size', 1, 512), ('epochs', 0, 1000000000), ('prefetch_batches', 0, 8), ('passes_per_shard', 1, 100), ('cache_limit_mb', 16, 16384), ('seed', 0, 2 ** 32 - 1)):
        _integer(settings[field], f'side_job.{field}', minimum, maximum)
    for field in ('learning_rate', 'weight_decay'):
        _number(settings[field], f'side_job.{field}', zero=field == 'weight_decay')
        if settings[field] > 1:
            raise ConfigError(f'side_job.{field} must not exceed 1')
    if type(settings['use_amp']) is not bool:
        raise ConfigError('side_job.use_amp must be true or false')
    vpn = settings['vpn']
    vpn.update(_section(raw.get('vpn', {}), set(vpn), 'side_job.vpn'))
    if type(vpn['enabled']) is not bool:
        raise ConfigError('side_job.vpn.enabled must be true or false')
    _string(vpn['link'], 'side_job.vpn.link')
    if vpn['enabled'] and (not vpn['link'].strip()):
        raise ConfigError('side_job.vpn.link must contain a V2Ray share link when side_job.vpn.enabled is true')
    if vpn['link'] and vpn['link'].strip().split(':', 1)[0].lower() not in {'vless', 'vmess', 'trojan', 'ss'}:
        raise ConfigError('side_job.vpn.link must use vless, vmess, trojan, or ss')
    vpn['core_path'] = _path(vpn['core_path'], 'side_job.vpn.core_path', directory, optional=True)
    _number(vpn['startup_timeout'], 'side_job.vpn.startup_timeout')
    if vpn['startup_timeout'] > 60:
        raise ConfigError('side_job.vpn.startup_timeout must not exceed 60 seconds')
    return settings

def _host(value, field='S3.host'):
    value = _string(value, field)
    if not value:
        return value
    if any((char.isspace() for char in value)) or any((char in value for char in '/@?#\\')):
        raise ConfigError(f'{field} must be a hostname or IP address without a URL or credentials')
    try:
        ipaddress.ip_address(value)
        return value
    except ValueError:
        pass
    if len(value.rstrip('.')) > 253:
        raise ConfigError(f'{field} is not a valid hostname or IP address')
    labels = value.rstrip('.').split('.')
    if not all((re.fullmatch('[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?', label) for label in labels)):
        raise ConfigError(f'{field} is not a valid hostname or IP address')
    return value

def _documentation(value, directory, field='S3.protocol_documentation'):
    if value is None:
        return None
    value = _string(value, field, nonempty=True)
    if '://' in value or value.startswith(('http:', 'https:')):
        return _http_url(value, field, allow_fragment=True)
    return _path(value, field, directory)

def _S3_protocol(value):
    if value == 'auto':
        raise ConfigError('S3.protocol must explicitly select KEX or WR; automatic detection is unavailable')
    if not isinstance(value, str) or value not in {'KEX', 'WR', 'S7', S7_LEGACY_ALIAS}:
        raise ConfigError('S3.protocol must select KEX or WR')
    return 'WR' if value in {'S7', S7_LEGACY_ALIAS} else value

def _validate_S3_connection(settings, section, directory):
    settings['host'] = _host(settings['host'], f'{section}.host')
    _integer(settings['port'], f'{section}.port', 1, 65535)
    if type(settings['tls']) is not bool:
        raise ConfigError(f'{section}.tls must be true or false')
    settings['ca_file'] = _path(settings['ca_file'], f'{section}.ca_file', directory, optional=True)
    if settings['ca_file'] is not None and (not settings['tls']):
        raise ConfigError(f'{section}.ca_file requires {section}.tls to be true')
    if not isinstance(settings['compression'], str) or settings['compression'] not in {'none', 'gzip'}:
        raise ConfigError(f'{section}.compression must be none or gzip')
    if settings['certificate_version'] is not None:
        _integer(settings['certificate_version'], f'{section}.certificate_version', 1, 3)
    settings['protocol_documentation'] = _documentation(settings['protocol_documentation'], directory, f'{section}.protocol_documentation')

def resolve_S3_settings(settings: dict, S5: str | None=None) -> dict:
    if not isinstance(settings, dict):
        raise ConfigError('S3 must be an object')
    selected = _S3_protocol(settings.get('protocol', 'KEX') if S5 is None else S5)
    effective = copy.deepcopy(_S3_DEFAULTS)
    effective.update(copy.deepcopy(settings))
    if any((name in settings for name in _S3_PROFILES)):
        for name in _S3_PROFILES:
            if name in settings:
                _section(settings[name], _S3_CONNECTION_FIELDS, f'S3.{name}')
        for field in _S3_CONNECTION_FIELDS:
            effective[field] = _S3_DEFAULTS[field]
        effective.update(copy.deepcopy(settings.get(selected, {})))
    for name in _S3_PROFILES:
        effective.pop(name, None)
    effective['protocol'] = selected
    return effective

def _config_path(path) -> Path:
    if not isinstance(path, (str, Path)):
        raise ConfigError('configuration path must be a string or Path')
    try:
        return Path(path).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        raise ConfigError('configuration path is not a valid filesystem path') from None

def _open_text(config_path, key=None):
    if key is not None:
        return key
    try:
        return read_open_text(config_path.parent / '.env')
    except (ExportError, OSError, UnicodeError):
        raise ConfigError('could not read open_text from the adjacent .env file') from None

def _parse_json(raw):
    try:
        return json.loads(raw, object_pairs_hook=_object_without_duplicates, parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        raise ConfigError('configuration must contain valid JSON') from None

def read_config(path, *, key=None) -> dict:
    config_path = _config_path(path)
    try:
        raw = config_path.read_text(encoding='utf-8')
    except (OSError, UnicodeError):
        raise ConfigError('could not read the configuration file') from None
    document = _parse_json(raw)
    try:
        config = decode_config_document(document, _open_text(config_path, key))
    except ExportError:
        raise ConfigError('could not decode the encoded configuration with open_text') from None
    _validate_config(config, config_path)
    return config

def load_config(path, *, key=None) -> dict:
    config_path = _config_path(path)
    return _validate_config(read_config(config_path, key=key), config_path)

def _atomic_write(config_path, payload: str):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=config_path.parent, prefix='.config.json-', delete=False) as stream:
            temporary = Path(stream.name)
            os.fchmod(stream.fileno(), 384)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, config_path)
    except (OSError, UnicodeError):
        raise ConfigError('could not write the encrypted configuration file') from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

def save_config(path, values: dict, *, key=None) -> None:
    config_path = _config_path(path)
    _validate_config(values, config_path)
    try:
        document = encode_config_document(values, _open_text(config_path, key))
        payload = json.dumps(document, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
    except (ExportError, TypeError, ValueError, RecursionError):
        raise ConfigError('could not encode the configuration with open_text') from None
    _atomic_write(config_path, payload + '\n')

def update_config(path, changes: dict, *, key=None) -> None:
    if not isinstance(changes, dict):
        raise ConfigError('configuration changes must be an object')
    config = read_config(path, key=key)

    def merge(target, patch):
        for name, value in patch.items():
            if isinstance(value, dict) and isinstance(target.get(name), dict):
                merge(target[name], value)
            else:
                target[name] = copy.deepcopy(value)
    merge(config, changes)
    save_config(path, config, key=key)

def _validate_config(config, config_path) -> dict:
    config = _section(config, _TOP_LEVEL, 'configuration')
    trainer_name, telegram = _telegram_config(config)
    if 'version' not in config or type(config['version']) is not int or config['version'] != 1:
        raise ConfigError('configuration.version must be the integer 1')
    mode = config.get('mode', 'S3')
    if mode != 'S3':
        raise ConfigError('configuration.mode must be S3')
    device = _string(config.get('device', 'cuda:0'), 'device', nonempty=True)
    if re.fullmatch('cuda(?::[0-9]+)?', device) is None:
        raise ConfigError('device must select CUDA, for example cuda:0')
    directory = config_path.parent
    official_source = _path(config.get('official_source', 'src/LRAEP_helpers'), 'official_source', directory)
    raw_training = config.get('training', {})
    if isinstance(raw_training, dict) and {'main_effort', 'side_job_effort'} & raw_training.keys():
        raise ConfigError('training.main_effort and training.side_job_effort were replaced by training.job_sequence')
    raw_training = _section(raw_training,
                            set(_TRAINING_DEFAULTS) | {'max_gpu_utilization'}, 'training')
    training = copy.deepcopy(_TRAINING_DEFAULTS)
    training.update({field: value for field, value in raw_training.items()
                     if field != 'max_gpu_utilization'})
    if 'max_gpu_utilization' in raw_training:
        legacy = _integer(raw_training['max_gpu_utilization'], 'training.max_gpu_utilization', 1, 100)
        if 'job_sequence' not in raw_training:
            training['job_sequence'] = [['main', 60, legacy]]
    try:
        training['job_sequence'] = normalize_job_sequence(
            training['job_sequence'], field='training.job_sequence',
        )
    except ValueError as exc:
        raise ConfigError(str(exc)) from None
    for field in ('m', 'n'):
        _integer(training[field], f'training.{field}', 128, 4096)
    _integer(training['k'], 'training.k', 2048, 65536)
    if training['k'] % 128:
        raise ConfigError('training.k must be a multiple of 128')
    for field in ('tile_h', 'tile_w'):
        _integer(training[field], f'training.{field}', 2, 128)
        if training[field] % 2:
            raise ConfigError(f'training.{field} must be even')
    if not 32 <= training['tile_h'] * training['tile_w'] <= 256:
        raise ConfigError('training tile area must be between 32 and 256')
    if training['m'] % training['tile_h'] or training['n'] % training['tile_w']:
        raise ConfigError('training matrix dimensions must be divisible by their tile dimensions')
    if (training['tile_h'] + training['tile_w']) * training['k'] > 1 << 22:
        raise ConfigError("training proof inputs exceed S6's 4 MiB worker-input limit")
    _integer(training['max_batches'], 'training.max_batches', 0)
    _number(training['max_seconds'], 'training.max_seconds', zero=True)
    training['output_dir'] = _path(training['output_dir'], 'training.output_dir', directory)
    raw_S3 = _section(config.get('S3', {}), set(_S3_DEFAULTS) | set(_S3_PROFILES), 'S3')
    S3 = dict(_S3_DEFAULTS)
    S3.update({field: value for field, value in raw_S3.items() if field not in _S3_PROFILES})
    _validate_S3_connection(S3, 'S3', directory)
    for field in ('WR_S2', 'KEX_uname', 'worker', 'password'):
        _string(S3[field], f'S3.{field}', controls=field == 'password')
    S3['protocol'] = _S3_protocol(S3['protocol'])
    if S3['protocol'] != 'WR':
        raise ConfigError('S3.protocol must be WR')
    if type(S3['stop_on_unknown_request']) is not bool:
        raise ConfigError('S3.stop_on_unknown_request must be true or false')
    S3['unknown_request_dir'] = _path(S3['unknown_request_dir'], 'S3.unknown_request_dir', directory, optional=True)
    _string(S3['agent'], 'S3.agent', nonempty=True)
    if len(S3['agent']) > 128:
        raise ConfigError('S3.agent must contain at most 128 characters')
    for field in ('connect_timeout', 'read_timeout', 'reconnect_delay'):
        _number(S3[field], f'S3.{field}')
    for name in _S3_PROFILES:
        if name in raw_S3:
            profile = {field: _S3_DEFAULTS[field] for field in _S3_CONNECTION_FIELDS}
            profile.update(_section(raw_S3[name], _S3_CONNECTION_FIELDS, f'S3.{name}'))
            _validate_S3_connection(profile, f'S3.{name}', directory)
            S3[name] = profile
    node = dict(_NODE_DEFAULTS)
    node.update(_section(config.get('node', {}), set(node), 'node'))
    node['rpc_url'] = _http_url(node['rpc_url'], 'node.rpc_url')
    for field in ('username', 'password', 'address'):
        _string(node[field], f'node.{field}', controls=field == 'password')
    if ':' in node['username']:
        raise ConfigError('node.username must not contain a colon')
    node['ca_file'] = _path(node['ca_file'], 'node.ca_file', directory, optional=True)
    if node['ca_file'] is not None and urlsplit(node['rpc_url']).scheme != 'https':
        raise ConfigError('node.ca_file requires an HTTPS node.rpc_url')
    for field in ('poll_interval', 'refresh_interval', 'rpc_timeout'):
        _number(node[field], f'node.{field}')
    vpn = dict(_VPN_DEFAULTS)
    vpn.update(_section(config.get('vpn', {}), set(vpn), 'vpn'))
    if vpn['enabled'] is not True:
        raise ConfigError('VPN must be enabled')
    _string(vpn['link'], 'vpn.link')
    if vpn['enabled'] and (not vpn['link'].strip()):
        raise ConfigError('vpn.link must contain a V2Ray share link when vpn.enabled is true')
    if vpn['link'] and vpn['link'].strip().split(':', 1)[0].lower() not in {'vless', 'vmess', 'trojan', 'ss'}:
        raise ConfigError('vpn.link must use vless, vmess, trojan, or ss')
    vpn['core_path'] = _path(vpn['core_path'], 'vpn.core_path', directory, optional=True)
    vpn['links_file'] = _path(vpn['links_file'], 'vpn.links_file', directory)
    _number(vpn['startup_timeout'], 'vpn.startup_timeout')
    if vpn['startup_timeout'] > 60:
        raise ConfigError('vpn.startup_timeout must not exceed 60 seconds')
    if vpn['check_url'] is not None:
        vpn['check_url'] = _http_url(vpn['check_url'], 'vpn.check_url')
        if urlsplit(vpn['check_url']).scheme != 'https':
            raise ConfigError('vpn.check_url must use HTTPS')
    side_job = _side_job_config(config.get('side_job', {}), directory,
                                required=any(entry[0] == 'side-job' and entry[2] > 0
                                             for entry in training['job_sequence']))
    return {'version': 1, 'mode': mode, 'official_source': official_source, 'device': device, 'training': training, 'S3': S3, 'node': node, 'vpn': vpn, 'side_job': side_job, 'trainer_name': trainer_name, 'telegram': telegram}
