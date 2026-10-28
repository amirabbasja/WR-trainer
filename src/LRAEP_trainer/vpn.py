from __future__ import annotations
import base64
import binascii
import ipaddress
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import subprocess
import time
from urllib.parse import parse_qsl, unquote, urlsplit
from .paths import artifacts_dir, bundled_xray
from .network import SocksProxy

class VPNError(RuntimeError):
    pass
VPNProxy = SocksProxy

def _text(value, label, *, empty=False):
    if not isinstance(value, str) or (not value and (not empty)) or any((ord(c) < 32 or ord(c) == 127 for c in value)):
        raise VPNError(f'VPN link contains an invalid {label}')
    return value

def _decode(value):
    try:
        encoded = value.encode('ascii')
        return base64.b64decode(encoded + b'=' * (-len(encoded) % 4), altchars=b'-_', validate=True).decode('utf-8')
    except (UnicodeError, ValueError, binascii.Error):
        raise VPNError('VPN link contains invalid base64 data') from None

def _port(value):
    try:
        if isinstance(value, bool):
            raise ValueError
        result = int(value)
        if not 1 <= result <= 65535 or str(result) != str(value):
            raise ValueError
        return result
    except (ValueError, TypeError):
        raise VPNError('VPN link contains an invalid server port') from None

def _host(value):
    value = _text(value, 'server address')
    if any((c.isspace() or c in '/\\@?#' for c in value)):
        raise VPNError('VPN link contains an invalid server address')
    try:
        ipaddress.ip_address(value)
    except ValueError:
        try:
            encoded = value.encode('idna').decode('ascii')
        except UnicodeError:
            raise VPNError('VPN link contains an invalid server address') from None
        if len(encoded.rstrip('.')) > 253 or not all((re.fullmatch('[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?', p) for p in encoded.rstrip('.').split('.'))):
            raise VPNError('VPN link contains an invalid server address')
    return value

def _boolean(value, label):
    if isinstance(value, bool):
        return value
    if value in ('1', 'true', 'True'):
        return True
    if value in ('0', 'false', 'False', ''):
        return False
    raise VPNError(f'VPN link contains an invalid {label}')

def _options(query):
    try:
        pairs = parse_qsl(query, keep_blank_values=True, strict_parsing=True)
    except ValueError:
        raise VPNError('VPN link contains an invalid query') from None
    result = {}
    for key, value in pairs:
        if key in result:
            raise VPNError('VPN link contains duplicate query options')
        _text(value, 'query value', empty=True)
        result[key] = value
    return result

def _one(options, *names, default=''):
    values = [options[name] for name in names if name in options]
    if len(values) > 1:
        raise VPNError('VPN link contains conflicting aliases')
    return values[0] if values else default

def _stream(options, *, default_security='none'):
    allowed = {'type', 'security', 'sni', 'serverName', 'fp', 'fingerprint', 'pbk', 'sid', 'spx', 'pqv', 'alpn', 'host', 'path', 'serviceName', 'mode', 'authority', 'headerType', 'allowInsecure', 'insecure', 'flow', 'encryption', 'scy', 'aid', 'alterId'}
    if options.keys() - allowed:
        raise VPNError('VPN link uses unsupported query options')
    network = options.get('type', 'tcp') or 'tcp'
    network = {'raw': 'tcp', 'websocket': 'ws', 'splithttp': 'xhttp'}.get(network, network)
    if network not in {'tcp', 'ws', 'grpc', 'httpupgrade', 'xhttp'}:
        raise VPNError('VPN link uses an unsupported transport')
    if options.get('headerType', 'none') not in {'', 'none'}:
        raise VPNError('VPN link uses unsupported transport header obfuscation')
    security = options.get('security', default_security) or 'none'
    if security not in {'none', 'tls', 'reality'}:
        raise VPNError('VPN link uses unsupported transport security')
    stream = {'network': network, 'security': security}
    host, path = (options.get('host', ''), options.get('path', '/') or '/')
    if network in {'ws', 'httpupgrade', 'xhttp'}:
        if not path.startswith('/'):
            raise VPNError('VPN HTTP transport path must begin with a slash')
        settings = {'path': path}
        if host:
            settings['host'] = host
        if network == 'xhttp':
            mode = options.get('mode', 'auto')
            if mode not in {'auto', 'packet-up', 'stream-up', 'stream-one'}:
                raise VPNError('VPN link uses unsupported XHTTP mode')
            settings['mode'] = mode
        elif options.get('mode'):
            raise VPNError('VPN link has mode settings for a different transport')
        stream[{'ws': 'wsSettings', 'httpupgrade': 'httpupgradeSettings', 'xhttp': 'xhttpSettings'}[network]] = settings
        if options.get('serviceName') or options.get('authority'):
            raise VPNError('VPN link has gRPC settings for a different transport')
    elif network == 'grpc':
        mode = options.get('mode', 'gun')
        if mode not in {'', 'gun', 'multi'}:
            raise VPNError('VPN link uses unsupported gRPC mode')
        settings = {'serviceName': options.get('serviceName', ''), 'multiMode': mode == 'multi'}
        authority = options.get('authority', '') or host
        if authority:
            settings['authority'] = authority
        stream['grpcSettings'] = settings
        if options.get('path'):
            raise VPNError('VPN gRPC links must use serviceName instead of path')
    elif any((options.get(name) for name in ('host', 'path', 'serviceName', 'authority', 'mode'))):
        raise VPNError('VPN link has HTTP transport settings for TCP')
    server_name = _one(options, 'sni', 'serverName')
    fingerprint = _one(options, 'fp', 'fingerprint')
    insecure = _boolean(_one(options, 'allowInsecure', 'insecure', default='false'), 'TLS verification setting')
    if insecure:
        raise VPNError('VPN links that disable TLS certificate verification are unsupported')
    if security == 'tls':
        tls = {'allowInsecure': insecure}
        if server_name:
            tls['serverName'] = server_name
        if fingerprint:
            tls['fingerprint'] = fingerprint
        if options.get('alpn'):
            alpn = options['alpn'].split(',')
            if any((not item for item in alpn)):
                raise VPNError('VPN link contains invalid ALPN settings')
            tls['alpn'] = alpn
        stream['tlsSettings'] = tls
        if any((options.get(name) for name in ('pbk', 'sid', 'spx', 'pqv'))):
            raise VPNError('VPN link has REALITY settings for TLS')
    elif security == 'reality':
        if network not in {'tcp', 'grpc', 'xhttp'}:
            raise VPNError('REALITY is unsupported with this VPN transport')
        key = options.get('pbk', '')
        try:
            raw_key = base64.b64decode(key.encode('ascii') + b'=' * (-len(key) % 4), altchars=b'-_', validate=True)
        except (UnicodeError, ValueError, binascii.Error):
            raw_key = b''
        if len(raw_key) != 32 or not server_name:
            raise VPNError('VPN REALITY link requires a valid public key and server name')
        short_id = options.get('sid', '')
        if len(short_id) > 16 or len(short_id) % 2 or (not re.fullmatch('[0-9a-fA-F]*', short_id)):
            raise VPNError('VPN REALITY link contains an invalid short ID')
        if insecure or options.get('alpn'):
            raise VPNError('VPN REALITY link contains unsupported TLS-only settings')
        reality = {'serverName': server_name, 'fingerprint': fingerprint or 'chrome', 'publicKey': key, 'shortId': short_id, 'spiderX': options.get('spx', '/')}
        if options.get('pqv'):
            reality['mldsa65Verify'] = options['pqv']
        stream['realitySettings'] = reality
    elif server_name or fingerprint or insecure or options.get('alpn') or any((options.get(name) for name in ('pbk', 'sid', 'spx', 'pqv'))):
        raise VPNError('VPN link contains security settings without TLS or REALITY')
    return stream

def _parse_vmess(link):
    payload = link[len('vmess://'):]
    if '#' in payload:
        payload = payload.split('#', 1)[0]

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise VPNError('VPN VMess link contains duplicate JSON fields')
            result[key] = value
        return result
    try:
        data = json.loads(_decode(payload), object_pairs_hook=unique)
    except (ValueError, TypeError):
        raise VPNError('VPN VMess link contains invalid JSON') from None
    if not isinstance(data, dict):
        raise VPNError('VPN VMess link must contain a JSON object')
    allowed = {'v', 'ps', 'add', 'port', 'id', 'aid', 'scy', 'net', 'type', 'host', 'path', 'tls', 'sni', 'alpn', 'fp', 'allowInsecure', 'serviceName', 'mode'}
    if data.keys() - allowed:
        raise VPNError('VPN VMess link uses unsupported options')
    for name, value in data.items():
        if name not in {'aid', 'port', 'v', 'allowInsecure'}:
            _text(value, 'VMess field', empty=True)
    aid = data.get('aid', 0)
    if aid not in (0, '0', ''):
        raise VPNError('legacy VMess alterId is unsupported; use AEAD with alterId 0')
    cipher = data.get('scy', 'auto') or 'auto'
    if cipher not in {'auto', 'aes-128-gcm', 'chacha20-poly1305'}:
        raise VPNError('VPN link uses unsupported VMess encryption')
    network = data.get('net', 'tcp') or 'tcp'
    options = {'type': network, 'security': data.get('tls', 'none') or 'none'}
    for name in ('host', 'sni', 'alpn', 'fp', 'allowInsecure', 'mode', 'serviceName'):
        if name in data and data[name] != '':
            options[name] = data[name]
    path = data.get('path', '')
    if network == 'grpc':
        if path and options.get('serviceName'):
            raise VPNError('VPN VMess link contains conflicting gRPC names')
        if path:
            options['serviceName'] = path
        if data.get('type') == 'multi':
            options['mode'] = 'multi'
        elif data.get('type', 'none') not in {'none', '', 'gun'}:
            raise VPNError('VPN VMess link uses unsupported gRPC mode')
    else:
        if path:
            options['path'] = path
        options['headerType'] = data.get('type', 'none') or 'none'
    user = {'id': _text(data.get('id'), 'VMess user ID'), 'alterId': 0, 'security': cipher}
    return {'tag': 'vpn', 'protocol': 'vmess', 'settings': {'vnext': [{'address': _host(data.get('add')), 'port': _port(data.get('port')), 'users': [user]}]}, 'streamSettings': _stream(options)}

def _parse_ss(link):
    payload = link[len('ss://'):].split('#', 1)[0]
    if '@' not in payload:
        if '?' in payload:
            encoded, query = payload.split('?', 1)
            payload = _decode(encoded) + '?' + query
        else:
            payload = _decode(payload)
    parsed = urlsplit('ss://' + payload)
    if parsed.path not in {'', '/'}:
        raise VPNError('VPN Shadowsocks link contains an unsupported path')
    options = _options(parsed.query)
    if options:
        raise VPNError('VPN Shadowsocks plugins and query options are unsupported')
    userinfo = parsed.netloc.rsplit('@', 1)[0]
    if '@' not in parsed.netloc:
        raise VPNError('VPN Shadowsocks link is missing authentication')
    userinfo = unquote(userinfo)
    info = userinfo if ':' in userinfo else _decode(userinfo)
    if ':' not in info:
        raise VPNError('VPN Shadowsocks link is missing its encryption method')
    method, password = info.split(':', 1)
    methods = {'aes-128-gcm', 'aes-256-gcm', 'chacha20-poly1305', 'chacha20-ietf-poly1305', 'xchacha20-poly1305', 'xchacha20-ietf-poly1305', '2022-blake3-aes-128-gcm', '2022-blake3-aes-256-gcm', '2022-blake3-chacha20-poly1305'}
    if method not in methods:
        raise VPNError('VPN link uses an unsupported Shadowsocks encryption method')
    server = {'address': _host(parsed.hostname), 'port': _port(parsed.port), 'method': method, 'password': _text(password, 'Shadowsocks password')}
    return {'tag': 'vpn', 'protocol': 'shadowsocks', 'settings': {'servers': [server]}}

def parse_link(link: str) -> dict:
    _text(link, 'share link')
    if len(link) > 65536 or link != link.strip():
        raise VPNError('VPN share link is too long or has surrounding whitespace')
    try:
        if link.startswith('vmess://'):
            return _parse_vmess(link)
        if link.startswith('ss://'):
            return _parse_ss(link)
        parsed = urlsplit(link)
        if parsed.scheme not in {'vless', 'trojan'}:
            raise VPNError('VPN link must use vless://, vmess://, trojan://, or ss://')
        if parsed.path not in {'', '/'} or parsed.password is not None:
            raise VPNError('VPN link contains an unsupported authority or path')
        user = _text(unquote(parsed.username or ''), 'authentication')
        host, port = (_host(parsed.hostname), _port(parsed.port))
        options = _options(parsed.query)
        stream = _stream(options, default_security='tls' if parsed.scheme == 'trojan' else 'none')
        if parsed.scheme == 'vless':
            flow = options.get('flow', '')
            if flow not in {'', 'xtls-rprx-vision', 'xtls-rprx-vision-udp443'}:
                raise VPNError('VPN link uses unsupported VLESS flow')
            if flow and (stream['network'] != 'tcp' or stream['security'] not in {'tls', 'reality'}):
                raise VPNError('VLESS Vision requires TCP with TLS or REALITY')
            record = {'id': user, 'encryption': options.get('encryption', 'none') or 'none'}
            if flow:
                record['flow'] = flow
            if any((options.get(name) for name in ('scy', 'aid', 'alterId'))):
                raise VPNError('VPN VLESS link contains VMess-only settings')
            settings = {'vnext': [{'address': host, 'port': port, 'users': [record]}]}
        else:
            if any((options.get(name) for name in ('flow', 'encryption', 'scy', 'aid', 'alterId'))):
                raise VPNError('VPN Trojan link contains unsupported protocol settings')
            settings = {'servers': [{'address': host, 'port': port, 'password': user}]}
        return {'tag': 'vpn', 'protocol': parsed.scheme, 'settings': settings, 'streamSettings': stream}
    except VPNError:
        raise
    except (ValueError, TypeError, UnicodeError, OverflowError):
        raise VPNError('VPN share link is malformed') from None

def _recv_exact(connection, length):
    value = b''
    while len(value) < length:
        part = connection.recv(length - len(value))
        if not part:
            raise OSError('local SOCKS endpoint closed')
        value += part
    return value

def _ready(proxy: VPNProxy, timeout: float) -> bool:
    try:
        with socket.create_connection((proxy.host, proxy.port), timeout=timeout) as connection:
            connection.settimeout(timeout)
            connection.sendall(b'\x05\x01\x02')
            if _recv_exact(connection, 2) != b'\x05\x02':
                return False
            user, password = (proxy.username.encode(), proxy.password.encode())
            connection.sendall(b'\x01' + bytes([len(user)]) + user + bytes([len(password)]) + password)
            return _recv_exact(connection, 2) == b'\x01\x00'
    except (OSError, ValueError):
        return False

class ManagedVPN:

    def __init__(self, settings: dict, *, link: str | None=None):
        self.settings = settings
        self.link = settings.get('link', '') if link is None else link
        self.proxy: VPNProxy | None = None
        self.process: subprocess.Popen | None = None

    def __enter__(self) -> VPNProxy:
        if self.process is not None:
            raise VPNError('VPN context is already active')
        outbound = parse_link(self.link)
        executable = self.settings.get('core_path')
        if executable:
            executable = str(Path(executable).expanduser())
            if not Path(executable).is_file() or not os.access(executable, os.X_OK):
                raise VPNError('configured Xray core executable is unavailable')
        else:
            core = bundled_xray()
            executable = str(core) if core is not None else shutil.which('xray')
            if not executable:
                bundled = artifacts_dir().parent / 'bin' / ('xray.exe' if os.name == 'nt' else 'xray')
                if bundled.is_file() and os.access(bundled, os.X_OK):
                    executable = str(bundled)
        if not executable:
            raise VPNError('Xray core is required; install it or set vpn.core_path')
        startup = self.settings.get('startup_timeout', 10.0)
        if type(startup) not in (int, float) or not math.isfinite(startup) or (not 0 < startup <= 300):
            raise VPNError('vpn.startup_timeout must be between 0 and 300 seconds')
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
            reservation.bind(('127.0.0.1', 0))
            port = reservation.getsockname()[1]
        proxy = VPNProxy('127.0.0.1', port, secrets.token_hex(16), secrets.token_hex(24))
        self.proxy = proxy
        try:
            configuration = {'log': {'loglevel': 'none', 'access': 'none', 'error': 'none'}, 'inbounds': [{'tag': 'app', 'listen': proxy.host, 'port': proxy.port, 'protocol': 'socks', 'settings': {'auth': 'password', 'accounts': [{'user': proxy.username, 'pass': proxy.password}], 'udp': False}, 'sniffing': {'enabled': False}}], 'outbounds': [outbound], 'routing': {'domainStrategy': 'AsIs', 'rules': [{'type': 'field', 'inboundTag': ['app'], 'outboundTag': 'vpn'}]}}
            self.process = subprocess.Popen([executable, 'run', '-format', 'json', '-config', 'stdin:'], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
            self.process.stdin.write(json.dumps(configuration).encode('utf-8'))
            self.process.stdin.close()
            deadline = time.monotonic() + startup
            while True:
                if self.process.poll() is not None:
                    raise VPNError('Xray core rejected the VPN configuration or exited during startup')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise VPNError('Xray core did not become ready before vpn.startup_timeout')
                if _ready(proxy, min(remaining, 0.2)):
                    if self.process.poll() is not None:
                        raise VPNError('Xray core exited during startup')
                    return proxy
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        except BaseException as exc:
            self.close()
            if isinstance(exc, (KeyboardInterrupt, SystemExit, VPNError)):
                raise
            raise VPNError('could not start the Xray core') from None

    def close(self):
        process, self.process = (self.process, None)
        try:
            if process is not None:
                if process.poll() is None:
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        finally:
            if process is not None and process.stdin is not None and (not process.stdin.closed):
                try:
                    process.stdin.close()
                except OSError:
                    pass
            self.proxy = None

    def __exit__(self, *_):
        self.close()
        return False
