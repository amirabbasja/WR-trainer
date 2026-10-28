from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import http.client
import ipaddress
import math
import socket
import ssl
import struct
import time
import urllib.request

class NetworkError(OSError):
    pass

@dataclass(frozen=True)
class SocksProxy:
    host: str
    port: int
    username: str = field(default='', repr=False)
    password: str = field(default='', repr=False)

    def __post_init__(self):
        try:
            local = ipaddress.ip_address(self.host).is_loopback
        except (ValueError, TypeError):
            local = False
        if not local or type(self.port) is not int or (not 1 <= self.port <= 65535):
            raise NetworkError('VPN SOCKS endpoint must use a loopback IP and valid port')
        try:
            valid = isinstance(self.username, str) and isinstance(self.password, str) and (bool(self.username) == bool(self.password)) and (len(self.username.encode('utf-8')) <= 255) and (len(self.password.encode('utf-8')) <= 255)
        except UnicodeError:
            valid = False
        if not valid:
            raise NetworkError('VPN SOCKS credentials must be matching UTF-8 strings of at most 255 bytes')
_ROUTE: ContextVar[SocksProxy | None] = ContextVar('LRAEP_network_route', default=None)

def current_proxy() -> SocksProxy | None:
    return _ROUTE.get()

@contextmanager
def routed(proxy: SocksProxy | None):
    if proxy is not None and (not isinstance(proxy, SocksProxy)):
        raise NetworkError('Invalid VPN SOCKS endpoint')
    token = _ROUTE.set(proxy)
    try:
        yield proxy
    finally:
        _ROUTE.reset(token)

def _target(address):
    try:
        host, port = address
        if not isinstance(host, str) or not host or type(port) is not int or (not 1 <= port <= 65535) or ('%' in host) or any((ord(char) < 33 or ord(char) == 127 for char in host)) or any((char in host for char in '/\\@?#')):
            raise ValueError
        if host.startswith('[') and host.endswith(']'):
            host = str(ipaddress.IPv6Address(host[1:-1]))
        try:
            literal = ipaddress.ip_address(host)
        except ValueError:
            domain = host.encode('idna')
            if not 1 <= len(domain) <= 255:
                raise ValueError
            encoded = b'\x03' + bytes([len(domain)]) + domain
        else:
            encoded = (b'\x01' if literal.version == 4 else b'\x04') + literal.packed
        return encoded + struct.pack('!H', port)
    except (ValueError, TypeError, UnicodeError, OverflowError):
        raise NetworkError('Invalid VPN destination address') from None

def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('VPN proxy connection timed out')
    return remaining

def _read(connection, length, deadline):
    result = bytearray()
    while len(result) < length:
        connection.settimeout(_remaining(deadline))
        part = connection.recv(length - len(result))
        if not part:
            raise NetworkError('VPN proxy closed its connection during SOCKS negotiation')
        result.extend(part)
    return bytes(result)

def _send(connection, value, deadline):
    connection.settimeout(_remaining(deadline))
    connection.sendall(value)

def create_connection(address, timeout=30, *, proxy: SocksProxy | None=None, source_address=None) -> socket.socket:
    proxy = current_proxy() if proxy is None else proxy
    if not isinstance(proxy, SocksProxy):
        raise NetworkError('VPN SOCKS routing is required')
    target = _target(address)
    if timeout is socket._GLOBAL_DEFAULT_TIMEOUT or timeout is None:
        timeout = socket.getdefaulttimeout() or 30
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise NetworkError('VPN connection timeout must be positive and finite')
    try:
        valid_timeout = math.isfinite(timeout) and timeout > 0
    except OverflowError:
        valid_timeout = False
    if not valid_timeout:
        raise NetworkError('VPN connection timeout must be positive and finite')
    deadline = time.monotonic() + timeout
    connection = None
    try:
        connection = socket.create_connection((proxy.host, proxy.port), timeout=_remaining(deadline), source_address=source_address)
        method = 2 if proxy.username else 0
        _send(connection, bytes([5, 1, method]), deadline)
        if _read(connection, 2, deadline) != bytes([5, method]):
            raise NetworkError('VPN proxy rejected the required SOCKS authentication method')
        if method == 2:
            username, password = (proxy.username.encode('utf-8'), proxy.password.encode('utf-8'))
            _send(connection, bytes([1, len(username)]) + username + bytes([len(password)]) + password, deadline)
            if _read(connection, 2, deadline) != b'\x01\x00':
                raise NetworkError('VPN proxy rejected SOCKS credentials')
        _send(connection, b'\x05\x01\x00' + target, deadline)
        reply = _read(connection, 4, deadline)
        if reply[0] != 5 or reply[2] != 0:
            raise NetworkError('VPN proxy returned an invalid SOCKS response')
        if reply[1] != 0:
            raise NetworkError('VPN proxy could not connect to the destination')
        if reply[3] == 1:
            _read(connection, 4, deadline)
        elif reply[3] == 4:
            _read(connection, 16, deadline)
        elif reply[3] == 3:
            size = _read(connection, 1, deadline)[0]
            if size == 0:
                raise NetworkError('VPN proxy returned an invalid SOCKS bound address')
            _read(connection, size, deadline)
        else:
            raise NetworkError('VPN proxy returned an unsupported SOCKS address type')
        _read(connection, 2, deadline)
        connection.settimeout(_remaining(deadline))
        return connection
    except TimeoutError:
        if connection is not None:
            connection.close()
        raise TimeoutError('VPN proxy connection timed out') from None
    except NetworkError:
        if connection is not None:
            connection.close()
        raise
    except (OSError, ValueError, UnicodeError, OverflowError):
        if connection is not None:
            connection.close()
        raise NetworkError('VPN proxy connection failed') from None
    except BaseException:
        if connection is not None:
            connection.close()
        raise

class _SocksHTTPConnection(http.client.HTTPConnection):

    def __init__(self, host, *, proxy, **kwargs):
        super().__init__(host, **kwargs)
        self._proxy = proxy

    def connect(self):
        if self._tunnel_host:
            raise NetworkError('Additional HTTP proxies are disabled during VPN routing')
        self.sock = create_connection((self.host, self.port), self.timeout, proxy=self._proxy, source_address=self.source_address)

class _SocksHTTPSConnection(http.client.HTTPSConnection):

    def __init__(self, host, *, proxy, **kwargs):
        super().__init__(host, **kwargs)
        self._proxy = proxy

    def connect(self):
        if self._tunnel_host:
            raise NetworkError('Additional HTTP proxies are disabled during VPN routing')
        raw = create_connection((self.host, self.port), self.timeout, proxy=self._proxy, source_address=self.source_address)
        try:
            self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
        except BaseException:
            raw.close()
            raise

class _SocksHTTPHandler(urllib.request.HTTPHandler):

    def __init__(self, proxy):
        super().__init__()
        self._proxy = proxy

    def http_open(self, request):

        def factory(host, **kwargs):
            return _SocksHTTPConnection(host, proxy=self._proxy, **kwargs)
        return self.do_open(factory, request)

class _SocksHTTPSHandler(urllib.request.HTTPSHandler):

    def __init__(self, proxy, context):
        super().__init__(context=context)
        self._proxy = proxy

    def https_open(self, request):

        def factory(host, **kwargs):
            return _SocksHTTPSConnection(host, proxy=self._proxy, **kwargs)
        return self.do_open(factory, request, context=self._context)

def http_opener(proxy: SocksProxy | None, *, context: ssl.SSLContext | None=None, handlers=()) -> urllib.request.OpenerDirector:
    if not isinstance(proxy, SocksProxy):
        raise NetworkError('VPN SOCKS routing is required')
    transports = (_SocksHTTPHandler(proxy), _SocksHTTPSHandler(proxy, context))
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), *transports, *handlers)
