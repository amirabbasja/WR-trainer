from __future__ import annotations
import ipaddress
import math
import os
import re
import socket
import ssl
from . import network
MAX_IO = 8 * 1024 * 1024
_DNS_LABEL = re.compile('[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\\Z')

class S3ConnectionError(ConnectionError):
    pass

def _hostname(host: str) -> str:
    if not isinstance(host, str) or not host or any((c.isspace() for c in host)):
        raise S3ConnectionError('S3 host must be a DNS name or IP address')
    if '%' in host:
        raise S3ConnectionError('S3 IP addresses must not contain a scope ID')
    if host.startswith('[') and host.endswith(']'):
        try:
            return str(ipaddress.IPv6Address(host[1:-1]))
        except ValueError:
            raise S3ConnectionError('Invalid S3 IPv6 address') from None
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        pass
    try:
        ascii_host = host.encode('idna').decode('ascii')
    except UnicodeError:
        raise S3ConnectionError('Invalid S3 DNS name') from None
    dns_name = ascii_host.removesuffix('.')
    if len(dns_name) > 253 or not all((_DNS_LABEL.fullmatch(label) for label in dns_name.split('.'))):
        raise S3ConnectionError('S3 host must be a DNS name or IP address')
    return ascii_host

def _timeout(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise S3ConnectionError(f'S3 {label} timeout must be a positive finite number')
    try:
        result = float(value)
    except OverflowError:
        result = math.inf
    if not math.isfinite(result) or result <= 0:
        raise S3ConnectionError(f'S3 {label} timeout must be a positive finite number')
    return result

class S3Transport:

    def __init__(self, host: str, port: int, tls: bool=False, ca_file: str | os.PathLike[str] | None=None, connect_timeout: float=15, read_timeout: float=60) -> None:
        self.host = _hostname(host)
        if isinstance(port, bool) or not isinstance(port, int) or (not 1 <= port <= 65535):
            raise S3ConnectionError('S3 port must be an integer from 1 to 65535')
        if not isinstance(tls, bool):
            raise S3ConnectionError('S3 TLS setting must be a boolean')
        if ca_file is not None and (not isinstance(ca_file, (str, os.PathLike))):
            raise S3ConnectionError('S3 certificate authority file must be a path')
        if ca_file is not None and (not tls):
            raise S3ConnectionError('A S3 certificate authority file requires TLS')
        self.port = port
        self.tls = tls
        self.ca_file = ca_file
        self.connect_timeout = _timeout(connect_timeout, 'connect')
        self.read_timeout = _timeout(read_timeout, 'read')
        self._socket: socket.socket | None = None
        self._proxy = network.current_proxy()

    @property
    def endpoint(self) -> str:
        host = f'[{self.host}]' if ':' in self.host else self.host
        return f'{host}:{self.port}'

    @property
    def tls_version(self) -> str | None:
        if self.tls and self._socket is not None:
            return self._socket.version()
        return None

    def open(self) -> S3Transport:
        if not isinstance(self._proxy, network.SocksProxy):
            raise S3ConnectionError('VPN SOCKS routing is required')
        if self._socket is not None:
            return self
        context = None
        if self.tls:
            try:
                context = ssl.create_default_context(cafile=self.ca_file)
            except (OSError, ValueError, TypeError):
                raise S3ConnectionError('Could not load S3 certificate authorities') from None
        connection = None
        try:
            connection = network.create_connection((self.host, self.port), timeout=self.connect_timeout, proxy=self._proxy)
            if context is not None:
                connection = context.wrap_socket(connection, server_hostname=self.host)
            connection.settimeout(self.read_timeout)
            self._socket = connection
            return self
        except TimeoutError:
            raise S3ConnectionError(f'S3 connection timed out: {self.endpoint}') from None
        except ssl.SSLError:
            raise S3ConnectionError(f'S3 TLS verification or handshake failed: {self.endpoint}') from None
        except (OSError, ValueError, OverflowError):
            raise S3ConnectionError(f'Could not connect to S3: {self.endpoint}') from None
        finally:
            if self._socket is None and connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass

    def close(self) -> None:
        connection, self._socket = (self._socket, None)
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                connection.close()
            except OSError:
                pass

    def __enter__(self) -> S3Transport:
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _connected(self) -> socket.socket:
        if self._socket is None:
            raise S3ConnectionError('S3 connection is not open')
        return self._socket

    def send(self, data: bytes) -> None:
        if not isinstance(data, bytes):
            raise S3ConnectionError('S3 payload must be bytes')
        if len(data) > MAX_IO:
            raise S3ConnectionError(f'S3 payload exceeds {MAX_IO} bytes')
        connection = self._connected()
        try:
            connection.sendall(data)
        except TimeoutError:
            self.close()
            raise S3ConnectionError(f'S3 send timed out: {self.endpoint}') from None
        except OSError:
            self.close()
            raise S3ConnectionError(f'S3 send failed: {self.endpoint}') from None

    def receive(self, max_bytes: int=65536) -> bytes:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or (not 1 <= max_bytes <= MAX_IO):
            raise S3ConnectionError(f'S3 receive limit must be from 1 to {MAX_IO} bytes')
        connection = self._connected()
        try:
            fragment = connection.recv(max_bytes)
        except TimeoutError:
            self.close()
            raise S3ConnectionError(f'S3 receive timed out: {self.endpoint}') from None
        except OSError:
            self.close()
            raise S3ConnectionError(f'S3 receive failed: {self.endpoint}') from None
        if not fragment:
            self.close()
            raise S3ConnectionError(f'S3 closed the connection: {self.endpoint}')
        return fragment
