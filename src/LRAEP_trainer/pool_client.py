from __future__ import annotations
from dataclasses import dataclass, field
import json
import math
import threading
import time
from typing import Any
from .pool_protocol import KEXDialect, PoolJob, PoolProtocolError
from .pool_transport import MAX_IO, PoolConnectionError, PoolTransport

class PoolAuthenticationError(PoolConnectionError):
    pass

class PoolCompatibilityError(PoolConnectionError):
    pass

class StalePoolJob(RuntimeError):
    pass

@dataclass(frozen=True)
class JobSnapshot:
    job: PoolJob
    generation: int

@dataclass(frozen=True)
class ShareResult:
    accepted: bool
    error: str | None = None

@dataclass
class _Pending:
    event: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] | None = None
    error: PoolConnectionError | None = None

def _seconds(value: float, *, allow_zero: bool=False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('Pool wait timeout must be a finite number')
    try:
        seconds = float(value)
    except OverflowError:
        seconds = math.inf
    if not math.isfinite(seconds) or seconds < 0 or (seconds == 0 and (not allow_zero)):
        raise ValueError('Pool wait timeout must be a positive finite number')
    return seconds

def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError('duplicate JSON field')
        result[name] = value
    return result

def _reject_constant(value: str) -> None:
    raise ValueError('non-finite JSON constant')

class PoolSession:
    AUTH_METHOD = 'mining.authorize'
    MAX_FRAME_BYTES = MAX_IO
    MAX_PENDING_SUBMISSIONS = 64

    def __init__(self, settings: dict, stop_event: threading.Event | None=None) -> None:
        self.settings = dict(settings)
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        compression = self.settings.get('compression', 'none')
        if compression not in ('none', 'gzip'):
            raise PoolProtocolError('Pool compression must be none or gzip')
        self.dialect = self._create_dialect()
        self.transport = PoolTransport(host=self.settings['host'], port=self.settings['port'], tls=self.settings.get('tls', False), ca_file=self.settings.get('ca_file'), connect_timeout=self.settings.get('connect_timeout', 15), read_timeout=self.settings.get('read_timeout', 60))
        self._condition = threading.Condition()
        self._send_lock = threading.Lock()
        self._closed_event = threading.Event()
        self._opened = False
        self._authorized = False
        self._failure: PoolConnectionError | None = None
        self._job: PoolJob | None = None
        self._generation = 0
        self._next_id = 1
        self._auth_id: int | None = None
        self._pending: dict[int, _Pending] = {}
        self._negotiated_gzip = False
        self._reader: threading.Thread | None = None
        self._stop_watcher: threading.Thread | None = None

    def _create_dialect(self) -> KEXDialect:
        return KEXDialect(KEX_uname=self.settings.get('KEX_uname', ''), worker=self.settings.get('worker', ''), password=self.settings.get('password', 'x'), agent=self.settings.get('agent', 'LRAEP-trainer/0.1'), request_gzip=self.settings.get('compression', 'none') == 'gzip')

    @property
    def negotiated_gzip(self) -> bool:
        with self._condition:
            return self._negotiated_gzip

    def _raise_failure(self) -> None:
        if self._failure is not None:
            raise self._failure
        if self.stop_event.is_set() or self._closed_event.is_set():
            raise PoolConnectionError('Pool session is stopped')

    def open(self) -> PoolSession:
        with self._condition:
            self._raise_failure()
            if self._authorized:
                return self
            if self._opened:
                raise PoolConnectionError('Pool session is already opening')
            self._opened = True
        deadline = time.monotonic() + self.transport.connect_timeout
        try:
            self.transport.open()
            requests = self.dialect.auth_requests(start_id=self._next_id)
            self._auth_id = next((r['id'] for r in requests if r['method'] == self.AUTH_METHOD))
            self._next_id = max((r['id'] for r in requests)) + 1
            self._reader = threading.Thread(target=self._read_loop, name='LRAEP-pool-reader', daemon=True)
            self._stop_watcher = threading.Thread(target=self._watch_stop, name='LRAEP-pool-stop', daemon=True)
            self._reader.start()
            self._stop_watcher.start()
            with self._send_lock:
                for request in requests:
                    self._send(request)
            with self._condition:
                while not self._authorized:
                    self._raise_failure()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise PoolConnectionError('Pool authentication timed out')
                    self._condition.wait(min(remaining, 0.1))
                self._raise_failure()
            return self
        except (PoolConnectionError, PoolProtocolError):
            self.close()
            raise

    def __enter__(self) -> PoolSession:
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _fail(self, error: PoolConnectionError) -> None:
        with self._condition:
            if self._failure is None:
                self._failure = error
            self._authorized = False
            self._job = None
            self._generation += 1
            self._closed_event.set()
            for pending in self._pending.values():
                if not pending.event.is_set():
                    pending.error = self._failure
                    pending.event.set()
            self._condition.notify_all()
        self.transport.close()

    def close(self) -> None:
        self._fail(PoolConnectionError('Pool session closed'))
        current = threading.current_thread()
        for thread in (self._reader, self._stop_watcher):
            if thread is not None and thread is not current:
                thread.join(timeout=1)

    def _watch_stop(self) -> None:
        while not self._closed_event.wait(0.05):
            if self.stop_event.is_set():
                self._fail(PoolConnectionError('Pool session stopped'))
                return

    def _send(self, message: dict[str, Any]) -> None:
        encoded = json.dumps(message, separators=(',', ':'), allow_nan=False).encode('utf-8') + b'\n'
        if len(encoded) > self.MAX_FRAME_BYTES:
            raise PoolProtocolError('Pool request exceeds the frame size limit')
        self.transport.send(encoded)

    def _read_loop(self) -> None:
        buffered = bytearray()
        try:
            while not self._closed_event.is_set():
                buffered.extend(self.transport.receive(65536))
                while True:
                    boundary = buffered.find(b'\n')
                    if boundary < 0:
                        if len(buffered) >= self.MAX_FRAME_BYTES:
                            raise PoolProtocolError('Pool message exceeds the frame size limit')
                        break
                    if boundary + 1 > self.MAX_FRAME_BYTES:
                        raise PoolProtocolError('Pool message exceeds the frame size limit')
                    frame = bytes(buffered[:boundary])
                    del buffered[:boundary + 1]
                    if not frame.strip():
                        continue
                    try:
                        response = json.loads(frame, object_pairs_hook=_json_object, parse_constant=_reject_constant)
                    except (ValueError, UnicodeError, RecursionError):
                        raise PoolProtocolError('Pool sent malformed JSON') from None
                    if not isinstance(response, dict):
                        raise PoolProtocolError('Pool messages must be JSON objects')
                    self._handle(response)
                    if self._closed_event.is_set():
                        return
        except PoolConnectionError as error:
            self._fail(error)
        except PoolProtocolError as error:
            self._fail(PoolCompatibilityError(str(error)))
        except Exception:
            self._fail(PoolConnectionError('Pool reader failed'))

    def _handle(self, response: dict[str, Any]) -> None:
        with self._condition:
            if self._closed_event.is_set():
                return
        method = response.get('method')
        if method == 'mining.notify':
            job = self.dialect.parse_job(response.get('params'), self.settings.get('certificate_version'))
            with self._condition:
                if self._closed_event.is_set():
                    return
                if job != self._job:
                    self._job = job
                    self._generation += 1
                    self._condition.notify_all()
            return
        if method in ('client.reconnect', 'mining.reconnect'):
            self._fail(PoolConnectionError('Pool requested reconnection to the configured endpoint'))
            return
        if method is not None:
            return
        request_id = response.get('id')
        if type(request_id) is not int:
            return
        with self._condition:
            if request_id == self._auth_id and (not self._authorized):
                if not self.dialect.accept(response):
                    self._fail(PoolAuthenticationError('Pool rejected authentication'))
                    return
                self._negotiated_gzip = self.dialect.negotiate_gzip(response)
                self._authorized = True
                self._condition.notify_all()
            elif request_id in self._pending:
                pending = self._pending[request_id]
                if not pending.event.is_set():
                    pending.response = response
                    pending.event.set()

    def wait_job(self, timeout: float=1) -> JobSnapshot | None:
        deadline = time.monotonic() + _seconds(timeout, allow_zero=True)
        with self._condition:
            if not self._opened:
                raise PoolConnectionError('Pool session is not open')
            while True:
                self._raise_failure()
                if self._authorized and self._job is not None:
                    return JobSnapshot(self._job, self._generation)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(min(remaining, 0.1))

    def is_current(self, snapshot: JobSnapshot) -> bool:
        with self._condition:
            return isinstance(snapshot, JobSnapshot) and self._authorized and (self._failure is None) and (not self.stop_event.is_set()) and (not self._closed_event.is_set()) and (snapshot.generation == self._generation) and (snapshot.job == self._job)

    def submit(self, snapshot: JobSnapshot, proof_b64: str, timeout: float | None=None) -> ShareResult:
        wait_seconds = _seconds(self.transport.read_timeout if timeout is None else timeout)
        with self._send_lock:
            with self._condition:
                self._raise_failure()
                if not self.is_current(snapshot):
                    raise StalePoolJob('Pool replaced this training job before submission')
                if len(self._pending) >= self.MAX_PENDING_SUBMISSIONS:
                    raise PoolConnectionError('Too many pending pool submissions')
                request_id = self._next_id
                self._next_id += 1
                negotiated = self._negotiated_gzip
            request = self.dialect.submit_request(snapshot.job, proof_b64, request_id, negotiated)
            pending = _Pending()
            with self._condition:
                self._raise_failure()
                if not self.is_current(snapshot):
                    raise StalePoolJob('Pool replaced this training job before submission')
                self._pending[request_id] = pending
            try:
                self._send(request)
            except (PoolConnectionError, PoolProtocolError):
                with self._condition:
                    self._pending.pop(request_id, None)
                self._fail(PoolConnectionError('Pool share send failed'))
                raise
        if not pending.event.wait(wait_seconds):
            with self._condition:
                self._pending.pop(request_id, None)
            error = PoolConnectionError('Pool share acknowledgement timed out')
            self._fail(error)
            raise error
        with self._condition:
            self._pending.pop(request_id, None)
        if pending.error is not None:
            raise pending.error
        if pending.response is None:
            raise PoolConnectionError('Pool closed before acknowledging the share')
        if self.dialect.accept(pending.response):
            return ShareResult(True)
        return ShareResult(False, 'Pool rejected share')
