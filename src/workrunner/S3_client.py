from __future__ import annotations
from dataclasses import dataclass, field
import json
import math
import threading
import time
from typing import Any
from . import S3_methods as methods
from .S3_protocol import KEXDialect, S3Job, S3ProtocolError
from .S3_transport import MAX_IO, S3ConnectionError, S3Transport
from . import wire_constants as wire
from .fixed_constants import constant_text

_SEND_FAILED = b'^puCx&&Jilht$H!!-v7f&db8X'
_ACK_TIMED_OUT = b'^puCx&&JilhsMOr$;i;l!o$GA%EHOhhttl=!o!Ei($f'
_ACK_MISSING = b'^puCh%gEHi!-vJf!N}FZhsMOr$;i;l!o$GM$-sxx&%%e)&&Jil'
_FAILED_RESULT = b'^puCy!p*|O)561t)X&D%!T'


class S3AuthenticationError(S3ConnectionError):
    pass

class S3CompatibilityError(S3ConnectionError):
    pass

class ReplacedS3Job(RuntimeError):
    pass

@dataclass(frozen=True)
class JobSnapshot:
    job: S3Job
    generation: int

@dataclass(frozen=True)
class SubmissionResult:
    confirmed: bool
    error: str | None = None

@dataclass
class _Pending:
    event: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] | None = None
    error: S3ConnectionError | None = None

def _seconds(value: float, *, allow_zero: bool=False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(constant_text(b'^puCt#?I4+)6UAm$kNk?%F@)+hsDB&#)rYq$<EWlhsn~)#lqD'))
    try:
        seconds = float(value)
    except OverflowError:
        seconds = math.inf
    if not math.isfinite(seconds) or seconds < 0 or (seconds == 0 and (not allow_zero)):
        raise ValueError(constant_text(b'^puCt#?I4+)6UAm$kNk?%F@)+hsDB&#)sF))Xvk+(ZYwp&dJWx!iUMy%EiLf'))
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

class S3Session:
    AUTH_METHOD = methods.COMPUTE_SESSION

    MAX_FRAME_BYTES = MAX_IO
    MAX_PENDING_SUBMISSIONS = 64

    def __init__(self, settings: dict, stop_event: threading.Event | None=None) -> None:
        self.settings = dict(settings)
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        compression = self.settings.get('compression', 'none')
        if compression not in ('none', 'gzip'):
            raise S3ProtocolError(constant_text(b'^puCh$jaB%!qn8x$jOJw($v$3#lnZl$jQQo$km6y-Okq'))
        self.dialect = self._create_dialect()
        self.transport = S3Transport(host=self.settings['host'], port=self.settings['port'], tls=self.settings.get('tls', False), ca_file=self.settings.get('ca_file'), connect_timeout=self.settings.get('connect_timeout', 15), read_timeout=self.settings.get('read_timeout', 60))
        self._condition = threading.Condition()
        self._send_lock = threading.Lock()
        self._closed_event = threading.Event()
        self._opened = False
        self._authorized = False
        self._failure: S3ConnectionError | None = None
        self._job: S3Job | None = None
        self._generation = 0
        self._next_id = 1
        self._auth_id: int | None = None
        self._pending: dict[int, _Pending] = {}
        self._negotiated_gzip = False
        self._reader: threading.Thread | None = None
        self._stop_watcher: threading.Thread | None = None

    def _create_dialect(self) -> KEXDialect:
        return KEXDialect(KEX_uname=self.settings.get('KEX_uname', ''), worker=self.settings.get(wire.DEVICE_ID, ''), password=self.settings.get('password', 'x'), **({'agent': self.settings['agent']} if 'agent' in self.settings else {}), request_gzip=self.settings.get('compression', 'none') == 'gzip')

    @property
    def negotiated_gzip(self) -> bool:
        with self._condition:
            return self._negotiated_gzip

    def _raise_failure(self) -> None:
        if self._failure is not None:
            raise self._failure
        if self.stop_event.is_set() or self._closed_event.is_set():
            raise S3ConnectionError(constant_text(b'^puCx!qn8x$jOJ!)Q8m5$k*4x!v'))

    def open(self) -> S3Session:
        with self._condition:
            self._raise_failure()
            if self._authorized:
                return self
            if self._opened:
                raise S3ConnectionError(constant_text(b'^puCx!qn8x$jOJ!)Q85))xyTZ-iOH7!pY9bzy'))
            self._opened = True
        deadline = time.monotonic() + self.transport.connect_timeout
        try:
            self.transport.open()
            requests = self.dialect.auth_requests(start_id=self._next_id)
            try:
                self._auth_id = next((r['id'] for r in requests if r['method'] == self.AUTH_METHOD))
                self._next_id = max((r['id'] for r in requests)) + 1
                self._reader = threading.Thread(target=self._read_loop, name=constant_text(b'@b!xHl#12D#>2wZ'), daemon=True)
                self._stop_watcher = threading.Thread(target=self._watch_stop, name=constant_text(b'@b!xHl#0~T$kz'), daemon=True)
                self._reader.start()
                self._stop_watcher.start()
                with self._send_lock:
                    for request in requests:
                        try:
                            self._send(request)
                        finally:
                            del request
            finally:
                del requests
            with self._condition:
                while not self._authorized:
                    self._raise_failure()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise S3ConnectionError(constant_text(b'^puCj($mkv$<xln#?#Ko$%oU<%EH5k$kNj'))
                    self._condition.wait(min(remaining, 0.1))
                self._raise_failure()
            return self
        except (S3ConnectionError, S3ProtocolError):
            self.close()
            raise

    def __enter__(self) -> S3Session:
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _fail(self, error: S3ConnectionError) -> None:
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
        self._fail(S3ConnectionError(constant_text(b'^puCx!qn8x$jOJq%gEHi!v')))
        current = threading.current_thread()
        for thread in (self._reader, self._stop_watcher):
            if thread is not None and thread is not current:
                thread.join(timeout=1)

    def _watch_stop(self) -> None:
        while not self._closed_event.wait(0.05):
            if self.stop_event.is_set():
                self._fail(S3ConnectionError(constant_text(b'^puCx!qn8x$jOJ))5zD?!ov')))
                return

    def _send(self, message: dict[str, Any]) -> None:
        encoded = None
        try:
            message = methods.s4_request(message)
            encoded = json.dumps(message, separators=(',', ':'), allow_nan=False).encode('utf-8') + b'\n'
            if len(encoded) > self.MAX_FRAME_BYTES:
                raise S3ProtocolError(constant_text(b'^puCy!q(El)YFH;-^9Yg!_<e<&%%en)yB%ght$s9!iUSw%Ffd'))
            self.transport.send(encoded)
        finally:
            del message, encoded

    def _read_loop(self) -> None:
        buffered = bytearray()
        try:
            while not self._closed_event.is_set():
                buffered.extend(self.transport.receive(65536))
                while True:
                    boundary = buffered.find(b'\n')
                    if boundary < 0:
                        if len(buffered) >= self.MAX_FRAME_BYTES:
                            raise S3ProtocolError(constant_text(b'^puCn!qn8pz`}>Z-^9Yg!_<e<&%%en)yB%ght$s9!iUSw%Ffd'))
                        break
                    if boundary + 1 > self.MAX_FRAME_BYTES:
                        raise S3ProtocolError(constant_text(b'^puCn!qn8pz`}>Z-^9Yg!_<e<&%%en)yB%ght$s9!iUSw%Ffd'))
                    frame = bytes(buffered[:boundary])
                    del buffered[:boundary + 1]
                    if not frame.strip():
                        continue
                    try:
                        response = json.loads(frame, object_pairs_hook=_json_object, parse_constant=_reject_constant)
                    except (ValueError, UnicodeError, RecursionError):
                        raise S3ProtocolError(constant_text(b'^puCx!pYNz%ErsV$koci!-wtk=;;')) from None
                    if not isinstance(response, dict):
                        raise S3ProtocolError(constant_text(b'^puCn!qn8pz{1pr%F@)+hsDB&?eysBhsedv!o<_m'))
                    self._handle(response)
                    if self._closed_event.is_set():
                        return
        except S3ConnectionError as error:
            self._fail(error)
        except S3ProtocolError as error:
            self._fail(S3CompatibilityError(str(error)))
        except Exception:
            self._fail(S3ConnectionError(constant_text(b'^puCy!p6hG)rY~x&db8X')))

    def _handle(self, response: dict[str, Any]) -> None:
        with self._condition:
            if self._closed_event.is_set():
                return
        method = response.get('method')
        notification = methods.s4_notification_method(method)
        if notification == methods.WORK_UPDATE:
            job = self.dialect.parse_job(response.get('params'), self.settings.get(wire.FORMAT_SETTING))
            with self._condition:
                if self._closed_event.is_set():
                    return
                if job != self._job:
                    self._job = job
                    self._generation += 1
                    self._condition.notify_all()
            return
        if notification == methods.COMPUTE_RECONNECT:
            self._fail(S3ConnectionError(constant_text(b'^puCy!q(El)YHPlht<Nw$jQmV#M92m$%oU(htto(hs4Oq!Op<a)xyJv!pXze$j-^r')))
            return
        if method is not None:
            return
        request_id = response.get('id')
        if type(request_id) is not int:
            return
        with self._condition:
            if request_id == self._auth_id and (not self._authorized):
                if not self.dialect.accept(response):
                    self._fail(S3AuthenticationError(constant_text(b'^puCy!p*|O)561t#?sTz!pYOl#KzOk$jJ')))
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
                raise S3ConnectionError(constant_text(b'^puCx!qn8x$jOJ!)Q8E)(}&2{!pQ'))
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

    def send_result(self, snapshot: JobSnapshot, proof_b64: str, timeout: float | None=None) -> SubmissionResult:
        wait_seconds = _seconds(self.transport.read_timeout if timeout is None else timeout)
        with self._send_lock:
            with self._condition:
                self._raise_failure()
                if not self.is_current(snapshot):
                    raise ReplacedS3Job(constant_text(b'^puCy!q>~j#KOae)6dS-htt)@&dJWnz=zGq#fQbh!N}FZht$%=%Ffi(&dA9'))
                if len(self._pending) >= self.MAX_PENDING_SUBMISSIONS:
                    raise S3ConnectionError(constant_text(b'^T^1D%Ermwhu6Z%!_LXThxC+()Y8Su&eYV-$jQ_'))
                request_id = self._next_id
                self._next_id += 1
                negotiated = self._negotiated_gzip
            request = self.dialect.submission_request(snapshot.job, proof_b64, request_id, negotiated)
            try:
                pending = _Pending()
                with self._condition:
                    self._raise_failure()
                    if not self.is_current(snapshot):
                        raise ReplacedS3Job(constant_text(b'^puCy!q>~j#KOae)6dS-htt)@&dJWnz=zGq#fQbh!N}FZht$%=%Ffi(&dA9'))
                    self._pending[request_id] = pending
                try:
                    self._send(request)
                except (S3ConnectionError, S3ProtocolError):
                    with self._condition:
                        self._pending.pop(request_id, None)
                    self._fail(S3ConnectionError(constant_text(_SEND_FAILED)))
                    raise
            finally:
                del request
        if not pending.event.wait(wait_seconds):
            with self._condition:
                self._pending.pop(request_id, None)
            error = S3ConnectionError(constant_text(_ACK_TIMED_OUT))
            self._fail(error)
            raise error
        with self._condition:
            self._pending.pop(request_id, None)
        if pending.error is not None:
            raise pending.error
        if pending.response is None:
            raise S3ConnectionError(constant_text(_ACK_MISSING))
        if self.dialect.accept(pending.response):
            return SubmissionResult(True)
        return SubmissionResult(False, constant_text(_FAILED_RESULT))
