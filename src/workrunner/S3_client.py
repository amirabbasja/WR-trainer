from __future__ import annotations
from dataclasses import dataclass, field
import math
import threading
import time
from typing import Any
from . import training_methods as methods
from .S3_protocol import S3Dialect, S3Job, S3ProtocolError
from .S3_transport import MAX_IO, S3ConnectionError, S3Transport
from . import training_constants as wire
from .fixed_constants import FixedAlias, constant_text, neutral_protocol_text

_SEND_FAILED = FixedAlias('training.validation.coordinator_submission_send_failed')
_ACK_TIMED_OUT = FixedAlias('training.validation.coordinator_submission_acknowledgement_timeout')
_ACK_MISSING = FixedAlias('training.validation.coordinator_closed_before_acknowledgement')
_FAILED_RESULT = FixedAlias('training.validation.coordinator_submission_rejected')


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
        raise ValueError(constant_text(FixedAlias('training.validation.coordinator_wait_timeout_finite')))
    try:
        seconds = float(value)
    except OverflowError:
        seconds = math.inf
    if not math.isfinite(seconds) or seconds < 0 or (seconds == 0 and (not allow_zero)):
        raise ValueError(constant_text(FixedAlias('training.validation.coordinator_wait_timeout_positive')))
    return seconds



class S3Session:
    AUTH_METHOD = methods.COMPUTE_SESSION

    MAX_FRAME_BYTES = MAX_IO
    MAX_PENDING_SUBMISSIONS = 64

    def __init__(self, settings: dict, stop_event: threading.Event | None=None) -> None:
        self.settings = dict(settings)
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        compression = self.settings.get('compression', 'none')
        if compression not in ('none', 'gzip'):
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.coordinator_compression_unsupported')))
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

    def _create_dialect(self) -> S3Dialect:
        return S3Dialect(session_identity=self.settings.get('KEX_uname', ''), worker=self.settings.get(wire.DEVICE_ID, ''), password=self.settings.get('password', 'x'), **({'agent': self.settings['agent']} if 'agent' in self.settings else {}), request_gzip=self.settings.get('compression', 'none') == 'gzip')

    @property
    def negotiated_gzip(self) -> bool:
        with self._condition:
            return self._negotiated_gzip

    def _raise_failure(self) -> None:
        if self._failure is not None:
            raise self._failure
        if self.stop_event.is_set() or self._closed_event.is_set():
            raise S3ConnectionError(constant_text(FixedAlias('training.validation.coordinator_session_already_stopped')))

    def _start_auth_reader(self, auth_id: int, next_id: int) -> None:
        self._auth_id = auth_id
        self._next_id = next_id
        self._reader = threading.Thread(target=self._read_loop, name=constant_text(FixedAlias('training.thread.batch_message_reader')), daemon=True)
        self._stop_watcher = threading.Thread(target=self._watch_stop, name=constant_text(FixedAlias('training.thread.batch_session_stop_monitor')), daemon=True)
        self._reader.start()
        self._stop_watcher.start()

    def _send_auth_requests(self) -> None:
        requests = self.dialect.auth_requests(start_id=self._next_id)
        try:
            auth_id = next(r['id'] for r in requests if r['method'] == self.AUTH_METHOD)
            self._start_auth_reader(auth_id, max(r['id'] for r in requests) + 1)
            with self._send_lock:
                for request in requests:
                    try:
                        self._send(request)
                    finally:
                        del request
        finally:
            del requests

    @neutral_protocol_text()
    def open(self) -> S3Session:
        with self._condition:
            self._raise_failure()
            if self._authorized:
                return self
            if self._opened:
                raise S3ConnectionError(constant_text(FixedAlias('training.validation.coordinator_session_opening')))
            self._opened = True
        deadline = time.monotonic() + self.transport.connect_timeout
        try:
            self.transport.open()
            self._send_auth_requests()
            with self._condition:
                while not self._authorized:
                    self._raise_failure()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise S3ConnectionError(constant_text(FixedAlias('training.validation.coordinator_authentication_timeout')))
                    self._condition.wait(min(remaining, 0.1))
                self._raise_failure()
            return self
        except BaseException:
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
        self._fail(S3ConnectionError(constant_text(FixedAlias('training.validation.coordinator_session_closed'))))
        current = threading.current_thread()
        for thread in (self._reader, self._stop_watcher):
            if thread is not None and thread.ident is not None and thread is not current:
                thread.join(timeout=1)

    def _watch_stop(self) -> None:
        while not self._closed_event.wait(0.05):
            if self.stop_event.is_set():
                self._fail(S3ConnectionError(constant_text(FixedAlias('training.validation.coordinator_session_stopped'))))
                return

    @neutral_protocol_text()
    def _send(self, message: dict[str, Any]) -> None:
        try:
            message = methods.wire_request(message)
            try:
                self.transport.send_message(message, profile="s3", max_frame=self.MAX_FRAME_BYTES)
            except (TypeError, ValueError, RecursionError):
                raise S3ProtocolError("S3 request cannot be serialized within the frame limit") from None
        finally:
            del message

    @neutral_protocol_text()
    def _read_loop(self) -> None:
        try:
            while not self._closed_event.is_set():
                try:
                    response = self.transport.receive_message(profile="s3", max_frame=self.MAX_FRAME_BYTES)
                except (ValueError, TypeError, UnicodeError, RecursionError):
                    raise S3ProtocolError("S3 sent a malformed or oversized message") from None
                try:
                    if response is not None:
                        self._handle(response)
                finally:
                    del response
        except S3ConnectionError as error:
            self._fail(error)
        except S3ProtocolError as error:
            self._fail(S3CompatibilityError(str(error)))
        except Exception:
            self._fail(S3ConnectionError("S3 reader failed"))

    def _handle(self, response: dict[str, Any]) -> None:
        with self._condition:
            if self._closed_event.is_set():
                return
        method = response.get('method')
        notification = methods.notification_method(method)
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
            self._fail(S3ConnectionError(constant_text(FixedAlias('training.validation.coordinator_reconnect_requested'))))
            return
        if method is not None:
            return
        request_id = response.get('id')
        if type(request_id) is not int:
            return
        with self._condition:
            if request_id == self._auth_id and (not self._authorized):
                if not self.dialect.accept(response):
                    self._fail(S3AuthenticationError(constant_text(FixedAlias('training.validation.coordinator_authentication_rejected'))))
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
                raise S3ConnectionError(constant_text(FixedAlias('training.validation.coordinator_session_not_open')))
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

    @neutral_protocol_text()
    def send_result(self, snapshot: JobSnapshot, proof_b64: str, timeout: float | None=None) -> SubmissionResult:
        wait_seconds = _seconds(self.transport.read_timeout if timeout is None else timeout)
        with self._send_lock:
            with self._condition:
                self._raise_failure()
                if not self.is_current(snapshot):
                    raise ReplacedS3Job(constant_text(FixedAlias('training.validation.batch_replaced_before_submission')))
                if len(self._pending) >= self.MAX_PENDING_SUBMISSIONS:
                    raise S3ConnectionError(constant_text(FixedAlias('training.validation.coordinator_pending_submission_limit')))
                request_id = self._next_id
                self._next_id += 1
                negotiated = self._negotiated_gzip
            request = self.dialect.submission_request(snapshot.job, proof_b64, request_id, negotiated)
            try:
                pending = _Pending()
                with self._condition:
                    self._raise_failure()
                    if not self.is_current(snapshot):
                        raise ReplacedS3Job(constant_text(FixedAlias('training.validation.batch_replaced_before_submission')))
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
        return self._submission_result(pending.response)

    def _submission_result(self, response: dict) -> SubmissionResult:
        if self.dialect.accept(response):
            return SubmissionResult(True)
        return SubmissionResult(False, constant_text(_FAILED_RESULT))
