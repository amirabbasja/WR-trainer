from __future__ import annotations
from dataclasses import replace
import logging
import threading
import time
from typing import Callable
from .pearl_codec import CodecError, decode_frame, encode_frame
from .pool_client import JobSnapshot, PoolAuthenticationError, PoolCompatibilityError, PoolSession, _seconds
from .pool_protocol import PoolProtocolError
from .pool_transport import PoolConnectionError
from .wr_protocol import UnknownWRRequest, WRDialect
LOG = logging.getLogger('LRAEP_trainer')

class WRSession(PoolSession):
    AUTH_METHOD = 'login'
    MAX_PENDING_REPORTS = 4

    def __init__(self, settings: dict, stop_event: threading.Event | None=None, unknown_handler: Callable[..., None] | None=None) -> None:
        if settings.get('compression', 'none') != 'none':
            raise PoolProtocolError('PearlHash compression must be none')
        super().__init__(settings, stop_event)
        self._worker_nonce: bytes | None = None
        self._unknown_handler = unknown_handler
        self._recording_unknown = False
        self._compatibility_failure: PoolCompatibilityError | None = None
        self._report_pending: dict[int, float] = {}
        self._latest_report_id: int | None = None
        self._latest_report_accepted: bool | None = None

    def _create_dialect(self) -> WRDialect:
        return WRDialect(WR_wallet=self.settings.get('WR_wallet', ''), worker=self.settings.get('worker', ''), agent=self.settings.get('agent', 'LRAEP-trainer/0.1'))

    def _raise_failure(self) -> None:
        if self._compatibility_failure is not None:
            raise self._compatibility_failure
        super()._raise_failure()

    def _fail(self, error: PoolConnectionError) -> None:
        with self._condition:
            self._report_pending.clear()
            if self._compatibility_failure is not None:
                error = self._compatibility_failure
                self._failure = error
        super()._fail(error)

    def _expire_reports_locked(self) -> None:
        now = time.monotonic()
        expired = [request_id for request_id, deadline in self._report_pending.items() if now >= deadline]
        for request_id in expired:
            del self._report_pending[request_id]
            LOG.warning('PearlHash GPU report acknowledgement timed out')

    @property
    def latest_report_id(self) -> int | None:
        with self._condition:
            return self._latest_report_id

    @property
    def latest_report_accepted(self) -> bool | None:
        with self._condition:
            self._expire_reports_locked()
            return self._latest_report_accepted

    def report_info(self, gpu_name: str, hashrate: float) -> int:
        request = self.dialect.report_request(gpu_name, hashrate, 0)
        with self._send_lock:
            with self._condition:
                self._raise_failure()
                if not self._authorized:
                    raise PoolConnectionError('PearlHash session must authenticate before reporting GPU information')
                self._expire_reports_locked()
                if len(self._report_pending) >= self.MAX_PENDING_REPORTS:
                    del self._report_pending[next(iter(self._report_pending))]
                    LOG.warning('PearlHash discarded an unacknowledged GPU report')
                request_id = self._next_id
                request = self.dialect.report_request(gpu_name, hashrate, request_id)
                self._next_id += 1
                self._report_pending[request_id] = time.monotonic() + self.transport.read_timeout
                self._latest_report_id = request_id
                self._latest_report_accepted = None
            try:
                self._send(request)
            except (PoolConnectionError, PoolProtocolError):
                self._fail(PoolConnectionError('PearlHash GPU report send failed'))
                raise
        return request_id

    def _watch_stop(self) -> None:
        while not self._closed_event.wait(0.05):
            with self._condition:
                self._expire_reports_locked()
            if self.stop_event.is_set():
                self._fail(PoolConnectionError('Pool session stopped'))
                return

    def _send(self, message: dict) -> None:
        try:
            encoded = encode_frame(message)
        except CodecError as error:
            raise PoolProtocolError('Could not encode PearlHash request') from error
        if len(encoded) > self.MAX_FRAME_BYTES:
            raise PoolProtocolError('PearlHash request exceeds the frame size limit')
        self.transport.send(encoded)

    def _unknown(self, frame: bytes, response: dict | None, reason: str) -> None:
        with self._condition:
            if self._closed_event.is_set():
                return
            self._authorized = False
            self._job = None
            self._generation += 1
            self._recording_unknown = True
            self._compatibility_failure = PoolCompatibilityError('Unknown PearlHash request; training stopped')
        diagnostic_failed = False
        if self._unknown_handler is not None:
            try:
                self._unknown_handler(frame, response, reason, fatal=True)
            except Exception:
                diagnostic_failed = True
        message = 'Unknown PearlHash request; training stopped'
        if diagnostic_failed:
            message += '; diagnostic recording failed'
        error = PoolCompatibilityError(message)
        with self._condition:
            self._failure = error
            self._compatibility_failure = error
        self._fail(error)
        self.stop_event.set()
        with self._condition:
            self._recording_unknown = False
            self._condition.notify_all()

    def _read_loop(self) -> None:
        buffered = bytearray()
        try:
            while not self._closed_event.is_set():
                buffered.extend(self.transport.receive(65536))
                while True:
                    boundary = buffered.find(b'\n')
                    if boundary < 0:
                        if len(buffered) >= self.MAX_FRAME_BYTES:
                            self._unknown(bytes(buffered), None, 'PearlHash frame exceeds the size limit')
                            return
                        break
                    if boundary + 1 > self.MAX_FRAME_BYTES:
                        self._unknown(bytes(buffered[:boundary + 1]), None, 'PearlHash frame exceeds the size limit')
                        return
                    frame = bytes(buffered[:boundary + 1])
                    del buffered[:boundary + 1]
                    response = None
                    try:
                        response = decode_frame(frame)
                        self._handle(response)
                    except UnknownWRRequest as error:
                        if self.settings.get('stop_on_unknown_request', True):
                            self._unknown(frame, response, str(error))
                            return
                        try:
                            if self._unknown_handler is not None:
                                self._unknown_handler(frame, response, str(error), fatal=False)
                        except Exception:
                            self._unknown(frame, response, 'Could not record ignored PearlHash request')
                            return
                    except (CodecError, PoolProtocolError) as error:
                        self._unknown(frame, response, str(error))
                        return
                    if self._closed_event.is_set():
                        return
        except PoolConnectionError as error:
            if buffered and (not self._closed_event.is_set()):
                self._unknown(bytes(buffered), None, 'PearlHash connection ended with an unterminated frame')
            else:
                self._fail(error)
        except Exception:
            self._fail(PoolConnectionError('PearlHash reader failed'))

    def _handle(self, response: dict) -> None:
        with self._condition:
            if self._closed_event.is_set():
                return
        kind = self.dialect.classify(response)
        if kind == 'extranonce':
            worker_nonce = self.dialect.parse_extranonce(response['result'])
            with self._condition:
                if self._closed_event.is_set():
                    return
                if worker_nonce != self._worker_nonce:
                    self._worker_nonce = worker_nonce
                    if self._job is not None:
                        self._job = replace(self._job, worker_nonce=worker_nonce)
                    self._generation += 1
                    self._condition.notify_all()
            return
        if kind == 'notify':
            with self._condition:
                if self._closed_event.is_set():
                    return
                job = self.dialect.parse_job(response['result'], self.settings.get('certificate_version'), self._worker_nonce)
                if job != self._job:
                    self._job = job
                    self._generation += 1
                    self._condition.notify_all()
            return
        request_id = response['id']
        with self._condition:
            self._expire_reports_locked()
            if request_id == self._auth_id and (not self._authorized):
                if not self.dialect.accept(response):
                    self._fail(PoolAuthenticationError('PearlHash rejected authentication'))
                    return
                self._authorized = True
                self._condition.notify_all()
            elif request_id in self._report_pending:
                del self._report_pending[request_id]
                accepted = self.dialect.accept(response)
                if request_id == self._latest_report_id:
                    self._latest_report_accepted = accepted
                if accepted:
                    LOG.debug('PearlHash acknowledged GPU information report')
                else:
                    LOG.warning('PearlHash rejected GPU information report')
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
                if self._authorized and self._job is not None and (self._job.worker_nonce is not None):
                    return JobSnapshot(self._job, self._generation)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(min(remaining, 0.1))

    def raise_if_failed(self) -> None:
        deadline = time.monotonic() + 1
        with self._condition:
            while self._recording_unknown:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(min(remaining, 0.1))
            if self._compatibility_failure is not None:
                raise self._compatibility_failure
            if self._failure is not None:
                raise self._failure
