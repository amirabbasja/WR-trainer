from __future__ import annotations
from .fixed_constants import constant_text
from dataclasses import replace
import logging
import threading
import time
from typing import Callable
from .S6_codec import CodecError, decode_frame, encode_frame
from .S3_client import JobSnapshot, S3AuthenticationError, S3CompatibilityError, S3Session, _seconds
from .S3_protocol import S3ProtocolError
from .S3_transport import S3ConnectionError
from .wr_protocol import UnknownWRRequest, WRDialect
from . import wire_constants as wire
LOG = logging.getLogger(wire.LOGGER_NAME)

class _AuthMethod:
    def __get__(self, instance, owner):
        return wire.LOGIN


class WRSession(S3Session):
    AUTH_METHOD = _AuthMethod()
    MAX_PENDING_REPORTS = 4

    def __init__(self, settings: dict, stop_event: threading.Event | None=None, unknown_handler: Callable[..., None] | None=None) -> None:
        if settings.get('compression', 'none') != 'none':
            raise S3ProtocolError(constant_text(b'^pJ<d$jaB%!qn8x$jOJw($v$3#lnZl$jQP'))
        super().__init__(settings, stop_event)
        self._device_seed: bytes | None = None
        self._unknown_handler = unknown_handler
        self._recording_unknown = False
        self._compatibility_failure: S3CompatibilityError | None = None
        self._report_pending: dict[int, float] = {}
        self._latest_report_id: int | None = None
        self._latest_report_ok: bool | None = None

    def _create_dialect(self) -> WRDialect:
        return WRDialect(WR_S2=self.settings.get('WR_S2', ''), worker=self.settings.get(wire.DEVICE_ID, ''), **({'agent': self.settings['agent']} if 'agent' in self.settings else {}))

    def _raise_failure(self) -> None:
        if self._compatibility_failure is not None:
            raise self._compatibility_failure
        super()._raise_failure()

    def _fail(self, error: S3ConnectionError) -> None:
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
            LOG.warning(constant_text(b'^pJ<(_wt9;!q>>v(}%{y%*n{m%fiFJ!pg$Q(}&Z}%EH5k$kNj'))

    @property
    def latest_report_id(self) -> int | None:
        with self._condition:
            return self._latest_report_id

    @property
    def latest_report_ok(self) -> bool | None:
        with self._condition:
            self._expire_reports_locked()
            return self._latest_report_ok

    def send_device_report(self, gpu_name: str, work_rate: float) -> int:
        self.dialect.report_request(gpu_name, work_rate, 0)
        with self._send_lock:
            with self._condition:
                self._raise_failure()
                if not self._authorized:
                    raise S3ConnectionError(constant_text(b'^pJ<t!qn8x$jOJw($v$3#?sTz!pYOl#KzOYhsDCd$koD!)xy`v)zi+&z=z=X@`ujJ!N}Fh#?#Ko$p'))
                self._expire_reports_locked()
                if len(self._report_pending) >= self.MAX_PENDING_REPORTS:
                    del self._report_pending[next(iter(self._report_pending))]
                    LOG.warning(constant_text(b'^pJ<c&eX)l)x*NWhsMc=(#gie%*n{m%fiFJ!o!E)_wt9;!q>>v(*'))
                request_id = self._next_id
                request = self.dialect.report_request(gpu_name, work_rate, request_id)
                self._next_id += 1
                self._report_pending[request_id] = time.monotonic() + self.transport.read_timeout
                self._latest_report_id = request_id
                self._latest_report_ok = None
            try:
                self._send(request)
            except (S3ConnectionError, S3ProtocolError):
                self._fail(S3ConnectionError(constant_text(b'^pJ<(_wt9;!q>>v(}&c;$-{@i#?H&a!v')))
                raise
            finally:
                del request
        return request_id

    def _watch_stop(self) -> None:
        while not self._closed_event.wait(0.05):
            with self._condition:
                self._expire_reports_locked()
            if self.stop_event.is_set():
                self._fail(S3ConnectionError(constant_text(b'^puCx!qn8x$jOJ))5zD?!ov')))
                return

    def _send(self, message: dict) -> None:
        encoded = None
        try:
            try:
                encoded = encode_frame(message)
            except CodecError as error:
                raise S3ProtocolError(constant_text(b'<jB&?!-vVp(}%*z#K^<KhxCw#)xy@&!qn3')) from error
            if len(encoded) > self.MAX_FRAME_BYTES:
                raise S3ProtocolError(constant_text(b'^pJ<u!q(El)YFH;-^9Yg!_<e<&%%en)yB%ght$s9!iUSw%Ffd'))
            self.transport.send(encoded)
        finally:
            del message, encoded

    def _unknown(self, frame: bytes, response: dict | None, reason: str) -> None:
        with self._condition:
            if self._closed_event.is_set():
                return
            self._authorized = False
            self._job = None
            self._generation += 1
            self._recording_unknown = True
            self._compatibility_failure = S3CompatibilityError(constant_text(b'^2yA}$k54$^pJ<u!q(El)YF`Y)78e#$<E2Zht$)^*Vn?s'))
        diagnostic_failed = False
        if self._unknown_handler is not None:
            try:
                self._unknown_handler(frame, response, reason, fatal=True)
            except Exception:
                diagnostic_failed = True
        message = constant_text(b'^2yA}$k54$^pJ<u!q(El)YF`Y)78e#$<E2Zht$)^*Vn?s')
        if diagnostic_failed:
            message += '; diagnostic recording failed'
        error = S3CompatibilityError(message)
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
                            self._unknown(bytes(buffered), None, constant_text(b'^pJ<a)yB%ghr-{)!otJUhtto(ht$s9!iUSw%Ffd'))
                            return
                        break
                    if boundary + 1 > self.MAX_FRAME_BYTES:
                        self._unknown(bytes(buffered[:boundary + 1]), None, constant_text(b'^pJ<a)yB%ghr-{)!otJUhtto(ht$s9!iUSw%Ffd'))
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
                            self._unknown(frame, response, constant_text(b'<jB&?!-vVp(}&f<#K_gdht9ys$koEbhxCw#)xy@&!qn3'))
                            return
                    except (CodecError, S3ProtocolError) as error:
                        self._unknown(frame, response, str(error))
                        return
                    if self._closed_event.is_set():
                        return
        except S3ConnectionError as error:
            if buffered and (not self._closed_event.is_set()):
                self._unknown(bytes(buffered), None, constant_text(b'^pJ<d$jQmV#M92m$%n$p!@|Rd(9YA(hsMc=(#g}p)ymGv#?!*Xhr!jx%EA'))
            else:
                self._fail(error)
        except Exception:
            self._fail(S3ConnectionError(constant_text(b'^pJ<u!p6hG)rY~x&db8X')))

    def _handle(self, response: dict) -> None:
        with self._condition:
            if self._closed_event.is_set():
                return
        kind = self.dialect.classify(response)
        if kind == wire.SERVER_SEED:
            device_seed = self.dialect.parse_server_seed(response['result'])
            with self._condition:
                if self._closed_event.is_set():
                    return
                if device_seed != self._device_seed:
                    self._device_seed = device_seed
                    if self._job is not None:
                        self._job = replace(self._job, device_seed=device_seed)
                    self._generation += 1
                    self._condition.notify_all()
            return
        if kind == wire.JOB_UPDATE:
            with self._condition:
                if self._closed_event.is_set():
                    return
                job = self.dialect.parse_job(response['result'], self.settings.get(wire.FORMAT_SETTING), self._device_seed)
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
                    self._fail(S3AuthenticationError(constant_text(b'^pJ<u!p*|O)561t#?sTz!pYOl#KzOk$jJ')))
                    return
                self._authorized = True
                self._condition.notify_all()
            elif request_id in self._report_pending:
                del self._report_pending[request_id]
                confirmed = self.dialect.accept(response)
                if request_id == self._latest_report_id:
                    self._latest_report_ok = confirmed
                if confirmed:
                    LOG.debug(constant_text(b'^pJ<f#LUUa(96QZz{10a;P>)}&dI^Z)yl@x&dAA!)xy`v)zb'))
                else:
                    LOG.warning(constant_text(b'^pJ<u!p*|O)561t;P>)}&dI^Z)yl@x&dAA!)xy`v)zb'))
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
                if self._authorized and self._job is not None and (self._job.device_seed is not None):
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
