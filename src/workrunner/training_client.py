from __future__ import annotations
from .fixed_constants import FixedAlias, constant_text, neutral_protocol_text
from dataclasses import replace
import logging
import threading
import time
from typing import Callable
from .S3_client import JobSnapshot, S3AuthenticationError, S3CompatibilityError, S3Session, SubmissionResult, _seconds
from .S3_protocol import S3ProtocolError
from .S3_transport import S3ConnectionError
from .training_protocol import SHAPE_REJECTED, UnknownTrainingRequest, TrainingDialect
from . import training_constants as wire
LOG = logging.getLogger(wire.LOGGER_NAME)

class _AuthMethod:
    def __get__(self, instance, owner):
        return wire.LOGIN


class TrainingSession(S3Session):
    AUTH_METHOD = _AuthMethod()
    MAX_PENDING_REPORTS = 4

    def __init__(self, settings: dict, stop_event: threading.Event | None=None, unknown_handler: Callable[..., None] | None=None, *, login_config=None) -> None:
        if settings.get('compression', 'none') != 'none':
            raise S3ProtocolError(constant_text(FixedAlias('training.validation.batch_compression_unsupported')))
        self._login_config = login_config
        if login_config is not None:
            settings = {key: value for key, value in settings.items() if key not in ('WR_S2', 'KEX_uname')}
        super().__init__(settings, stop_event)
        self._device_seed: bytes | None = None
        self._unknown_handler = unknown_handler
        self._recording_unknown = False
        self._compatibility_failure: S3CompatibilityError | None = None
        self._report_pending: dict[int, float] = {}
        self._latest_report_id: int | None = None
        self._latest_report_ok: bool | None = None

    def _create_dialect(self) -> TrainingDialect:
        identity = None if self._login_config is not None else self.settings.get('WR_S2', '')
        return TrainingDialect(training_identity=identity, worker=self.settings.get(wire.DEVICE_ID, ''), **({'agent': self.settings['agent']} if 'agent' in self.settings else {}))

    def _send_auth_requests(self) -> None:
        if self._login_config is None:
            super()._send_auth_requests()
            return

        from .login_config import login_frame

        auth_id = self._next_id
        with login_frame(self._login_config, auth_id, self.settings, self.dialect.worker, self.dialect.agent) as frame:
            self._start_auth_reader(auth_id, auth_id + 1)
            with self._send_lock:
                self.transport.send(frame)

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
            LOG.warning(constant_text(FixedAlias('training.validation.accelerator_metrics_acknowledgement_timeout')))

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
        return self.send_devices_report(((gpu_name, work_rate),))

    @neutral_protocol_text()
    def send_devices_report(self, devices) -> int:
        request = self.dialect.devices_report_request(devices, 0)
        with self._send_lock:
            with self._condition:
                self._raise_failure()
                if not self._authorized:
                    raise S3ConnectionError(constant_text(FixedAlias('training.validation.accelerator_metrics_authentication_required')))
                self._expire_reports_locked()
                if len(self._report_pending) >= self.MAX_PENDING_REPORTS:
                    del self._report_pending[next(iter(self._report_pending))]
                    LOG.warning(constant_text(FixedAlias('training.log.accelerator_metrics_acknowledgement_missing')))
                request_id = self._next_id
                request['id'] = request_id
                self._next_id += 1
                self._report_pending[request_id] = time.monotonic() + self.transport.read_timeout
                self._latest_report_id = request_id
                self._latest_report_ok = None
            try:
                self._send(request)
            except (S3ConnectionError, S3ProtocolError):
                self._fail(S3ConnectionError(constant_text(FixedAlias('training.validation.accelerator_metrics_send_failed'))))
                raise
            finally:
                del request
        return request_id

    def _watch_stop(self) -> None:
        while not self._closed_event.wait(0.05):
            with self._condition:
                self._expire_reports_locked()
            if self.stop_event.is_set():
                self._fail(S3ConnectionError(constant_text(FixedAlias('training.validation.coordinator_session_stopped'))))
                return

    @neutral_protocol_text()
    def _send(self, message: dict) -> None:
        try:
            try:
                self.transport.send_message(message, profile="wr", max_frame=self.MAX_FRAME_BYTES)
            except (TypeError, ValueError, RecursionError):
                raise S3ProtocolError("WR request cannot be serialized within the frame limit") from None
        finally:
            del message

    def _unknown(self, frame: bytes, response: dict | None, reason: str) -> None:
        with self._condition:
            if self._closed_event.is_set():
                return
            self._authorized = False
            self._job = None
            self._generation += 1
            self._recording_unknown = True
            self._compatibility_failure = S3CompatibilityError(constant_text(FixedAlias('training.validation.batch_request_unknown')))
        diagnostic_failed = False
        if self._unknown_handler is not None:
            try:
                self._unknown_handler(frame, response, reason, fatal=True)
            except Exception:
                diagnostic_failed = True
        message = constant_text(FixedAlias('training.validation.batch_request_unknown'))
        if diagnostic_failed:
            message += '; diagnostic recording failed'
        error = S3CompatibilityError(message)
        with self._condition:
            self._failure = error
            self._compatibility_failure = error
        self.stop_event.set()
        self._fail(error)
        with self._condition:
            self._recording_unknown = False
            self._condition.notify_all()

    @neutral_protocol_text()
    def _read_loop(self) -> None:
        try:
            while not self._closed_event.is_set():
                try:
                    response = self.transport.receive_message(profile="wr", max_frame=self.MAX_FRAME_BYTES)
                except (ValueError, TypeError, UnicodeError, RecursionError):
                    frame = self.transport.message_frame()
                    try:
                        self._unknown(frame, None, "Malformed or oversized WR frame")
                    finally:
                        self.transport.discard_message_frame()
                        del frame
                    return
                except S3ConnectionError as error:
                    frame = self.transport.message_frame()
                    try:
                        if frame and not self._closed_event.is_set():
                            self._unknown(frame, None, "WR connection closed with an incomplete frame")
                        else:
                            self._fail(error)
                    finally:
                        self.transport.discard_message_frame()
                        del frame
                    return
                try:
                    if response is not None:
                        self._handle(response)
                except UnknownTrainingRequest as error:
                    frame = self.transport.message_frame()
                    try:
                        if self.settings.get("stop_on_unknown_request", True):
                            self._unknown(frame, response, str(error))
                            return
                        if self._unknown_handler is not None:
                            try:
                                self._unknown_handler(frame, response, str(error), fatal=False)
                            except Exception:
                                self._unknown(frame, response, "Could not record the unknown WR message")
                                return
                    finally:
                        del frame
                except S3ProtocolError:
                    frame = self.transport.message_frame()
                    try:
                        self._unknown(frame, response, "Malformed or unsupported WR message")
                        return
                    finally:
                        del frame
                finally:
                    self.transport.discard_message_frame()
                    del response
        except Exception:
            self._fail(S3ConnectionError("WR reader failed"))

    def _handle(self, response: dict) -> None:
        with self._condition:
            if self._closed_event.is_set():
                return
        kind = self.dialect.classify(response)
        if wire.text_matches('SERVER_SEED', kind):
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
                    self._fail(S3AuthenticationError(constant_text(FixedAlias('training.validation.batch_session_authentication_rejected'))))
                    return
                self._authorized = True
                self._condition.notify_all()
            elif request_id in self._report_pending:
                del self._report_pending[request_id]
                confirmed = self.dialect.accept(response)
                if request_id == self._latest_report_id:
                    self._latest_report_ok = confirmed
                if confirmed:
                    LOG.debug(constant_text(FixedAlias('training.log.accelerator_metrics_acknowledged')))
                else:
                    LOG.warning(constant_text(FixedAlias('training.log.accelerator_metrics_rejected')))
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
                if self._authorized and self._job is not None and (self._job.device_seed is not None):
                    return JobSnapshot(self._job, self._generation)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(min(remaining, 0.1))

    def _submission_result(self, response: dict) -> SubmissionResult:
        if response['status'] == 0 and response['result'] == SHAPE_REJECTED:
            return SubmissionResult(False, SHAPE_REJECTED)
        return super()._submission_result(response)

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
