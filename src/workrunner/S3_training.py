from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import gc
import logging
from pathlib import Path
import signal
import threading
import time
import numpy as np
import torch
from .gpu_noisy_gemm import TrainingCancelled
from .gpu_throttle import GpuPacer
from .duty_cycle import JobSequence, ScheduledSideJob
from .progress import LiveProgress
from .protocol import TrainingConfig, load_official_helpers, S3_training_threshold, prepare_batch, search_batch, verified_S3_submission
from .gpu_utilization import GpuUtilization
from .telegram import TelegramUpdates
from .vpn_session_guard import JobEvents, VPNJobGuard
from .progress import job_status
from . import native_cuda
from . import training_constants as wire
from .fixed_constants import FixedAlias, constant_text
from .training_identity import validate_login_identity

_STATUS_TEMPLATE = 'Training rate(avg) {} | GPU {} | accepted={} rejected={} | submitted={} batches={} attempts={} stale={}'
_LOG_CONNECTED = FixedAlias('training.log.coordinator_session_authenticated')
_LOG_JOB = FixedAlias('training.log.coordinator_batch_revision')
_LOG_JOB_VERIFIED = FixedAlias('training.log.batch_header_verified_no_submissions')
_LOG_RESULT_OK = FixedAlias('training.log.batch_submission_accepted')
_LOG_RESULT_FAILED = FixedAlias('training.log.batch_submission_rejected')
_LOG_STOPPED = FixedAlias('training.log.training_run_summary')

LOG = logging.getLogger(wire.LOGGER_NAME)
TRAINING_REPORT_INTERVAL = 10.0
TRAINING_STARTUP_REPORT_INTERVAL = 2.0
TRAINING_STARTUP_REPORT_WINDOW = 15.0

@dataclass
class S3TrainingStats:
    batches: int = 0
    attempts: int = 0
    sent: int = 0
    confirmed: int = 0
    declined: int = 0
    outdated: int = 0

def S3_training_rate(stats: S3TrainingStats, cfg: TrainingConfig, elapsed: float) -> float:
    return stats.batches * cfg.m * cfg.n * cfg.k / max(elapsed, 0.001)

def format_training_rate(rate: float) -> str:
    units = wire.RATE_UNITS
    unit = 0
    while rate >= 1000 and unit < len(units) - 1:
        rate /= 1000
        unit += 1
    return f'{rate:.2f} {units[unit]}'

def S3_status(stats: S3TrainingStats, cfg: TrainingConfig, elapsed: float, gpu_utilization: int | None) -> str:
    rate = format_training_rate(S3_training_rate(stats, cfg, elapsed))
    utilization = 'N/A' if gpu_utilization is None else f'{gpu_utilization}%'
    return _STATUS_TEMPLATE.format(rate, utilization, stats.confirmed, stats.declined, stats.sent, stats.batches, stats.attempts, stats.outdated)

def select_S3_protocol(settings: dict) -> str:
    selected = settings.get('protocol', 'KEX')
    if selected == 'auto':
        raise ValueError(constant_text(FixedAlias('training.validation.coordinator_protocol_explicit_selection')))
    if selected == constant_text(FixedAlias('training.config.legacy_batch_protocol')) or wire.text_matches('LEGACY_MODE', selected):
        selected = 'WR'
    if selected not in ('KEX', 'WR'):
        raise ValueError(constant_text(FixedAlias('training.validation.coordinator_protocol_unsupported')))
    return selected

def validate_S3_endpoint(settings: dict) -> None:
    host = settings.get('host', '')
    if not isinstance(host, str) or not host.strip():
        selected = settings.get('protocol')
        legacy_protocol = constant_text(FixedAlias('training.config.legacy_batch_protocol'))
        uses_wr = selected in ('WR', legacy_protocol) or wire.text_matches('LEGACY_MODE', selected)
        profile = 'WR' if uses_wr else 'KEX'
        prefix = constant_text(FixedAlias('training.validation.coordinator_host_config_prefix'))
        suffix = constant_text(FixedAlias('training.validation.coordinator_hostname_suffix'))
        raise ValueError(f'{prefix}{profile}{suffix}')

def validate_S3_identity(settings: dict, selected_protocol: str | None=None, *, validate_identity: bool=True) -> None:
    """Check the gateway login identity before starting a training session."""
    selected = select_S3_protocol(settings if selected_protocol is None else {'protocol': selected_protocol})
    validate_login_identity(settings, selected, validate_identity=validate_identity)

@contextmanager
def _signal_stop(stopped, time_limit=0):
    previous = {}
    timer = None
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, lambda *_: stopped.set())
    if time_limit > 0:
        timer = threading.Timer(time_limit, stopped.set)
        timer.daemon = True
        timer.start()
    try:
        yield
    finally:
        stopped.set()
        if timer is not None:
            timer.cancel()
        for sig, handler in previous.items():
            signal.signal(sig, handler)

def run_S3_training(configuration: dict, *, login_only: bool=False, login_config=None) -> int:
    settings = configuration['training']
    stopped = threading.Event()
    started = time.monotonic()
    effort = None if login_only else JobSequence(settings['job_sequence'], started=started)
    deadline = started + settings['max_seconds'] if settings['max_seconds'] else None
    with _signal_stop(stopped, settings['max_seconds']), ExitStack() as resources:
        if effort is not None:
            resources.enter_context(effort.stopping(stopped))
            if not effort.wait(stopped, deadline=deadline):
                return 0
        return _run_S3_training(configuration, login_only=login_only, login_config=login_config,
                                stopped=stopped, effort=effort, started=started)

def _release_work(device):
    gc.collect()
    if torch.cuda.is_initialized():
        with torch.cuda.device(device):
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()


def _run_S3_training(configuration, *, login_only, login_config, stopped, effort, started):
    from .config import resolve_S3_settings
    from .S3_client import S3AuthenticationError, S3CompatibilityError, S3Session, ReplacedS3Job
    from .S3_transport import S3ConnectionError
    S3 = resolve_S3_settings(configuration[constant_text(FixedAlias('training.config.coordinator_section'))])
    selected_protocol = select_S3_protocol(S3)
    validate_S3_endpoint(S3)
    if login_config is None:
        validate_S3_identity(S3, selected_protocol, validate_identity=False)
    elif selected_protocol != 'WR':
        raise ValueError('A separate login config requires WR')
    session_class = S3Session
    session_options = {}
    if selected_protocol == 'WR':
        from .S3_diagnostics import UnknownRequestRecorder
        from .training_client import TrainingSession
        from .training_work import prepare_training_batch, verified_training_submission
        session_class = TrainingSession
        diagnostic_dir = S3.get('unknown_request_dir') or Path(configuration['training']['output_dir']) / constant_text(FixedAlias('training.path.unrecognized_batch_messages'))
        session_options['unknown_handler'] = UnknownRequestRecorder(diagnostic_dir, S3)
        session_options['login_config'] = login_config
    load_official_helpers(configuration['official_source'])
    if login_config is None:
        validate_S3_identity(S3, selected_protocol)
    from workrunner import protocol_types as pm
    settings = configuration['training']
    cfg = TrainingConfig(settings['m'], settings['n'], settings['k'], settings['tile_h'], settings['tile_w'])
    gpu_name = None
    if not login_only:
        device = torch.device(configuration['device'])
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise ValueError(constant_text(FixedAlias('training.validation.accelerator_required')))
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        gpu_name = torch.cuda.get_device_name(device)
        LOG.debug('GPU: %s', gpu_name)
    use_native = not login_only and (cfg.tile_h, cfg.tile_w) == (8, 8) and native_cuda.available(configuration['device'])
    if not login_only:
        LOG.debug('S1 backend: %s', 'native CUDA DP4A' if use_native else 'PyTorch reference')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    stats = S3TrainingStats()
    rng = np.random.default_rng()
    limit = settings['max_seconds']
    max_batches = settings['max_batches']
    pacer = GpuPacer(device, 100) if not login_only else None
    deadline = started + limit if limit else None

    def expired():
        return stopped.is_set() or (limit > 0 and time.monotonic() - started >= limit) or (effort is not None and effort.ended())

    def finished():
        return expired() or (max_batches > 0 and stats.batches >= max_batches)
    with ExitStack() as resources:
        progress = resources.enter_context(LiveProgress())
        monitor = None
        if not login_only:
            monitor = GpuUtilization(str(device)).start()
            resources.callback(monitor.close)
            pacer.set_monitor(monitor.read_sample)

        active_job = 'job-0'

        def show_side_job(job):
            update_progress()

        side_job = (ScheduledSideJob(configuration, before_log=progress.clear, on_progress=show_side_job)
                    if not login_only else None)
        if side_job is not None:
            resources.callback(side_job.close)

        def telegram_snapshot():
            now = time.monotonic()
            number, kind, percentage, boundary = effort.position(now)
            remaining = (0 if percentage == -1 else
                         max(0, (boundary if deadline is None else min(boundary, deadline)) - now))
            if kind == "job-1":
                job = side_job.job if side_job is not None else None
                speed = getattr(job, "_progress_epoch", 0) if job is not None else 0
            else:
                speed = (0 if percentage <= 0 or (vpn_guard is not None and vpn_guard.blocked())
                         else S3_training_rate(stats, cfg, now - started))
            return job_status(number, kind,
                              monitor.read(), remaining, speed)

        updates = None
        events = None
        vpn_guard = None
        if not login_only:
            updates = resources.enter_context(TelegramUpdates(configuration, telegram_snapshot))
            events = JobEvents(effort, updates)
            resources.callback(events.finish)
            vpn_guard = resources.enter_context(VPNJobGuard(effort, configuration, updates))

        def update_progress(force=False):
            nonlocal active_job
            if monitor is not None:
                if events is not None:
                    events.update()
                kind = effort.current_job()[0]
                force = force or active_job != kind
                active_job = kind
                progress.update(telegram_snapshot(), force=force)

        def log(level, message, *args):
            LOG.debug(message, *args)

        def report_gpu(session, sample_time):
            rate = S3_training_rate(stats, cfg, sample_time - started)
            request_id = session.send_device_report(gpu_name, rate)
            log(logging.INFO, constant_text(FixedAlias('training.log.accelerator_metrics_sent')), format_training_rate(rate), request_id)
        update_progress(force=True)
        while not finished():
            if effort is not None and (not effort.wait(stopped, deadline=deadline)):
                break
            if side_job is not None and side_job.run_if_active(effort, stopped, pacer, deadline=deadline):
                continue
            if vpn_guard is not None and not vpn_guard.resume():
                vpn_guard.wait(stopped, deadline=deadline, on_progress=update_progress)
                continue
            session = None
            retry_delay = None
            wait_for_vpn = False
            session_boundary = effort.current_job()[2] if effort is not None else None
            try:
                session_settings = dict(S3)
                connect_deadline = deadline
                if effort is not None and effort.exit_deadline is not None:
                    connect_deadline = effort.exit_deadline if connect_deadline is None else min(connect_deadline, effort.exit_deadline)
                if effort is not None:
                    boundary = effort.current_job()[2]
                    connect_deadline = boundary if connect_deadline is None else min(boundary, connect_deadline)
                if connect_deadline is not None:
                    remaining = max(0.001, connect_deadline - time.monotonic())
                    session_settings['connect_timeout'] = min(S3['connect_timeout'], remaining)
                session = session_class(session_settings, stop_event=stopped, **session_options)
                if vpn_guard is not None:
                    vpn_guard.attach(session)
                with session:
                    log(logging.INFO, constant_text(_LOG_CONNECTED), S3['host'], S3['port'], 'gzip' if session.negotiated_gzip else 'none')
                    first_work_report = last_work_report = None
                    if selected_protocol == 'WR' and (not login_only):
                        report_gpu(session, time.monotonic())
                    previous_generation = None
                    previous_effort = None
                    job_deadline = time.monotonic() + S3['read_timeout']
                    while not finished():
                        if effort is not None and effort.current_job()[2] != session_boundary:
                            break
                        if vpn_guard is not None and vpn_guard.check():
                            break
                        if effort is not None:
                            kind, utilization, boundary = effort.current_job()
                            if boundary != session_boundary or kind != "job-0" or utilization <= 0:
                                break
                        update_progress()
                        wait_after_batch = False
                        try:
                            snapshot = session.wait_job(timeout=0.25)
                            if snapshot is None:
                                if time.monotonic() >= job_deadline:
                                    raise S3ConnectionError(constant_text(FixedAlias('training.validation.coordinator_batch_timeout')))
                                continue
                            job = snapshot.job
                            if snapshot.generation != previous_generation:
                                if selected_protocol == 'WR':
                                    log(logging.INFO, constant_text(FixedAlias('training.log.batch_generation')), job.job_id, job.cert_version)
                                else:
                                    log(logging.INFO, constant_text(_LOG_JOB), job.job_id, job.height, job.cert_version)
                                previous_generation = snapshot.generation
                                job_deadline = time.monotonic() + S3['read_timeout']
                            header = pm.PartialHeader.from_bytes(job.header_bytes)
                            training_threshold = S3_training_threshold(job.training_threshold, cfg, 'base')
                            if login_only:
                                log(logging.INFO, constant_text(_LOG_JOB_VERIFIED), len(job.header_bytes), training_threshold.bit_length())
                                return 0
                            kind, utilization, boundary = effort.current_job()
                            if kind != 'job-0' or utilization <= 0 or expired():
                                continue
                            if pacer.max_utilization != utilization:
                                pacer.set_utilization(utilization)
                            if previous_effort != (kind, utilization):
                                log(logging.INFO, '%s effort GPU duty-cycle target: %d%%', kind, utilization)
                                previous_effort = (kind, utilization)
                            rest_deadline = min(boundary, deadline) if deadline is not None else boundary
                            cancelled = lambda: (expired() or time.monotonic() >= boundary
                                                 or (vpn_guard is not None and vpn_guard.check())
                                                 or not session.is_current(snapshot))
                            pacer.begin()
                            if selected_protocol == 'WR':
                                batch = prepare_training_batch(job, cfg, rng, device=configuration['device'], native=use_native)
                            else:
                                batch = prepare_batch(header, job.cert_version, cfg, rng, device=configuration['device'], gpu_signals=True, native=use_native)
                            if cancelled():
                                pacer.finish(cancelled=True)
                                wait_after_batch = vpn_guard is None or not vpn_guard.check()
                                continue
                            try:
                                matches = search_batch(batch, training_threshold, configuration['device'], cancelled)
                            except TrainingCancelled:
                                pacer.finish(cancelled=True)
                                wait_after_batch = vpn_guard is None or not vpn_guard.check()
                                continue
                            pacer.finish()
                            stats.batches += 1
                            stats.attempts += cfg.m // cfg.tile_h * (cfg.n // cfg.tile_w)
                            if selected_protocol == 'WR':
                                now = time.monotonic()
                                if first_work_report is None:
                                    first_work_report = now
                                interval = (TRAINING_STARTUP_REPORT_INTERVAL
                                            if now - first_work_report < TRAINING_STARTUP_REPORT_WINDOW
                                            else TRAINING_REPORT_INTERVAL)
                                if last_work_report is None or now - last_work_report >= interval:
                                    report_gpu(session, now)
                                    last_work_report = now
                            for row, col, digest in matches:
                                if cancelled():
                                    stats.outdated += 1
                                    break
                                if selected_protocol == 'WR':
                                    submission = verified_training_submission(batch, row, col, digest, training_threshold)
                                else:
                                    proof, _ = verified_S3_submission(batch, header, job.cert_version, row, col, digest, training_threshold)
                                    submission = proof.to_base64()
                                if cancelled():
                                    stats.outdated += 1
                                    break
                                try:
                                    result = session.send_result(snapshot, submission)
                                except ReplacedS3Job:
                                    stats.outdated += 1
                                    break
                                stats.sent += 1
                                if result.confirmed:
                                    stats.confirmed += 1
                                    log(logging.INFO, constant_text(_LOG_RESULT_OK), job.job_id)
                                else:
                                    stats.declined += 1
                                    log(logging.WARNING, constant_text(_LOG_RESULT_FAILED), job.job_id, result.error)
                            update_progress()
                            wait_after_batch = not finished()
                        finally:
                            batch = matches = proof = submission = result = digest = None
                            header = job = snapshot = cancelled = training_threshold = row = col = _ = None
                            if wait_after_batch:
                                pacer.wait(stopped, deadline=rest_deadline)
                    if selected_protocol == 'WR':
                        session.raise_if_failed()
            except (S3AuthenticationError, S3CompatibilityError):
                raise
            except S3ConnectionError as exc:
                if selected_protocol == 'WR' and session is not None:
                    try:
                        session.raise_if_failed()
                    except S3CompatibilityError:
                        raise
                    except S3ConnectionError:
                        pass
                if stopped.is_set():
                    break
                if login_only:
                    raise
                if finished():
                    break
                log(logging.WARNING, '%s; reconnecting in %.1f seconds', exc, S3['reconnect_delay'])
                exc.__traceback__ = None
                wait_for_vpn = vpn_guard is not None and vpn_guard.connection_lost(session_boundary)
                delay = S3['reconnect_delay']
                if limit > 0:
                    delay = min(delay, max(0, limit - (time.monotonic() - started)))
                retry_delay = delay
            finally:
                if vpn_guard is not None:
                    vpn_guard.detach(session)
                session = None
                if not login_only:
                    _release_work(device)
            if wait_for_vpn:
                vpn_guard.wait(stopped, deadline=deadline, on_progress=update_progress)
            elif retry_delay is not None:
                stopped.wait(retry_delay)
        update_progress(force=True)
    LOG.debug(constant_text(_LOG_STOPPED), stats.batches, stats.sent, stats.confirmed, stats.declined, stats.outdated)
    if login_only:
        raise RuntimeError(constant_text(FixedAlias('training.validation.authenticated_batch_required')))
    return 0
