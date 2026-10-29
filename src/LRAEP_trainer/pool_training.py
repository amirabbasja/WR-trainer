from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import logging
from pathlib import Path
import signal
import threading
import time
import numpy as np
import torch
from .gpu import TrainingCancelled
from .gpu_pacing import GpuPacer
from .effort import JobSequence
from .progress import LiveProgress
from .protocol import TrainingConfig, load_official_helpers, pool_share_target, prepare_batch, search_batch, verified_pool_share
from .telemetry import GpuUtilization
from . import native_cuda
LOG = logging.getLogger('LRAEP_trainer')
WR_REPORT_INTERVAL = 120.0
WR_STARTUP_REPORT_INTERVAL = 10.0
WR_STARTUP_REPORT_WINDOW = 120.0

@dataclass
class PoolTrainingStats:
    batches: int = 0
    attempts: int = 0
    submitted: int = 0
    accepted: int = 0
    rejected: int = 0
    stale: int = 0

def pool_hashrate(stats: PoolTrainingStats, cfg: TrainingConfig, elapsed: float) -> float:
    return stats.batches * cfg.m * cfg.n * cfg.k / max(elapsed, 0.001)

def format_hashrate(rate: float) -> str:
    units = ('H/s', 'kH/s', 'MH/s', 'GH/s', 'TH/s', 'PH/s', 'EH/s')
    unit = 0
    while rate >= 1000 and unit < len(units) - 1:
        rate /= 1000
        unit += 1
    return f'{rate:.2f} {units[unit]}'

def pool_status(stats: PoolTrainingStats, cfg: TrainingConfig, elapsed: float, gpu_utilization: int | None) -> str:
    rate = format_hashrate(pool_hashrate(stats, cfg, elapsed))
    utilization = 'N/A' if gpu_utilization is None else f'{gpu_utilization}%'
    return f'Hashrate(avg) {rate} | GPU {utilization} | accepted={stats.accepted} rejected={stats.rejected} | submitted={stats.submitted} batches={stats.batches} attempts={stats.attempts} stale={stats.stale}'

def select_pool_protocol(settings: dict) -> str:
    selected = settings.get('protocol', 'KEX')
    if selected == 'auto':
        raise ValueError('pool.protocol must explicitly select KEX or WR; automatic detection is unavailable')
    if selected == 'pearlhash':
        selected = 'WR'
    if selected not in ('KEX', 'WR'):
        raise ValueError('pool.protocol must select KEX or WR')
    return selected

def validate_pool_endpoint(settings: dict) -> None:
    host = settings.get('host', '')
    if not isinstance(host, str) or not host.strip():
        profile = 'WR' if settings.get('protocol') in ('WR', 'pearlhash') else 'KEX'
        raise ValueError(f'fill pool.host for a flat configuration, or pool.{profile}.host in the shared configuration, with your pool hostname or IP address')

def validate_pool_identity(settings: dict, selected_protocol: str | None=None, *, validate_address: bool=True) -> None:
    selected = select_pool_protocol(settings if selected_protocol is None else {'protocol': selected_protocol})
    identity_field = 'WR_wallet' if selected == 'WR' else 'KEX_uname'
    identity = settings.get(identity_field, '')
    worker = settings.get('worker', '')
    if not isinstance(identity, str) or not identity.strip():
        description = 'WR payout address' if selected == 'WR' else 'KEX username or payout address'
        raise ValueError(f'fill pool.{identity_field} in your configuration with your {description}')
    if not isinstance(worker, str):
        raise ValueError('pool.worker must be a string')
    if any((character.isspace() for character in identity + worker)):
        raise ValueError(f'pool.{identity_field} and pool.worker must not contain whitespace')
    if identity.startswith('solo:'):
        raise ValueError('remove the solo: prefix to use the requested pool training mode')
    if validate_address and identity.startswith('prl1'):
        from LRAEP_gateway.blockchain_utils.blockchain_utils import get_script_pubkey_from_p2tr_address
        try:
            get_script_pubkey_from_p2tr_address(identity)
        except ValueError:
            raise ValueError(f'pool.{identity_field} must contain a valid PEARL Taproot address') from None

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

def run_pool_training(configuration: dict, *, login_only: bool=False) -> int:
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
        return _run_pool_training(configuration, login_only=login_only, stopped=stopped, effort=effort, started=started)

def _run_pool_training(configuration, *, login_only, stopped, effort, started):
    from .config import resolve_pool_settings
    from .pool_client import PoolAuthenticationError, PoolCompatibilityError, PoolSession, StalePoolJob
    from .pool_transport import PoolConnectionError
    pool = resolve_pool_settings(configuration['pool'])
    selected_protocol = select_pool_protocol(pool)
    validate_pool_endpoint(pool)
    validate_pool_identity(pool, selected_protocol, validate_address=False)
    session_class = PoolSession
    session_options = {}
    if selected_protocol == 'WR':
        from .pool_diagnostics import UnknownRequestRecorder
        from .wr_client import WRSession
        from .wr_work import prepare_wr_batch, verified_wr_share
        session_class = WRSession
        diagnostic_dir = pool.get('unknown_request_dir') or Path(configuration['training']['output_dir']) / 'unknown-pool-requests'
        session_options['unknown_handler'] = UnknownRequestRecorder(diagnostic_dir, pool)
    load_official_helpers(configuration['official_source'])
    validate_pool_identity(pool, selected_protocol)
    from LRAEP_trainer import primitives as pm
    settings = configuration['training']
    cfg = TrainingConfig(settings['m'], settings['n'], settings['k'], settings['tile_h'], settings['tile_w'])
    gpu_name = None
    if not login_only:
        device = torch.device(configuration['device'])
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise ValueError('an available CUDA GPU is required for pool training')
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        gpu_name = torch.cuda.get_device_name(device)
        LOG.info('GPU: %s', gpu_name)
    use_native = not login_only and (cfg.tile_h, cfg.tile_w) == (8, 8) and native_cuda.available(configuration['device'])
    if not login_only:
        LOG.info('Mining backend: %s', 'native CUDA DP4A' if use_native else 'PyTorch reference')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    stats = PoolTrainingStats()
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

        def update_progress(force=False):
            if monitor is not None:
                progress.update(pool_status(stats, cfg, time.monotonic() - started, monitor.read()), force=force)

        def log(level, message, *args):
            progress.clear()
            LOG.log(level, message, *args)

        def report_gpu(session, sample_time):
            rate = pool_hashrate(stats, cfg, sample_time - started)
            request_id = session.report_info(gpu_name, rate)
            log(logging.INFO, 'PearlHash GPU report sent: %s (request=%d)', format_hashrate(rate), request_id)
        update_progress(force=True)
        while not finished():
            if effort is not None and (not effort.wait(stopped, deadline=deadline)):
                break
            session = None
            try:
                session_settings = dict(pool)
                connect_deadline = deadline
                if effort is not None and effort.exit_deadline is not None:
                    connect_deadline = effort.exit_deadline if connect_deadline is None else min(connect_deadline, effort.exit_deadline)
                if connect_deadline is not None:
                    remaining = max(0.001, connect_deadline - time.monotonic())
                    session_settings['connect_timeout'] = min(pool['connect_timeout'], remaining)
                session = session_class(session_settings, stop_event=stopped, **session_options)
                with session:
                    log(logging.INFO, 'Pool authenticated at %s:%d; proof compression=%s', pool['host'], pool['port'], 'gzip' if session.negotiated_gzip else 'none')
                    report_connection_started = time.monotonic()
                    report_started = report_connection_started
                    reported_work = False
                    if selected_protocol == 'WR' and (not login_only):
                        report_gpu(session, report_started)
                    previous_generation = None
                    previous_effort = None
                    job_deadline = time.monotonic() + pool['read_timeout']
                    while not finished():
                        if effort is not None:
                            paused = effort.current()[0] == 0
                            if not effort.wait(stopped, deadline=deadline):
                                break
                            if paused:
                                job_deadline = time.monotonic() + pool['read_timeout']
                        update_progress()
                        snapshot = session.wait_job(timeout=0.25)
                        if snapshot is None:
                            if time.monotonic() >= job_deadline:
                                raise PoolConnectionError('pool sent no usable training job before the job timeout')
                            continue
                        job = snapshot.job
                        if snapshot.generation != previous_generation:
                            if selected_protocol == 'WR':
                                log(logging.INFO, 'PearlHash job=%s generation=%d', job.job_id, job.cert_version)
                            else:
                                log(logging.INFO, 'Pool job=%s height=%d certificate=V%d', job.job_id, job.height, job.cert_version)
                            previous_generation = snapshot.generation
                            job_deadline = time.monotonic() + pool['read_timeout']
                        header = pm.IncompleteBlockHeader.from_bytes(job.header_bytes)
                        target = pool_share_target(job.target, cfg, 'base')
                        if login_only:
                            log(logging.INFO, 'Authenticated job verified: header=%d bytes, target=%d bits; no GPU work or shares submitted', len(job.header_bytes), target.bit_length())
                            return 0
                        kind, utilization, boundary = effort.current_job()
                        if utilization <= 0 or expired():
                            continue
                        if pacer.max_utilization != utilization:
                            pacer.set_utilization(utilization)
                        if previous_effort != (kind, utilization):
                            log(logging.INFO, '%s effort GPU duty-cycle target: %d%%', kind.capitalize(), utilization)
                            previous_effort = (kind, utilization)
                        rest_deadline = min(boundary, deadline) if deadline is not None else boundary
                        cancelled = lambda: expired() or time.monotonic() >= boundary or (not session.is_current(snapshot))
                        pacer.begin()
                        if selected_protocol == 'WR':
                            batch = prepare_wr_batch(job, cfg, rng, device=configuration['device'], native=use_native)
                        else:
                            batch = prepare_batch(header, job.cert_version, cfg, rng, device=configuration['device'], gpu_signals=True, native=use_native)
                        if cancelled():
                            pacer.finish(cancelled=True)
                            pacer.wait(stopped, deadline=rest_deadline)
                            continue
                        try:
                            matches = search_batch(batch, target, configuration['device'], cancelled)
                        except TrainingCancelled:
                            pacer.finish(cancelled=True)
                            pacer.wait(stopped, deadline=rest_deadline)
                            continue
                        pacer.finish()
                        stats.batches += 1
                        stats.attempts += cfg.m // cfg.tile_h * (cfg.n // cfg.tile_w)
                        if selected_protocol == 'WR':
                            now = time.monotonic()
                            interval = WR_STARTUP_REPORT_INTERVAL if now - report_connection_started < WR_STARTUP_REPORT_WINDOW else WR_REPORT_INTERVAL
                            if not reported_work or now - report_started >= interval:
                                report_gpu(session, now)
                                report_started = now
                                reported_work = True
                        for row, col, digest in matches:
                            if cancelled():
                                stats.stale += 1
                                break
                            if selected_protocol == 'WR':
                                submission = verified_wr_share(batch, row, col, digest, target)
                            else:
                                proof, _ = verified_pool_share(batch, header, job.cert_version, row, col, digest, target)
                                submission = proof.to_base64()
                            if cancelled():
                                stats.stale += 1
                                break
                            try:
                                result = session.submit(snapshot, submission)
                            except StalePoolJob:
                                stats.stale += 1
                                break
                            stats.submitted += 1
                            if result.accepted:
                                stats.accepted += 1
                                log(logging.INFO, 'Pool accepted training share job=%s', job.job_id)
                            else:
                                stats.rejected += 1
                                log(logging.WARNING, 'Pool rejected training share job=%s: %s', job.job_id, result.error)
                        update_progress()
                        if not finished():
                            pacer.wait(stopped, deadline=rest_deadline)
                    if selected_protocol == 'WR':
                        session.raise_if_failed()
            except (PoolAuthenticationError, PoolCompatibilityError):
                raise
            except PoolConnectionError as exc:
                if selected_protocol == 'WR' and session is not None:
                    try:
                        session.raise_if_failed()
                    except PoolCompatibilityError:
                        raise
                    except PoolConnectionError:
                        pass
                if stopped.is_set():
                    break
                if login_only:
                    raise
                if finished():
                    break
                log(logging.WARNING, '%s; reconnecting in %.1f seconds', exc, pool['reconnect_delay'])
                delay = pool['reconnect_delay']
                if limit > 0:
                    delay = min(delay, max(0, limit - (time.monotonic() - started)))
                stopped.wait(delay)
        update_progress(force=True)
    LOG.info('Pool training stopped: batches=%d submitted=%d accepted=%d rejected=%d stale=%d', stats.batches, stats.submitted, stats.accepted, stats.rejected, stats.stale)
    if login_only:
        raise RuntimeError('no authenticated pool job was received before stopping')
    return 0
