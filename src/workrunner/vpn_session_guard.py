import threading
import time
from urllib.request import Request

from .duty_cycle import MINING_JOBS

from .socks_proxy import current_proxy, http_opener
from .vpn import current_vpn, VPNError


class VPNJobGuard:
    """Pause main work on connection loss and retry on the training thread."""

    def __init__(self, schedule, configuration, updates):
        self.schedule = schedule
        self.updates = updates
        self.proxy = current_proxy()
        self.manager = current_vpn()
        self.settings = configuration.get("vpn", {})
        self.enabled = self.proxy is not None and self.settings.get("enabled", False)
        self.retry_delay = max(0.1, float(configuration.get("S3", {}).get("reconnect_delay", 3)))
        self._active_until = None
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._thread = None
        self._session = None
        self._verified = False
        self._disconnected = False
        self._retrying = False
        self._retry_at = 0
        self._generation = 0
        self._latency_ms = None
        self._latency_boundary = None

    def start(self):
        if self.enabled and self.settings.get("check_url"):
            self._thread = threading.Thread(target=self._run, name="WR-vpn-watch", daemon=True)
            self._thread.start()
        return self

    def attach(self, session):
        with self._lock:
            self._session = session
            blocked = self.enabled and not self._verified
        if blocked:
            session.close()

    def detach(self, session):
        with self._lock:
            if self._session is session:
                self._session = None

    def connection_lost(self, boundary=None, *, generation=None):
        if not self.enabled:
            return False
        number, kind, percentage, until = self.schedule.position()
        with self._lock:
            if (kind not in MINING_JOBS or percentage <= 0
                    or (boundary is not None and (boundary != until or boundary != self._active_until))
                    or (generation is not None and generation != self._generation)):
                return False
            if self._disconnected:
                return True
            self._disconnected = True
            self._verified = False
            self._generation += 1
            self._retry_at = time.monotonic() + self.retry_delay
            self._latency_ms = None
            self._latency_boundary = None
            session = self._session
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
        self.updates.event(f"Job {number} {kind}: VPN connection lost; hashing paused; retrying connection", direct=True)
        return True

    def blocked(self):
        with self._lock:
            return self.enabled and not self._verified

    def _core_ready(self):
        return self.manager is None or (self.manager.process is not None
                                       and self.manager.process.poll() is None)

    def status(self):
        """Describe cached connection state without sending a request."""
        if not self.enabled:
            return "Disabled"
        _, kind, percentage, boundary = self.schedule.position()
        if kind not in MINING_JOBS or percentage <= 0:
            return "Idle"
        with self._lock:
            if self._disconnected or not self._core_ready():
                if self._retrying:
                    return "Disconnected; retrying"
                remaining = max(0, self._retry_at - time.monotonic())
                return f"Disconnected; retrying in {remaining:.1f}s"
            if self._retrying or not self._verified or boundary != self._active_until:
                return "Checking"
            if not self.settings.get("check_url"):
                return "Connected (latency unavailable)"
            if self._latency_ms is None or self._latency_boundary != boundary:
                return "Checking"
            return f"Connected ({self._latency_ms:.0f} ms)"

    def check(self):
        if not self.enabled:
            return False
        boundary = self.schedule.current_job()[2]
        if boundary == self._active_until and not self._core_ready():
            self.connection_lost(boundary)
        return self.blocked()

    def _probe(self):
        if not self.settings.get("check_url"):
            return None
        started = time.perf_counter()
        with http_opener(self.proxy).open(Request(self.settings["check_url"]), timeout=5.0) as response:
            response.read(256)
        return (time.perf_counter() - started) * 1000

    def resume(self):
        if not self.enabled:
            return True
        kind, percentage, boundary = self.schedule.current_job()
        if self._stopped.is_set() or kind not in MINING_JOBS or percentage <= 0:
            return False
        with self._lock:
            if self._disconnected and time.monotonic() < self._retry_at:
                return False
            if self._verified and boundary == self._active_until and self._core_ready():
                return True
            self._active_until = boundary
            self._verified = False
            self._retrying = True
            self._generation += 1
            generation = self._generation
        try:
            # ManagedVPN uses context-local routing tokens. Restart it only on
            # the training thread, where its original context was entered.
            if self.manager is not None:
                self.manager.ensure_ready()
            latency_ms = self._probe()
            if not self._core_ready():
                raise VPNError("VPN core stopped during the connection check")
        except Exception:
            if not self._stopped.is_set():
                self.connection_lost(boundary, generation=generation)
                with self._lock:
                    self._retry_at = time.monotonic() + self.retry_delay
            return False
        else:
            with self._lock:
                if (self._stopped.is_set() or generation != self._generation
                        or boundary != self.schedule.current_job()[2]):
                    return False
                self._latency_ms = latency_ms
                self._latency_boundary = boundary
                self._disconnected = False
                self._verified = True
            return True
        finally:
            with self._lock:
                self._retrying = False

    def wait(self, stopped, *, deadline=None, on_progress=None):
        while not stopped.is_set() and not self._stopped.is_set():
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                return False
            kind, percentage, boundary = self.schedule.current_job()
            if kind not in MINING_JOBS or percentage <= 0 or not self.blocked():
                return True
            if on_progress is not None:
                on_progress()
            with self._lock:
                until = min(boundary, self._retry_at)
            if deadline is not None:
                until = min(until, deadline)
            if now >= until:
                return True
            stopped.wait(min(1.0, until - now))
        return False

    def _run(self):
        while not self._stopped.is_set():
            _, kind, percentage, boundary = self.schedule.position()
            with self._lock:
                probe = (kind in MINING_JOBS and percentage > 0 and self._verified
                         and boundary == self._active_until and not self._retrying)
                generation = self._generation
            if probe:
                try:
                    if not self._core_ready():
                        raise VPNError("VPN core stopped")
                    latency_ms = self._probe()
                    with self._lock:
                        # An old probe must never clear a newer failure or retry.
                        if (generation == self._generation and self._verified
                                and boundary == self._active_until == self.schedule.current_job()[2]):
                            self._latency_ms = latency_ms
                            self._latency_boundary = boundary
                except Exception:
                    if not self._stopped.is_set():
                        self.connection_lost(boundary, generation=generation)
            self._stopped.wait(5.0)

    def close(self):
        self._stopped.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5.5)

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
