import threading
import time
from urllib.request import Request

from .network import current_proxy, http_opener
from .vpn import current_vpn, VPNError


class JobEvents:
    def __init__(self, schedule, updates):
        self.schedule = schedule
        self.updates = updates
        self._active = None
        self._lock = threading.Lock()

    def update(self):
        number, kind, percentage, _ = self.schedule.position()
        if percentage == -1:
            self.finish()
            return
        current = number, "side" if kind == "side-job" else "main"
        with self._lock:
            if self._active == current:
                return
            if self._active is not None:
                self.updates.event("Job %d %s finished" % self._active)
            self._active = current
            self.updates.event("Job %d %s starting" % current)

    def finish(self):
        with self._lock:
            if self._active is not None:
                self.updates.event("Job %d %s finished" % self._active)
                self._active = None


class VPNJobGuard:
    def __init__(self, schedule, configuration, updates):
        self.schedule = schedule
        self.updates = updates
        self.proxy = current_proxy()
        self.manager = current_vpn()
        self.settings = configuration.get("vpn", {})
        self.enabled = self.proxy is not None and self.settings.get("enabled", False)
        self._until = 0
        self._active_until = None
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._thread = None
        self._session = None

    def start(self):
        if self.enabled and self.settings.get("check_url"):
            self._thread = threading.Thread(target=self._run, name="LRAEP-vpn-watch", daemon=True)
            self._thread.start()
        return self

    def attach(self, session):
        with self._lock:
            self._session = session
            blocked = time.monotonic() < self._until
        if blocked:
            session.close()

    def detach(self, session):
        with self._lock:
            if self._session is session:
                self._session = None

    def connection_lost(self, boundary=None):
        if not self.enabled:
            return False
        number, kind, percentage, until = self.schedule.position()
        now = time.monotonic()
        if kind != "main" or percentage <= 0 or (boundary is not None and (boundary != until or boundary != self._active_until)):
            return False
        with self._lock:
            if self._until == until:
                return True
            self._until = until
            session = self._session
        if session is not None:
            try:
                session.close()
            except Exception:
                pass
        self.updates.event(f"Job {number} main: VPN connection lost; waiting until the job finishes", direct=True)
        return now < until

    def blocked(self):
        return time.monotonic() < self._until

    def check(self):
        if not self.enabled:
            return False
        boundary = self.schedule.current_job()[2]
        if (boundary == self._active_until and self.manager is not None
                and (self.manager.process is None or self.manager.process.poll() is not None)):
            self.connection_lost(boundary)
        return self.blocked()

    def resume(self):
        if not self.enabled:
            return True
        if time.monotonic() < self._until:
            return False
        boundary = self.schedule.current_job()[2]
        if boundary == self._active_until and self.check():
            return False
        if self.manager is not None:
            try:
                self.manager.ensure_ready()
            except VPNError:
                self.connection_lost()
                return False
        self._active_until = boundary
        return True

    def wait(self, stopped, *, deadline=None, on_progress=None):
        while not stopped.is_set() and time.monotonic() < self._until:
            now = time.monotonic()
            until = self._until if deadline is None else min(self._until, deadline)
            if now >= until:
                return False
            if on_progress is not None:
                on_progress()
            stopped.wait(min(1.0, until - now))
        return not stopped.is_set()

    def _run(self):
        try:
            opener = http_opener(self.proxy)
        except Exception:
            self.connection_lost()
            return
        while not self._stopped.is_set():
            _, kind, percentage, boundary = self.schedule.position()
            core_ready = (self.manager is None or (self.manager.process is not None
                          and self.manager.process.poll() is None))
            if (kind == "main" and percentage > 0 and core_ready
                    and boundary == self._active_until and not self.blocked()):
                try:
                    with opener.open(Request(self.settings["check_url"]), timeout=5.0) as response:
                        response.read(256)
                except Exception:
                    if not self._stopped.is_set():
                        self.connection_lost(boundary)
            self._stopped.wait(5.0)

    def close(self):
        self._stopped.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5.5)

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
