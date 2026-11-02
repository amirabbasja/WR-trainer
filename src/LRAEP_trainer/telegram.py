import collections
import json
import logging
import threading
import time
from urllib.request import Request

from .network import current_proxy, http_opener


LOG = logging.getLogger("LRAEP_trainer")
_REQUEST_TIMEOUT = 5.0


class TelegramUpdates:
    def __init__(self, configuration, snapshot):
        settings = configuration.get("telegram", {})
        self._enabled = settings.get("enabled", False)
        self._name = configuration.get("trainer_name", "trainer")
        self._token = settings.get("tgBot_key", "")
        self._heartbeat_token = settings.get("heartbeat_tgBot_key", "")
        self._chat_id = settings.get("tgChatID", "")
        self._interval = settings.get("update_interval_seconds", 300)
        self._snapshot = snapshot
        self._proxy = current_proxy()
        self._stopped = threading.Event()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self._events = collections.deque()
        self._thread = None
        self._close_deadline = None

    def start(self):
        with self._lock:
            if self._enabled and self._thread is None and not self._stopped.is_set():
                self._thread = threading.Thread(
                    target=self._run, name="LRAEP-telegram-updates", daemon=True,
                )
                self._thread.start()
        return self

    def event(self, message, *, direct=False):
        with self._lock:
            if not self._enabled or self._stopped.is_set():
                return
            self._events.append((f"{self._name}\n{message}"[:4000], direct, 0.0))
            self._wake.set()

    def _request(self, opener, token, text):
        request = Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=json.dumps({"chat_id": self._chat_id, "text": text[:4000]}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        timeout = _REQUEST_TIMEOUT
        if self._close_deadline is not None:
            timeout = min(timeout, self._close_deadline - time.monotonic())
            if timeout <= 0:
                raise TimeoutError("Telegram delivery time expired")
        with opener.open(request, timeout=timeout) as response:
            result = json.loads(response.read(65536))
        if not isinstance(result, dict) or result.get("ok") is not True:
            raise ValueError("Telegram rejected the update")

    def _send(self, opener):
        if self._heartbeat_token:
            self._request(opener, self._heartbeat_token, str(self._snapshot()))

    def _run(self):
        opener = None
        try:
            opener = http_opener(self._proxy)
        except Exception:
            LOG.warning("Telegram route could not initialize; delivery will be retried")
        next_update = time.monotonic() + self._interval
        while True:
            now = time.monotonic()
            closing = self._stopped.is_set()
            with self._lock:
                self._wake.clear()
                events = []
                deferred = collections.deque()
                while self._events:
                    item = self._events.popleft()
                    (events if closing or now >= item[2] else deferred).append(item)
                self._events = deferred
            if closing and len(events) > 2:
                priority = [index for index, item in enumerate(events) if item[1]]
                if len(events) - 1 not in priority:
                    priority.append(len(events) - 1)
                priority.extend(index for index in range(len(events)) if index not in priority)
                events = [events[index] for index in priority]
            failed = []
            for text, direct, _ in events:
                try:
                    event_opener = http_opener(None) if direct else opener
                    if event_opener is None:
                        event_opener = http_opener(self._proxy)
                        opener = event_opener
                    self._request(event_opener, self._token, text)
                except Exception:
                    failed.append((text, direct, time.monotonic() + self._interval))
                    LOG.warning("Telegram event failed; delivery will be retried")
            if failed and not closing:
                with self._lock:
                    self._events.extendleft(reversed(failed))
            if closing:
                return
            now = time.monotonic()
            if self._heartbeat_token and now >= next_update and not self._stopped.is_set():
                try:
                    if opener is None:
                        opener = http_opener(self._proxy)
                    self._send(opener)
                except Exception:
                    LOG.warning("Telegram update failed; will retry at the next interval")
                now = time.monotonic()
                next_update += self._interval
                if next_update <= now:
                    next_update = now + self._interval
            with self._lock:
                retry_at = min((item[2] for item in self._events), default=None)
            deadlines = []
            if self._heartbeat_token:
                deadlines.append(next_update)
            if retry_at is not None:
                deadlines.append(retry_at)
            delay = max(0, min(deadlines) - time.monotonic()) if deadlines else None
            if self._stopped.is_set():
                continue
            self._wake.wait(None if delay is None else min(threading.TIMEOUT_MAX, delay))

    def close(self):
        with self._lock:
            if self._close_deadline is None:
                self._close_deadline = time.monotonic() + 2 * _REQUEST_TIMEOUT
            self._stopped.set()
            self._wake.set()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2 * _REQUEST_TIMEOUT + 0.5)

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
