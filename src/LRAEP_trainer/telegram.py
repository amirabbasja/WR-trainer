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
        self._chat_id = settings.get("tgChatID", "")
        self._interval = settings.get("update_interval_seconds", 300)
        self._snapshot = snapshot
        self._proxy = current_proxy()
        self._stopped = threading.Event()
        self._thread = None

    def start(self):
        if self._enabled and self._thread is None and not self._stopped.is_set():
            self._thread = threading.Thread(
                target=self._run, name="LRAEP-telegram-updates", daemon=True,
            )
            self._thread.start()
        return self

    def _send(self, opener):
        text = f"{self._name}\n{self._snapshot()}"[:4000]
        request = Request(
            f"https://api.telegram.org/bot{self._token}/sendMessage",
            data=json.dumps({"chat_id": self._chat_id, "text": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with opener.open(request, timeout=_REQUEST_TIMEOUT) as response:
            result = json.loads(response.read(65536))
        if not isinstance(result, dict) or result.get("ok") is not True:
            raise ValueError("Telegram rejected the update")

    def _run(self):
        try:
            opener = http_opener(self._proxy)
        except Exception:
            LOG.warning("Telegram updates could not initialize; training continues")
            return
        next_update = time.monotonic() + self._interval
        while not self._stopped.wait(min(threading.TIMEOUT_MAX,
                                         max(0, next_update - time.monotonic()))):
            if time.monotonic() < next_update:
                continue
            try:
                self._send(opener)
            except Exception:
                LOG.warning("Telegram update failed; will retry at the next interval")
            now = time.monotonic()
            next_update += self._interval
            if next_update <= now:
                next_update = now + self._interval

    def close(self):
        self._stopped.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=_REQUEST_TIMEOUT + 0.5)

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
