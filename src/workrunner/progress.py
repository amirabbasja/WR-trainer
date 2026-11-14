from contextvars import copy_context
import math
import os
import re
import shutil
import sys
import threading
import time
import unicodedata
_ANSI_ESCAPE = re.compile('\\x1b(?:\\[[0-?]*[ -/]*[@-~]|\\][^\\x07]*?(?:\\x07|\\x1b\\\\)|[@-_])')

def _plain_text(text: str) -> str:
    text = _ANSI_ESCAPE.sub('', str(text))
    return ''.join((' ' if character in '\r\n\t' else character for character in text if character in '\r\n\t' or not unicodedata.category(character).startswith('C')))

def _fit_columns(text: str, columns: int) -> str:
    used = 0
    result = []
    for character in text:
        width = 0 if unicodedata.combining(character) else 2 if unicodedata.east_asian_width(character) in 'WF' else 1
        if used + width > columns:
            break
        result.append(character)
        used += width
    return ''.join(result)


def job_status(job_number, job_title, gpu_percent, time_remaining, speed, *, cnn_epoch=None):
    title = job_title
    if title not in {"job-0", "job-1", "job-2"}:
        raise ValueError("job title must be job-0, job-1, or job-2")
    utilization = "N/A" if gpu_percent is None else f"{gpu_percent}%"
    if time_remaining is None or not math.isfinite(time_remaining):
        remaining = "--:--:--"
    else:
        hours, remainder = divmod(int(max(0, time_remaining)), 3600)
        minutes, seconds = divmod(remainder, 60)
        remaining = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    if speed is None or not math.isfinite(speed):
        value = "0"
    elif isinstance(speed, int):
        value = str(speed)
    else:
        value = f"{speed:.2f}".rstrip("0").rstrip(".")
    if title == "job-2":
        value = f"{value} mining, CNN epoch {0 if cnn_epoch is None else cnn_epoch}"
    return f"{job_number} | {title} | {utilization} | {remaining} | {value}"


def named_status(trainer_name, status):
    return f"{trainer_name} | {status}"


def in_notebook() -> bool:
    # IPython is already loaded in a notebook kernel. Keep it optional for the CLI.
    ipython = sys.modules.get('IPython')
    shell = ipython.get_ipython() if ipython is not None else None
    return getattr(shell, 'kernel', None) is not None


class LiveProgress:

    def __init__(self, stream=None, interval: float=1.0, *, enabled=True, trainer_name=None):
        if not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval < 0:
            raise ValueError('progress interval must be finite and nonnegative')
        self.stream = sys.stderr if stream is None else stream
        self.interval = interval
        self._trainer_name = trainer_name
        try:
            self._tty = bool(self.stream.isatty())
        except (AttributeError, OSError):
            self._tty = False
        self._notebook = enabled and stream is None and in_notebook()
        if self._notebook:
            self._tty = False
        self._refresh_stop = threading.Event()
        self._refresh_thread = None
        self._lock = threading.RLock()
        self._latest_text = None
        self._rendered_text = None
        self._last_update = -math.inf
        self._active = False
        self._closed = not enabled

    def start_refresh(self, snapshot):
        """Keep the notebook status live while training is blocked in a long call."""
        if not self._notebook or self._closed or self._refresh_thread is not None:
            return

        self.update(snapshot())

        def refresh():
            while not self._refresh_stop.wait(self.interval or 1.0):
                self.update(snapshot())

        # The display publisher uses the current cell's execution context.
        context = copy_context()
        thread = threading.Thread(target=context.run, args=(refresh,), name='notebook-progress', daemon=True)
        thread.start()
        self._refresh_thread = thread

    def _columns(self) -> int:
        try:
            columns = os.get_terminal_size(self.stream.fileno()).columns
        except (AttributeError, OSError, ValueError):
            columns = shutil.get_terminal_size(fallback=(120, 24)).columns
        return max(0, columns - 1)

    def _render(self, text: str) -> None:
        if self._notebook:
            from IPython.display import clear_output, display

            # Some notebook frontends do not apply update_display_data messages.
            if self._active:
                clear_output(wait=True)
            display({'text/plain': text}, raw=True)
        else:
            self.stream.write('\r\x1b[2K' + _fit_columns(text, self._columns()) if self._tty else text + '\n')
            self.stream.flush()
        self._rendered_text = text
        self._active = True

    def update(self, text: str, force: bool=False) -> None:
        with self._lock:
            if self._closed:
                return
            if self._trainer_name is not None:
                text = named_status(self._trainer_name, text)
            self._latest_text = _plain_text(text)
            if self._latest_text == self._rendered_text and self._active:
                return
            now = time.monotonic()
            if not force and self._active and (now - self._last_update < self.interval):
                return
            self._render(self._latest_text)
            self._last_update = now

    def clear(self) -> None:
        with self._lock:
            if self._closed or not self._active or not self._tty:
                return
            self.stream.write('\r\x1b[2K')
            self.stream.flush()
            self._active = False
            self._rendered_text = None

    def close(self) -> None:
        self._refresh_stop.set()
        if self._refresh_thread is not None:
            self._refresh_thread.join()
        with self._lock:
            if self._closed:
                return
            if self._latest_text is not None:
                if not self._active or self._rendered_text != self._latest_text:
                    self._render(self._latest_text)
                if self._tty:
                    self.stream.write('\n')
                    self.stream.flush()
            self._active = False
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
