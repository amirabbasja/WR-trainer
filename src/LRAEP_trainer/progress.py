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

class LiveProgress:

    def __init__(self, stream=None, interval: float=1.0):
        if not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval < 0:
            raise ValueError('progress interval must be finite and nonnegative')
        self.stream = sys.stderr if stream is None else stream
        self.interval = interval
        try:
            self._tty = bool(self.stream.isatty())
        except (AttributeError, OSError):
            self._tty = False
        self._lock = threading.RLock()
        self._latest_text = None
        self._rendered_text = None
        self._last_update = -math.inf
        self._active = False
        self._closed = False

    def _columns(self) -> int:
        try:
            columns = os.get_terminal_size(self.stream.fileno()).columns
        except (AttributeError, OSError, ValueError):
            columns = shutil.get_terminal_size(fallback=(120, 24)).columns
        return max(0, columns - 1)

    def _render(self, text: str) -> None:
        self.stream.write('\r\x1b[2K' + _fit_columns(text, self._columns()) if self._tty else text + '\n')
        self.stream.flush()
        self._rendered_text = text
        self._active = True

    def update(self, text: str, force: bool=False) -> None:
        with self._lock:
            if self._closed:
                return
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
        with self._lock:
            if self._closed:
                return
            if self._latest_text is not None:
                if self._tty:
                    if not self._active or self._rendered_text != self._latest_text:
                        self._render(self._latest_text)
                    self.stream.write('\n')
                elif self._rendered_text != self._latest_text:
                    self.stream.write(self._latest_text + '\n')
                self.stream.flush()
            self._active = False
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
