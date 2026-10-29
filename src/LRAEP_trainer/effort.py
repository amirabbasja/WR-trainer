from contextlib import contextmanager
import math
import threading
import time

def normalize_effort(value, *, field='main_effort', allow_empty=False):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f'{field} must be a list of (seconds, percentage) pairs')
    if not value and (not allow_empty):
        raise ValueError(f'{field} must contain at least one interval')
    entries = []
    total = 0.0
    for pair in value:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f'{field} intervals must contain seconds and percentage')
        seconds, percentage = pair
        if type(percentage) is not int or not -1 <= percentage <= 100:
            raise ValueError(f'{field} percentages must be integers between -1 and 100')
        try:
            finite = type(seconds) in (int, float) and math.isfinite(seconds)
        except OverflowError:
            finite = False
        if not finite or seconds < 0 or (seconds == 0 and percentage != -1):
            raise ValueError(f'{field} seconds must be positive and finite (zero is allowed for exit)')
        total += seconds
        if not math.isfinite(total):
            raise ValueError(f'{field} total duration must be finite')
        entries.append((seconds, percentage))
    return entries

class EffortSchedule:

    def __init__(self, entries, *, started=None, allow_empty=False):
        self.entries = normalize_effort(entries, allow_empty=allow_empty)
        self.started = time.monotonic() if started is None else started
        self._total = sum((seconds for seconds, _ in self.entries))
        self._exit_offset = None
        elapsed = 0
        for seconds, percentage in self.entries:
            if percentage == -1:
                self._exit_offset = elapsed
                break
            elapsed += seconds

    @property
    def exit_deadline(self):
        return None if self._exit_offset is None else self.started + self._exit_offset

    def _current(self, now):
        elapsed = max(0, now - self.started)
        if self.exit_deadline is not None and now >= self.exit_deadline:
            return (-1, math.inf)
        if not self.entries:
            return (0, math.inf)
        cycle, _ = divmod(elapsed, self._total)
        boundary = 0
        for seconds, percentage in self.entries:
            boundary += seconds
            until = self.started + cycle * self._total + boundary
            if now < until:
                return (percentage, until)
        until = self.started + (cycle + 1) * self._total + self.entries[0][0]
        return (self.entries[0][1], max(until, math.nextafter(now, math.inf)))

    def current(self):
        return self._current(time.monotonic())

    def ended(self):
        return self.current()[0] == -1

    def wait(self, stopped, *, deadline=None):
        while not stopped.is_set():
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                return False
            percentage, boundary = self._current(now)
            if percentage == -1:
                stopped.set()
                return False
            if percentage > 0:
                return True
            until = boundary if deadline is None else min(boundary, deadline)
            stopped.wait(None if math.isinf(until) else min(threading.TIMEOUT_MAX, max(0, until - now)))
        return False

    @contextmanager
    def stopping(self, stopped):
        timer = None
        if self._exit_offset is not None and (not stopped.is_set()):
            delay = self.exit_deadline - time.monotonic()
            if delay <= 0:
                stopped.set()
            else:
                callback = stopped.set
                if delay > threading.TIMEOUT_MAX:

                    def wait_until_exit():
                        while not timer.finished.is_set():
                            remaining = self.exit_deadline - time.monotonic()
                            if remaining <= 0:
                                stopped.set()
                                return
                            timer.finished.wait(min(remaining, threading.TIMEOUT_MAX))
                    callback = wait_until_exit
                timer = threading.Timer(min(delay, threading.TIMEOUT_MAX), callback)
                timer.daemon = True
                timer.start()
        try:
            yield
        finally:
            if timer is not None:
                timer.cancel()
                if threading.current_thread() is not timer:
                    timer.join()
