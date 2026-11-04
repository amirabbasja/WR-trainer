from contextlib import contextmanager
import logging
import math
import threading
import time

def normalize_job_sequence(value, *, field="job_sequence"):
    """Copy job dictionaries after validating titles, seconds, and utilization."""
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array of job dictionaries")
    if not value:
        raise ValueError(f"{field} must contain at least one interval")
    entries = []
    total = 0.0
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != {"title", "time", "max_gpu_utilization"}:
            raise ValueError(f"{field} entries must contain exactly title, time, and max_gpu_utilization")
        title = entry["title"]
        seconds = entry["time"]
        percentage = entry["max_gpu_utilization"]
        if type(title) is not str or title not in ("job-0", "job-1"):
            raise ValueError(f'{field} title must be "job-0" or "job-1"')
        if type(percentage) is not int or not -1 <= percentage <= 100:
            raise ValueError(f"{field} max_gpu_utilization must be an integer between -1 and 100")
        try:
            finite = type(seconds) in (int, float) and math.isfinite(seconds)
        except OverflowError:
            finite = False
        if not finite or seconds < 0 or (seconds == 0 and percentage != -1):
            raise ValueError(f"{field} time must be positive and finite (zero is allowed for exit)")
        total += seconds
        if not math.isfinite(total):
            raise ValueError(f"{field} total duration must be finite")
        entries.append(dict(entry))
    return entries


class JobSequence:
    """Repeat job-0 and job-1 intervals on one monotonic timeline."""

    def __init__(self, entries, *, started=None):
        self.entries = normalize_job_sequence(entries)
        self.started = time.monotonic() if started is None else started
        self._total = sum(entry["time"] for entry in self.entries)
        self._exit_offset = None
        self._exit_kind = None
        elapsed = 0
        for entry in self.entries:
            if entry["max_gpu_utilization"] == -1:
                self._exit_offset = elapsed
                self._exit_kind = entry["title"]
                break
            elapsed += entry["time"]

    @property
    def exit_deadline(self):
        """Absolute start of the first exit entry, or None for a repeating run."""
        return None if self._exit_offset is None else self.started + self._exit_offset

    def _current(self, now):
        elapsed = max(0, now - self.started)
        if self.exit_deadline is not None and now >= self.exit_deadline:
            return self._exit_kind, -1, math.inf
        cycle, _ = divmod(elapsed, self._total)
        boundary = 0
        for entry in self.entries:
            boundary += entry["time"]
            until = self.started + cycle * self._total + boundary
            if now < until:
                return entry["title"], entry["max_gpu_utilization"], until
        entry = self.entries[0]
        until = self.started + (cycle + 1) * self._total + entry["time"]
        return entry["title"], entry["max_gpu_utilization"], max(until, math.nextafter(now, math.inf))

    def current(self):
        """Return the configured utilization and the interval's absolute end time."""
        _, percentage, boundary = self.current_job()
        return percentage, boundary

    def current_job(self):
        """Return the active title, utilization, and absolute end time."""
        return self._current(time.monotonic())

    def position(self, now=None):
        now = time.monotonic() if now is None else now
        title, percentage, until = self._current(now)
        if percentage == -1:
            index = next(i for i, entry in enumerate(self.entries) if entry["max_gpu_utilization"] == -1)
            return index + 1, title, percentage, until
        elapsed = max(0, now - self.started)
        cycle = int(elapsed // self._total)
        boundary = self.started + cycle * self._total
        for index, entry in enumerate(self.entries):
            boundary += entry["time"]
            if now < boundary:
                return cycle * len(self.entries) + index + 1, title, percentage, until
        return (cycle + 1) * len(self.entries) + 1, title, percentage, until

    def ended(self):
        return self.current()[0] == -1

    def wait(self, stopped, *, deadline=None):
        while not stopped.is_set():
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                return False
            _, percentage, boundary = self._current(now)
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

class ScheduledSideJob:

    def __init__(self, configuration, *, before_log=None, on_progress=None):
        self.configuration = configuration
        self.before_log = before_log
        self.on_progress = on_progress
        self.job = None
        self.exhausted = False
        self._previous_interval = None

    def run_if_active(self, schedule, stopped, pacer, *, deadline=None):
        kind, percentage, boundary = schedule.current_job()
        if kind != 'job-1' or percentage <= 0 or stopped.is_set():
            return False
        until = boundary if deadline is None else min(boundary, deadline)
        if time.monotonic() >= until:
            return False
        if pacer.max_utilization != percentage:
            pacer.set_utilization(percentage)
        if self._previous_interval != (percentage, boundary):
            if self.before_log is not None:
                self.before_log()
            logging.getLogger('workrunner').debug('job-1 effort GPU duty-cycle target: %d%%', percentage)
            self._previous_interval = (percentage, boundary)
        if self.exhausted:
            if self.on_progress is not None:
                self.on_progress(self.job)
            stopped.wait(min(threading.TIMEOUT_MAX, max(0, until - time.monotonic())))
            return True
        if self.job is None:
            from .side_job import SideJob
            options = {'before_log': self.before_log} if self.before_log is not None else {}
            self.job = SideJob(self.configuration, **options)

        def cancelled():
            return stopped.is_set() or time.monotonic() >= until or schedule.current_job()[0] != 'job-1' or (schedule.current()[0] <= 0)
        if self.on_progress is not None:
            self.on_progress(self.job)
        pacer.begin()
        try:
            runnable = self.job.step(cancelled=cancelled, pacer=pacer, stopped=stopped, deadline=until)
        except BaseException:
            pacer.finish(cancelled=True)
            raise
        pacer.finish(cancelled=cancelled())
        self.exhausted = not runnable
        if self.on_progress is not None:
            self.on_progress(self.job)
        if not cancelled():
            if self.exhausted:
                stopped.wait(min(threading.TIMEOUT_MAX, max(0, until - time.monotonic())))
            else:
                pacer.wait(stopped, deadline=until)
        return True

    def close(self):
        if self.job is not None:
            self.job.close()
            self.job = None
