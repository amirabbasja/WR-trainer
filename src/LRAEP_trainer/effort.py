from contextlib import contextmanager
import logging
import math
import random
import threading
import time

def normalize_job_sequence(value, *, field='job_sequence'):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f'{field} must be a list of [kind, seconds, percentage, optional tolerance] entries')
    if not value:
        raise ValueError(f'{field} must contain at least one interval')
    entries = []
    total = 0.0
    for entry in value:
        if not isinstance(entry, (list, tuple)) or len(entry) not in (3, 4):
            raise ValueError(f'{field} intervals must contain kind, seconds, percentage and optional tolerance')
        kind, seconds, percentage = entry[:3]
        tolerance = entry[3] if len(entry) == 4 else 0
        if type(tolerance) is not int or not 0 <= tolerance <= 100:
            raise ValueError(f'{field} tolerance must be an integer between 0 and 100')
        if type(kind) is not str or kind not in ('main', 'side-job'):
            raise ValueError(f'{field} kinds must be "main" or "side-job"')
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
        entries.append(tuple(entry))
    return entries

class JobSequence:

    def __init__(self, entries, *, started=None):
        self.entries = normalize_job_sequence(entries)
        self.started = time.monotonic() if started is None else started
        self._total = sum((entry[1] for entry in self.entries))
        self._sample_lock = threading.Lock()
        self._sample_key = None
        self._sample_percentage = None
        self.ceiling = None
        self._exit_offset = None
        self._exit_kind = None
        elapsed = 0
        for entry in self.entries:
            kind, seconds, percentage = entry[:3]
            if percentage == -1:
                self._exit_offset = elapsed
                self._exit_kind = kind
                break
            elapsed += seconds

    @property
    def exit_deadline(self):
        return None if self._exit_offset is None else self.started + self._exit_offset

    def _current(self, now):
        elapsed = max(0, now - self.started)
        if self.exit_deadline is not None and now >= self.exit_deadline:
            return (self._exit_kind, -1, math.inf)
        cycle, _ = divmod(elapsed, self._total)
        boundary = 0
        for entry in self.entries:
            kind, seconds, percentage = entry[:3]
            boundary += seconds
            until = self.started + cycle * self._total + boundary
            if now < until:
                return (kind, self._target(entry, until, now), until)
        until = self.started + (cycle + 1) * self._total + self.entries[0][1]
        return (self.entries[0][0], self._target(self.entries[0], until, now), max(until, math.nextafter(now, math.inf)))

    def _target(self, entry, until, now):
        percentage = entry[2]
        tolerance = entry[3] if len(entry) == 4 else 0
        self.ceiling = min(100, percentage + tolerance) if percentage > 0 else percentage
        if percentage <= 0 or tolerance == 0:
            return percentage
        interval_start = until - entry[1]
        key = (until, int(max(0, now - interval_start) // 5))
        with self._sample_lock:
            if key != self._sample_key:
                self._sample_percentage = random.randint(max(1, percentage - tolerance), min(100, percentage + tolerance))
                self._sample_key = key
            return self._sample_percentage

    def current(self):
        _, percentage, boundary = self.current_job()
        return (percentage, boundary)

    def current_job(self):
        return self._current(time.monotonic())

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
        if kind != 'side-job' or percentage <= 0 or stopped.is_set():
            return False
        until = boundary if deadline is None else min(boundary, deadline)
        if time.monotonic() >= until:
            return False
        if pacer.max_utilization != percentage:
            pacer.set_utilization(percentage)
        if type(schedule.ceiling) is int:
            pacer.set_ceiling(schedule.ceiling)
        if self._previous_interval != (percentage, boundary):
            if self.before_log is not None:
                self.before_log()
            logging.getLogger('LRAEP_trainer').info('Side-job effort GPU duty-cycle target: %d%%', percentage)
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
            return stopped.is_set() or time.monotonic() >= until or schedule.current_job()[0] != 'side-job' or (schedule.current()[0] <= 0)
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
