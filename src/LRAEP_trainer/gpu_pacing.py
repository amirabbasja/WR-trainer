from contextlib import contextmanager
import math
import threading
import time
import torch
from torch.utils._python_dispatch import TorchDispatchMode


def _cuda_value(value):
    if isinstance(value, torch.Tensor):
        return value.is_cuda
    if isinstance(value, torch.device):
        return value.type == 'cuda'
    if isinstance(value, dict):
        return any(_cuda_value(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return any(_cuda_value(item) for item in value)
    return False


class _OperationPacing(TorchDispatchMode):
    def __init__(self, pacer, stopped, deadline):
        self.pacer = pacer
        self.stopped = stopped
        self.deadline = deadline

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        cuda = _cuda_value(args) or _cuda_value(kwargs)
        if cuda and self.pacer._operation_seconds == 0:
            self.pacer._wait_for_capacity(self.stopped, self.deadline)
        started = time.monotonic()
        result = func(*args, **kwargs)
        if cuda or _cuda_value(result):
            torch.cuda.synchronize(self.pacer.device)
            self.pacer._operation_seconds += max(0, time.monotonic() - started)
            if self.pacer._operation_seconds >= self.pacer._quantum:
                work = self.pacer._operation_seconds
                self.pacer._operation_seconds = 0.0
                rest = work * (100 - self.pacer.max_utilization) / self.pacer.max_utilization
                self.pacer._rest(self.stopped, rest, self.deadline)
                self.pacer._wait_for_capacity(self.stopped, self.deadline)
        return result


class GpuPacer:
    _quantum = 0.01
    _feedback_interval = 0.05
    _max_sample_age = 2.0

    def __init__(self, device, max_utilization):
        self.device = device
        self._started = None
        self._deadline = None
        self._reader = None
        self._ceiling = None
        self._operation_paced = False
        self._operation_seconds = 0.0
        self.max_utilization = None
        self.set_utilization(max_utilization)

    def set_utilization(self, percentage):
        if type(percentage) is not int or not 1 <= percentage <= 100:
            raise ValueError('GPU utilization must be an integer between 1 and 100')
        if percentage != self.max_utilization:
            self.max_utilization = percentage
            self._started = None
            self._deadline = None
            self._operation_paced = False
            self._operation_seconds = 0.0

    def set_monitor(self, reader):
        self._reader = reader

    def set_ceiling(self, percentage):
        if type(percentage) is not int or not 1 <= percentage <= 100:
            raise ValueError('GPU ceiling must be an integer between 1 and 100')
        self._ceiling = percentage

    def begin(self):
        self._operation_paced = False
        self._operation_seconds = 0.0
        if self.max_utilization < 100:
            self._started = time.monotonic()

    @contextmanager
    def operations(self, stopped=None, *, deadline=None):
        if self.max_utilization == 100:
            yield
            return
        self._operation_paced = True
        stopped = threading.Event() if stopped is None else stopped
        with _OperationPacing(self, stopped, deadline):
            yield

    def _rest(self, stopped, seconds, deadline):
        now = time.monotonic()
        until = now + seconds
        if deadline is not None:
            until = min(until, deadline)
        stopped.wait(min(threading.TIMEOUT_MAX, max(0, until - now)))

    def _wait_for_capacity(self, stopped, deadline):
        if self._reader is None or self.max_utilization == 100:
            return
        while not stopped.is_set():
            try:
                value, sampled_at = self._reader()
            except (TypeError, ValueError, OSError, RuntimeError):
                return
            if (type(value) is not int or not 0 <= value <= 100
                    or type(sampled_at) not in (int, float) or not math.isfinite(sampled_at)):
                return
            now = time.monotonic()
            age = now - sampled_at
            ceiling = self.max_utilization if self._ceiling is None else self._ceiling
            if age < 0 or age > self._max_sample_age or value <= ceiling:
                return
            if deadline is not None and now >= deadline:
                return
            self._rest(stopped, self._feedback_interval, deadline)

    def finish(self, *, cancelled=False):
        if self._started is None:
            if cancelled:
                torch.cuda.synchronize(self.device)
            return
        torch.cuda.synchronize(self.device)
        now = time.monotonic()
        work_seconds = self._operation_seconds if self._operation_paced else now - self._started
        self._deadline = now + work_seconds * (100 - self.max_utilization) / self.max_utilization
        self._started = None

    def wait(self, stopped, *, deadline=None):
        if self._deadline is not None:
            until = self._deadline if deadline is None else min(self._deadline, deadline)
            self._deadline = None
            stopped.wait(min(threading.TIMEOUT_MAX, max(0, until - time.monotonic())))
        self._wait_for_capacity(stopped, deadline)
