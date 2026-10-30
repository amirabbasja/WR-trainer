import os
import re
import subprocess
import threading
import time
_SAMPLE_INTERVAL = 0.2
_QUERY_TIMEOUT = 0.75
_UUID = re.compile('(?:GPU-|MIG-)?[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}')

def _uuid_identifier(value):
    candidate = str(value)
    if not _UUID.fullmatch(candidate):
        return None
    return candidate if candidate.startswith(('GPU-', 'MIG-')) else f'GPU-{candidate}'

def _device_identifier(device):
    try:
        import torch
        selected = torch.device(device)
        if selected.type != 'cuda':
            return None
        index = selected.index if selected.index is not None else torch.cuda.current_device()
    except (ImportError, RuntimeError, ValueError, TypeError, AssertionError):
        return None
    try:
        properties = torch.cuda.get_device_properties(index)
        identifier = _uuid_identifier(getattr(properties, 'uuid', None))
        if identifier is not None:
            return identifier
        pci = tuple((getattr(properties, name, None) for name in ('pci_domain_id', 'pci_bus_id', 'pci_device_id')))
        if all((type(value) is int and value >= 0 for value in pci)):
            domain, bus, slot = pci
            return f'{domain:08x}:{bus:02x}:{slot:02x}.0'
    except (RuntimeError, ValueError, TypeError, AssertionError):
        pass
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if 0 <= index < len(visible):
        return _uuid_identifier(visible[index].strip())
    return None

def _query_gpu_utilization(identifier):
    try:
        result = subprocess.run(['nvidia-smi', f'--id={identifier}', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=False, timeout=_QUERY_TIMEOUT)
        rows = result.stdout.strip().splitlines()
        if result.returncode != 0 or len(rows) != 1:
            return None
        percent = int(rows[0].strip())
        return percent if 0 <= percent <= 100 else None
    except (OSError, subprocess.SubprocessError, ValueError, UnicodeError):
        return None

class GpuUtilization:

    def __init__(self, device: str):
        self.device = device
        self._value = None
        self._sampled_at = None
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._thread = None

    def start(self):
        with self._lock:
            if self._thread is None and (not self._stopped.is_set()):
                self._thread = threading.Thread(target=self._run, name='LRAEP-gpu-telemetry', daemon=True)
                self._thread.start()
        return self

    def _run(self):
        identifier = _device_identifier(self.device)
        if identifier is None:
            return
        while not self._stopped.is_set():
            value = _query_gpu_utilization(identifier)
            with self._lock:
                self._value = value
                self._sampled_at = time.monotonic() if value is not None else None
            if self._stopped.wait(_SAMPLE_INTERVAL):
                return

    def read(self) -> int | None:
        with self._lock:
            return self._value

    def read_sample(self):
        with self._lock:
            return self._value, self._sampled_at

    def close(self):
        self._stopped.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=_QUERY_TIMEOUT + 0.25)

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
