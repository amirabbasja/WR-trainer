from contextlib import contextmanager
import logging
import os
import sys
import threading
import warnings


_output_lock = threading.RLock()
_artifacts_enabled = True


def artifacts_enabled():
    return _artifacts_enabled


@contextmanager
def training_output(generate_artifacts):
    """Scope the process-wide output and cache settings to a training run."""
    if generate_artifacts:
        yield
        return

    global _artifacts_enabled
    with _output_lock, warnings.catch_warnings():
        previous_artifacts = _artifacts_enabled
        previous_bytecode = sys.dont_write_bytecode
        previous_logging = logging.root.manager.disable
        previous_cuda_cache = os.environ.get('CUDA_CACHE_DISABLE')
        try:
            _artifacts_enabled = False
            sys.dont_write_bytecode = True
            logging.disable(sys.maxsize)
            warnings.simplefilter('ignore')
            os.environ['CUDA_CACHE_DISABLE'] = '1'
            yield
        finally:
            _artifacts_enabled = previous_artifacts
            sys.dont_write_bytecode = previous_bytecode
            logging.disable(previous_logging)
            if previous_cuda_cache is None:
                os.environ.pop('CUDA_CACHE_DISABLE', None)
            else:
                os.environ['CUDA_CACHE_DISABLE'] = previous_cuda_cache
