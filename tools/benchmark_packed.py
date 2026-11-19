"""CUDA implementation for the benchmark-only packed operand and transcript layout."""
import ctypes
from pathlib import Path
import subprocess
import tempfile

import torch


def packed_kernels(device):
    from workrunner import native_cuda

    sources = Path(__file__).with_name('cuda')
    major, minor = torch.cuda.get_device_capability(device)
    with tempfile.TemporaryDirectory(prefix='pearl-benchmark-') as build:
        output = Path(build) / 'packed.so'
        subprocess.run(['nvcc', '-O3', '-std=c++17', '--shared', '-Xcompiler=-fPIC',
                        f'-arch=sm_{major}{minor}', '-I', str(native_cuda.SOURCE),
                        str(sources / 'benchmark_preparation.cu'), str(sources / 'benchmark_transcripts.cu'),
                        '-o', str(output)], check=True)
        lib = ctypes.CDLL(str(output))
    pointer = ctypes.c_void_p
    lib.benchmark_noise.argtypes = [ctypes.c_int] * 4 + [pointer] * 5
    lib.benchmark_noise.restype = ctypes.c_int
    lib.benchmark_transcripts.argtypes = [pointer, pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                          pointer, pointer]
    lib.benchmark_transcripts.restype = ctypes.c_int
    runtime = native_cuda.load_native(device)

    def prepare(rows, k, value, key, device, *, right=False):
        if not (128 <= rows <= 4096 and rows % 128 == 0 and 128 <= k <= 65536 and k % 128 == 0):
            raise ValueError('packed operands require complete 128-row and rank-128 blocks')
        if not -128 <= value <= 127:
            raise ValueError('expected a signed byte for the constant operand')
        with torch.cuda.device(device):
            uniform = torch.empty((rows, 128), device=device, dtype=torch.int8)
            pairs = torch.empty((k, 2), device=device, dtype=torch.int32)
            packed = torch.empty((rows // 128, k // 4, 2, 16, 4, 4), device=device, dtype=torch.int8)
            stream = torch.cuda.current_stream(device)
            with native_cuda._temporary_words(device, key) as (key_gpu,):
                native_cuda._check(runtime, lib.benchmark_noise(
                    rows, k, value, int(right), key_gpu.data_ptr(), uniform.data_ptr(),
                    pairs.data_ptr(), packed.data_ptr(), stream.cuda_stream))
        return packed

    def transcripts(a, bt):
        for operand in (a, bt):
            if (operand.dtype != torch.int8 or operand.device != device or
                    not operand.is_contiguous() or operand.ndim != 6 or
                    tuple(operand.shape[2:]) != (2, 16, 4, 4)):
                raise ValueError('expected a packed int8 operand with 128-row blocks')
        if a.shape[1] != bt.shape[1]:
            raise ValueError('packed operands must have the same common dimension')
        m, n, k = a.shape[0] * 128, bt.shape[0] * 128, a.shape[1] * 4
        with torch.cuda.device(device):
            stream = torch.cuda.current_stream(device)
            a.record_stream(stream)
            bt.record_stream(stream)
            output = torch.empty((m // 8, n // 8, 16), dtype=torch.int32, device=device)
            native_cuda._check(runtime, lib.benchmark_transcripts(
                a.data_ptr(), bt.data_ptr(), m, n, k, output.data_ptr(), stream.cuda_stream))
        return output

    return prepare, transcripts
