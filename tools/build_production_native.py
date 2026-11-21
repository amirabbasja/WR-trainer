#!/usr/bin/env python3
"""Build source-free production compute libraries from this WR trainer checkout."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / 'native' / 'production'
SOURCE = NATIVE / 'src' / 'workrunner'
CUDA_FILES = ('transcripts.cu', 'candidate_filter.cu', 'preparation.cu', 'tree.cu', 'runtime.cu')


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--production', type=Path, default=ROOT.parent / 'Production-LREAP')
    parser.add_argument('--architectures', nargs='+', help='SM numbers; defaults to every target supported by nvcc')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    machine = {'amd64': 'x86_64', 'arm64': 'aarch64'}.get(platform.machine().lower(), platform.machine().lower())
    if platform.system() != 'Linux' or machine not in ('x86_64', 'aarch64'):
        parser.error('build on the target Linux x86_64 or aarch64 host')
    module = args.production / 'src' / 'workrunner'
    if not (module / '__init__.py').is_file():
        parser.error('production checkout not found: ' + str(args.production))
    destination = module / 'bin' / ('linux-' + machine)
    sources = [SOURCE / 'stream_digest_native.cpp'] + [SOURCE / 'cuda' / name for name in (*CUDA_FILES, 'digest_core.cuh')]
    sources += [NATIVE / 'tools' / 'cuda' / name for name in ('benchmark_preparation.cu', 'benchmark_transcripts.cu')]
    fingerprints = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    manifest_path = destination / 'compute_manifest.json'
    if args.check:
        manifest = json.loads(manifest_path.read_text())
        if manifest['sources'] != fingerprints:
            raise ValueError('production compute binaries have stale source fingerprints')
        for name, expected in manifest['artifacts'].items():
            if hashlib.sha256((destination / name).read_bytes()).hexdigest() != expected:
                raise ValueError('production artifact hash mismatch: ' + name)
        if args.architectures and sorted(set(args.architectures)) != manifest['cuda_architectures']:
            raise ValueError('production CUDA targets differ from requested architectures')
        print('Production compute sources and compiled artifacts agree.')
        return
    compiler, nvcc = shutil.which(os.environ.get('CXX', 'c++')), shutil.which('nvcc')
    if not compiler or not nvcc:
        parser.error('C++ compiler and nvcc are required in the trainer checkout')
    available = re.findall(r'^sm_(\d+)$', subprocess.check_output([nvcc, '--list-gpu-code'], text=True), re.M)
    architectures = sorted(set(args.architectures or [sm for sm in available if int(sm) >= 61]))
    if not architectures or any(sm not in available or int(sm) < 61 for sm in architectures):
        parser.error('unsupported CUDA architecture selection')
    flags = [f'-gencode=arch=compute_{sm},code=sm_{sm}' for sm in architectures]
    artifacts = {}
    with tempfile.TemporaryDirectory(prefix='production-native-') as temporary:
        staging = Path(temporary)
        commands = {
            'libstream_digest.so': [compiler, '-O3', '-std=c++17', '-shared', '-fPIC', '-s', str(sources[0])],
            'libnative_cuda.so': [nvcc, '-O3', '-std=c++17', '--shared', '-Xcompiler=-fPIC', *flags,
                                  '-I', str(SOURCE / 'cuda'), *[str(SOURCE / 'cuda' / name) for name in CUDA_FILES]],
            'libbenchmark_packed.so': [nvcc, '-O3', '-std=c++17', '--shared', '-Xcompiler=-fPIC', *flags,
                                       '-I', str(SOURCE / 'cuda'), *[str(path) for path in sources[-2:]]],
        }
        for name, command in commands.items():
            output = staging / name
            print('Building ' + name + '...', flush=True)
            subprocess.run([*command, '-o', str(output)], check=True)
            strip = shutil.which('strip')
            if strip:
                subprocess.run([strip, '--strip-unneeded', str(output)], check=True)
            library = ctypes.CDLL(str(output))
            symbol = {'libstream_digest.so': 'cnn3_init', 'libnative_cuda.so': 'wr_transcripts_i8',
                      'libbenchmark_packed.so': 'benchmark_noise'}[name]
            getattr(library, symbol)
            artifacts[name] = output.read_bytes()
        manifest = {
            'platform': 'linux-' + machine,
            'cuda_architectures': architectures,
            'cuda_compiler': subprocess.check_output([nvcc, '--version'], text=True).strip(),
            'sources': fingerprints,
            'artifacts': {name: hashlib.sha256(data).hexdigest() for name, data in artifacts.items()},
        }
        for name, data in artifacts.items():
            atomic_write(destination / name, data)
        atomic_write(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode())
    print('Installed production compute binaries for ' + ', '.join('sm_' + sm for sm in architectures) + '.')


if __name__ == '__main__':
    main()
