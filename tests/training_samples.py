"""Independent recorded protocol bytes, named for their training-facing purpose."""

import json
from pathlib import Path

_SAMPLES = json.loads((Path(__file__).with_name('fixtures') / 'training_protocol_samples.json').read_text())


def sample_bytes(name):
    return bytes.fromhex(_SAMPLES[name])


def sample_text(name):
    return sample_bytes(name).decode('utf-8')
