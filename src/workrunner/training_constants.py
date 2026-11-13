"""Neutral training names backed by the native compatibility vocabulary."""

from .fixed_constants import FixedAlias, constant_ascii_matches, constant_bytes, constant_text
from .native_text_codec import word_buffer


_TEXT = {
    'ACCOUNT': 'account',
    'LOCAL_SEED_FIELD': 'local_seed',
    'NOISE_LABEL': 'noise_seed_domain',
    'LEGACY_MODE': 'legacy_protocol',
    'SEND_RESULT': FixedAlias('training.message.submit_batch_result'),
    'JOB_UPDATE': FixedAlias('training.message.batch_update'),
    'DEVICE_REPORT': FixedAlias('training.message.accelerator_metrics'),
    'LOGIN': FixedAlias('training.message.open_session'),
    'DEVICE_ID': FixedAlias('training.field.training_worker'),
    'SERVER_SEED': 'server_seed',
    'DEVICE_SEED': 'device_seed',
    'SEED': FixedAlias('training.field.batch_seed'),
    'RATE': 'training_rate',
    'EVIDENCE': FixedAlias('training.field.dense_attestation'),
    'TRAINING_THRESHOLD': FixedAlias('training.field.acceptance_bound'),
    'FORMAT_SETTING': FixedAlias('training.field.attestation_version'),
    'FORMAT_FIELD': FixedAlias('training.field.attestation_revision'),
    'JOB_ID': FixedAlias('training.field.batch_identifier'),
    'DEFAULT_DEVICE': 'device_default',
    'CLIENT_AGENT': FixedAlias('training.metadata.client_agent'),
    'DEFAULT_TITLE': FixedAlias('training.metadata.trainer_title'),
    'IDENTITY_PREFIX': 'identity_prefix',
}
_RATE_UNITS = ('units/s', 'kunits/s', 'Munits/s', 'Gunits/s', 'Tunits/s', 'Punits/s', 'Eunits/s')
LOGGER_NAME = __package__
WR_CAPTURED_SHAPE = (4096, 262144, 8192, 8, 16)
__all__ = [*_TEXT, 'RATE_UNITS', 'LOGGER_NAME', 'local_seed_field',
           's6_noise_seed_prefix', 's7_legacy_alias', 'text_matches',
           'has_identity_mode_prefix']


def has_identity_mode_prefix(value):
    """Check the peer's unsupported identity mode without retaining its text."""
    if not isinstance(value, str):
        return False
    with word_buffer('identity_mode_prefix') as prefix:
        return len(value) >= len(prefix) and all(
            ord(value[index]) == byte for index, byte in enumerate(prefix)
        )


def __getattr__(name):
    if name == 'RATE_UNITS':
        return _RATE_UNITS
    try:
        original = _TEXT[name]
    except KeyError:
        raise AttributeError(name) from None
    return constant_text(original)


def text_matches(name, value):
    """Compare protocol text using a temporary buffer that is cleared afterward."""
    return constant_ascii_matches(_TEXT[name], value)


def local_seed_field():
    return constant_text(_TEXT['LOCAL_SEED_FIELD'])


def s6_noise_seed_prefix():
    return constant_bytes(_TEXT['NOISE_LABEL'])


def s7_legacy_alias():
    return constant_text(_TEXT['LEGACY_MODE'])
