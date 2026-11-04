from .fixed_constants import constant_bytes, constant_text


_TEXT = {
    'ACCOUNT': b'(8kNl!qW',
    'CLIENT_SEED': b'%FfBc)%eNC$;83',
    'NOISE_LABEL': b'*TTlt%ZS9n)zga6l!(d5&eXz+)WX8Uhy',
    'LEGACY_MODE': b'*TTlt%g@Ht&j',
    'SEND_RESULT': b')Y8Su&eH',
    'JOB_UPDATE': b'$;i{r!QK',
    'DEVICE_REPORT': b')xy`v)zkRS$-&3',
    'LOGIN': b'%gDga$p',
    'DEVICE_ID': b'(8$%y!qo',
    'SERVER_SEED': b'!r#-?#>vRZ#KH',
    'DEVICE_SEED': b'(8$%y!qxc6$jQXQ',
    'LOCAL_SEED': b'^pg0=$jQXQ',
    'SEED': b'$;ipX!T',
    'RATE': b'&&Jfx)yC7p',
    'EVIDENCE': b'*UQGv$@tgR$jHF',
    'THRESHOLD': b')5g`n!qW',
    'FORMAT_SETTING': b'#KP6n&cV*a#?!+1(Zbc#&dA9',
    'FORMAT_FIELD': b'#KP6n_|d}E)XvDs',
    'JOB_ID': b'&B(?0&cg',
    'DEFAULT_DEVICE': b')y}|*mzD',
    'CLIENT_AGENT': b'>-Fa1_lnci#?HyY)rgmgmH',
    'DEFAULT_TITLE': b'>-Fa1_lofKiu2XR&dI{n',
    'SOLO_PREFIX': b')X2-ood',
    'ADDRESS_PREFIX': b'*VW6G',
}
_RATE_UNITS = (b'?}*d', b'%<qWQ', b'>hFlu', b';O~gk', b'^Y4h%', b'_wR_*', b';_rym')
LOGGER_NAME = __package__
__all__ = [*_TEXT, 'RATE_UNITS', 'LOGGER_NAME', 'local_seed_field',
           's6_noise_seed_prefix', 's7_legacy_alias']


def __getattr__(name):
    if name == 'RATE_UNITS':
        return tuple(constant_text(unit) for unit in _RATE_UNITS)
    try:
        encoded = _TEXT[name]
    except KeyError:
        raise AttributeError(name) from None
    return constant_text(encoded)


def local_seed_field():
    return constant_text(_TEXT['CLIENT_SEED'])


def s6_noise_seed_prefix():
    return constant_bytes(_TEXT['NOISE_LABEL'])


def s7_legacy_alias():
    return constant_text(_TEXT['LEGACY_MODE'])
