"""Per-message field translation without decoded aliases in model schemas."""

from collections.abc import Mapping
from contextvars import ContextVar
from contextlib import contextmanager
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, ValidationError, model_serializer, model_validator

from .word_codec import decode_word, encode_word
from .wire_io import WireError, WireLimitError, dumps_buffer, loads, read_json_body


_name_policy = ContextVar('rpc_field_name_policy', default=(None, None))
_extra_policy = ContextVar('rpc_field_extra_policy', default=None)


def _validate_with_policy(method, value, options):
    token = _name_policy.set((options.get('by_alias'), options.get('by_name')))
    extra_token = _extra_policy.set(options.get('extra'))
    try:
        return method(value, **options)
    finally:
        _name_policy.reset(token)
        _extra_policy.reset(extra_token)
        del value, options, method


def _neutral_fields(value, fields, allow_alias, allow_name, deferred=None):
    translated = {}
    aliases = {}
    name = item = candidate = None
    try:
        for name, item in value.items():
            if isinstance(name, str):
                candidate = encode_word(name) if allow_alias else name
                if candidate != name and candidate in fields:
                    aliases[candidate] = item
                    continue
                if name in fields and not allow_name:
                    if deferred is not None:
                        deferred[name] = item
                    continue
            translated[name] = item
        if deferred is not None:
            deferred.update((name, translated[name]) for name in aliases if name in translated)
        translated.update(aliases)
        return translated
    except BaseException:
        translated.clear()
        raise
    finally:
        aliases.clear()
        del value, name, item, candidate


class WireModel(BaseModel):
    model_config = ConfigDict(
        validate_by_name=True,
        serialize_by_alias=True,
        cache_strings=False,
    )
    _wire_fields: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def model_validate(cls, obj: Any, **options):
        try:
            return _validate_with_policy(super().model_validate, obj, options)
        finally:
            del obj, options

    @classmethod
    def model_validate_json(cls, json_data: str | bytes | bytearray, **options):
        """Use native translation for the usual alias/name policy.

        Explicit Pydantic name policies and strict JSON semantics remain a
        compatibility API; those callers own any original input strings.
        """
        try:
            explicit_policy = any(options.get(name) is not None for name in ('by_alias', 'by_name'))
            usual_policy = (cls.model_config.get('validate_by_alias', True)
                            and cls.model_config.get('validate_by_name', False))
            strict = options.get('strict')
            if strict is None:
                strict = cls.model_config.get('strict', False)
            if explicit_policy or not usual_policy or strict:
                return _validate_with_policy(super().model_validate_json, json_data, options)
            options.pop('by_alias', None)
            options.pop('by_name', None)
            return cls.model_validate_wire(json_data, **options)
        finally:
            del json_data, options

    @classmethod
    def model_validate_wire(cls, source, *, limit=16 * 1024 * 1024, **options):
        """Read native JSON buffers and validate neutral Python fields.

        The byte limit applies to streams and raw inputs. A stream must
        implement readinto(). A caller-supplied str/bytes input
        remains caller-owned; use a stream to avoid that original Python copy.
        """
        neutral = None
        try:
            if 'by_alias' in options or 'by_name' in options:
                raise TypeError('model_validate_wire selects neutral field names')
            if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
                raise ValueError('body limit must be a positive integer')
            try:
                if hasattr(source, 'readinto'):
                    neutral = read_json_body(source, limit, profile='rpc')
                else:
                    if isinstance(source, str):
                        # Count UTF-8 bytes without creating another immutable
                        # copy of the caller's original JSON. ASCII is usual.
                        count = len(source)
                        if count <= limit and not source.isascii():
                            count = sum(1 + (ord(c) > 0x7f) + (ord(c) > 0x7ff)
                                        + (ord(c) > 0xffff) for c in source)
                    else:
                        with memoryview(source) as view:
                            count = view.nbytes
                    if count > limit:
                        raise WireLimitError('wire input exceeds the configured size limit')
                    neutral = loads(source, profile='rpc')
            except (WireError, UnicodeError):
                raise ValidationError.from_exception_data(cls.__name__, [{
                    'type': 'json_invalid', 'loc': (), 'input': None,
                    'ctx': {'error': 'Invalid JSON message'},
                }]) from None
            return cls.model_validate(neutral, by_alias=False, by_name=True, **options)
        finally:
            if isinstance(neutral, (dict, list)):
                neutral.clear()
            del source, neutral, options

    @contextmanager
    def wire_buffer(self, **options):
        """Serialize to a scoped native wire buffer, wiped after use.

        model_dump()/model_dump_json() remain compatibility APIs whose returned
        decoded names belong to the caller. Use this API for network writes.
        """
        neutral = None
        try:
            if options.get('by_alias'):
                raise TypeError('wire_buffer serializes neutral model fields')
            if any(name in (self.__pydantic_extra__ or {}) for name in self._wire_fields):
                raise ValueError('wire_buffer cannot serialize shadowing extra fields')
            options['by_alias'] = False
            options.setdefault('mode', 'json')
            neutral = self.model_dump(**options)
            with dumps_buffer(neutral, profile='rpc') as buffer:
                try:
                    yield buffer
                finally:
                    del buffer
        finally:
            if isinstance(neutral, dict):
                neutral.clear()
            del self, neutral, options

    @classmethod
    def model_validate_strings(cls, obj: Any, **options):
        try:
            return _validate_with_policy(super().model_validate_strings, obj, options)
        finally:
            del obj, options

    @model_validator(mode='wrap')
    @classmethod
    def _read_wire_fields(cls, value, handler):
        translated = None
        deferred = {}
        errors = []
        result = None
        try:
            if not cls._wire_fields or not isinstance(value, Mapping):
                return handler(value)
            allow_alias, allow_name = _name_policy.get()
            if allow_alias is None:
                allow_alias = cls.model_config.get('validate_by_alias', True)
            if allow_name is None:
                allow_name = cls.model_config.get('validate_by_name', False)
            translated = _neutral_fields(value, cls._wire_fields, allow_alias, allow_name, deferred)
            extra = _extra_policy.get() or cls.model_config.get('extra')
            if extra == 'forbid':
                errors = [
                    {'type': 'extra_forbidden', 'loc': (name,), 'input': item}
                    for name, item in deferred.items()
                ]
            try:
                result = handler(translated)
            except ValidationError as error:
                if errors:
                    raise ValidationError.from_exception_data(
                        cls.__name__, error.errors(include_url=False) + errors,
                    ) from None
                raise
            if errors:
                raise ValidationError.from_exception_data(cls.__name__, errors)
            if extra == 'allow' and deferred:
                result.__pydantic_extra__.update(deferred)
            return result
        finally:
            if translated is not None:
                translated.clear()
            deferred.clear()
            errors.clear()
            del value, translated, handler, deferred, errors, result

    @model_serializer(mode='wrap')
    def _write_wire_fields(self, handler, info):
        values = translated = name = item = key = extras = None
        copy_model = None
        shadowed = {}
        try:
            values = handler(self)
            by_alias = info.by_alias
            if by_alias is None:
                by_alias = self.model_config.get('serialize_by_alias', False)
            if not by_alias or not self._wire_fields or not isinstance(values, dict):
                return values
            extras = (self.__pydantic_extra__ or {}) if self.model_config.get('extra') == 'allow' else {}
            shadowed = {name: values[name] for name in self._wire_fields if name in extras and name in values}
            if shadowed:
                copy_model = self.model_copy()
                for name in shadowed:
                    copy_model.__pydantic_extra__.pop(name, None)
                values.clear()
                values = handler(copy_model)
            translated = {}
            for name, item in values.items():
                key = decode_word(name) if name in self._wire_fields else name
                translated[key] = item
            translated.update(shadowed)
            return translated
        except BaseException:
            if translated is not None:
                translated.clear()
            raise
        finally:
            if translated is not None:
                values.clear()
            shadowed.clear()
            del self, values, translated, name, item, key, extras, handler, info, copy_model
