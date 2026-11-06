import copy
import json
import io
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from workrunner.paths import DEFAULT_SOURCE
from workrunner.protocol import load_official_helpers


class RPCFieldLifetimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_official_helpers(DEFAULT_SOURCE)
        from job_gateway.rpc_types import TemplateEntry, RewardMetadata, WorkTemplateResponse
        cls.entry_model = TemplateEntry
        cls.metadata_model = RewardMetadata
        cls.template_model = WorkTemplateResponse

    def payload(self):
        return {
            'bits': '1d00ffff', 'curtime': 100, 'height': 14, 'version': 1,
            'target': 'ffff', 'maxtime': 200, 'mintime': 1,
            'mutable': [], 'capabilities': [],
            'previousblockhash': '00' * 32, 'vsizelimit': 100,
            'transactions': [{
                'data': '12', 'hash': '01' * 32, 'txid': '02' * 32,
                'depends': [], 'fee': 0, 'vsize': 1,
            }],
            'longpollid': 'abc', 'noncerange': '00000000ffffffff',
            'coinbaseaux': {'flags': 'cc'}, 'coinbasevalue': 1111,
            'default_witness_commitment': '00' * 32, 'requiredcertversion': 3,
        }

    def test_nested_wire_round_trip_preserves_input_and_neutral_access(self):
        payload = self.payload()
        original = copy.deepcopy(payload)
        model = self.template_model.model_validate_json(json.dumps(payload))
        self.assertEqual(model.records[0].record_id, '02' * 32)
        self.assertEqual(model.reward_amount, 1111)
        self.assertEqual(model.parent_digest, '00' * 32)
        self.assertEqual(model.reward_metadata.flags, 'cc')
        self.assertEqual(model.model_dump(), original)
        self.assertEqual(json.loads(model.model_dump_json()), original)
        self.assertEqual(payload, original)
        neutral = model.model_dump(by_alias=False)
        self.assertIn('parent_digest', neutral)
        self.assertNotIn('previousblockhash', neutral)
        self.assertIn('record_id', neutral['records'][0])
        self.assertEqual(self.template_model.model_validate(neutral).model_dump(), original)

    def test_native_wire_model_stream_and_scoped_output(self):
        class ReadIntoOnly(io.BytesIO):
            def read(self, *args):
                raise AssertionError('wire JSON must use a native readinto buffer')

        payload = self.payload()
        stream = ReadIntoOnly(json.dumps(payload).encode())
        model = self.template_model.model_validate_wire(stream)
        self.assertEqual(model.reward_amount, 1111)
        with model.wire_buffer(include={'records': {'__all__': {'record_id'}}}) as buffer:
            self.assertIsInstance(buffer, memoryview)
            self.assertEqual(json.loads(bytes(buffer)), {'transactions': [{'txid': '02' * 32}]})
            retained = buffer
        with self.assertRaises(ValueError):
            retained.tobytes()
        self.assertEqual(model.records[0].record_id, '02' * 32)

    def test_native_wire_input_errors_are_sanitized(self):
        with self.assertRaises(ValidationError) as caught:
            self.template_model.model_validate_wire(b'{"previousblockhash": "private-response", broken}')
        self.assertNotIn('private-response', str(caught.exception))
        self.assertNotIn('previousblockhash', str(caught.exception))
        self.assertIsNone(caught.exception.errors()[0]['input'])
        with self.assertRaises(TypeError):
            self.template_model.model_validate_wire('{}', by_name=True)

    def test_native_wire_limits_apply_to_raw_utf8_and_buffer_inputs(self):
        payload = self.payload()
        payload['longpollid'] = 'سلام'
        text = json.dumps(payload, ensure_ascii=False)
        raw = text.encode('utf-8')
        for source in (text, raw, bytearray(raw), memoryview(bytearray(raw))):
            with self.subTest(kind=type(source).__name__):
                self.assertEqual(
                    self.template_model.model_validate_wire(source, limit=len(raw)).poll_id,
                    'سلام',
                )
                with self.assertRaises(ValidationError) as caught:
                    self.template_model.model_validate_wire(source, limit=len(raw) - 1)
                self.assertEqual(caught.exception.errors()[0]['type'], 'json_invalid')
                self.assertIsNone(caught.exception.errors()[0]['input'])
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.template_model.model_validate_wire(raw, limit=limit)
        self.assertEqual(raw, text.encode('utf-8'))

    def test_runtime_alias_policy_and_precedence(self):
        payload = self.payload()
        neutral = self.template_model.model_validate(payload).model_dump(by_alias=False)
        for mixed in ({'reward_amount': 7, **payload}, {**payload, 'reward_amount': 7}):
            self.assertEqual(self.template_model.model_validate(mixed).reward_amount, 1111)
        with self.assertRaises(ValidationError):
            self.template_model.model_validate(payload, by_alias=False, by_name=True)
        self.assertEqual(
            self.template_model.model_validate(neutral, by_alias=False, by_name=True).reward_amount,
            1111,
        )
        with self.assertRaises(ValidationError):
            self.template_model.model_validate(neutral, by_alias=True, by_name=False)
        self.assertEqual(
            self.template_model.model_validate(payload, by_alias=True, by_name=False).records[0].record_id,
            '02' * 32,
        )

    def test_filtered_serialization_and_optional_defaults(self):
        payload = self.payload()
        del payload['default_witness_commitment'], payload['requiredcertversion']
        model = self.template_model.model_validate(payload)
        self.assertIsNone(model.default_attestation)
        self.assertEqual(int(model.required_version), 1)
        self.assertEqual(model.model_dump(include={'reward_amount'}), {'coinbasevalue': 1111})
        self.assertEqual(
            model.model_dump(include={'records': {'__all__': {'record_id'}}}),
            {'transactions': [{'txid': '02' * 32}]},
        )
        self.assertNotIn('default_witness_commitment', model.model_dump(exclude_none=True))

    def test_wire_names_are_not_retained_as_model_aliases_or_schema_strings(self):
        wire_names = {
            'txid', 'previousblockhash', 'vsizelimit', 'transactions', 'longpollid',
            'noncerange', 'coinbaseaux', 'coinbasevalue',
            'default_witness_commitment', 'requiredcertversion',
        }

        def strings(value):
            if isinstance(value, str):
                yield value
            elif isinstance(value, dict):
                for key, item in value.items():
                    yield from strings(key)
                    yield from strings(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    yield from strings(item)

        for model in (self.entry_model, self.metadata_model, self.template_model):
            with self.subTest(model=model.__name__):
                self.assertIs(model.model_config['cache_strings'], False)
                self.assertFalse(wire_names.intersection(model.model_fields))
                for field in model.model_fields.values():
                    self.assertIsNone(field.alias)
                    self.assertIsNone(field.validation_alias)
                    self.assertIsNone(field.serialization_alias)
                self.assertFalse(wire_names.intersection(strings(model.__pydantic_core_schema__)))
                self.assertFalse(wire_names.intersection(strings(model.model_json_schema())))

    def test_owned_translation_dictionaries_are_cleared_on_success_and_failure(self):
        from workrunner import rpc_fields

        original = rpc_fields._neutral_fields
        captured = []

        def observe(*args):
            translated = original(*args)
            captured.append(translated)
            return translated

        payload = self.payload()
        before = copy.deepcopy(payload)
        with patch.object(rpc_fields, '_neutral_fields', side_effect=observe):
            self.template_model.model_validate(payload)
            payload['height'] = 'invalid-height'
            with self.assertRaises(ValidationError):
                self.template_model.model_validate(payload)
        self.assertTrue(captured)
        self.assertTrue(all(value == {} for value in captured))
        payload['height'] = before['height']
        self.assertEqual(payload, before)

    def test_name_policy_applies_to_json_and_string_validation(self):
        from typing import ClassVar
        from workrunner.rpc_fields import WireModel
        from workrunner.word_codec import decode_word

        class Amount(WireModel):
            _wire_fields: ClassVar[tuple[str, ...]] = ('reward_amount',)
            reward_amount: int

        payload = {decode_word('reward_amount'): '123'}
        self.assertEqual(Amount.model_validate_strings(payload).reward_amount, 123)
        with self.assertRaises(ValidationError):
            Amount.model_validate_strings(payload, by_alias=False, by_name=True)
        with self.assertRaises(ValidationError):
            Amount.model_validate_json(json.dumps(payload), by_alias=False, by_name=True)
        self.assertEqual(
            Amount.model_validate_json('{"reward_amount": 123}', by_alias=False, by_name=True).reward_amount,
            123,
        )

    def test_extra_field_policy_preserves_shadowed_neutral_input(self):
        from pydantic import ConfigDict

        payload = self.payload()
        payload['reward_amount'] = 7
        original = copy.deepcopy(payload)
        serialized = {name: value for name, value in original.items() if name != 'reward_amount'}
        for by_name in (True, False):
            with self.subTest(by_name=by_name):
                with self.assertRaises(ValidationError):
                    self.template_model.model_validate(payload, by_alias=True, by_name=by_name, extra='forbid')
                model = self.template_model.model_validate(payload, by_alias=True, by_name=by_name, extra='allow')
                self.assertEqual(model.reward_amount, 1111)
                self.assertEqual(model.__pydantic_extra__, {'reward_amount': 7})
                self.assertEqual(model.model_dump(), serialized)
                self.assertEqual(
                    model.model_dump(include={'reward_amount'}),
                    {'coinbasevalue': 1111},
                )
                self.assertNotIn('coinbasevalue', model.model_dump(exclude={'reward_amount'}))

        class OpenTemplate(self.template_model):
            model_config = ConfigDict(extra='allow')

        model = OpenTemplate.model_validate(payload)
        self.assertEqual(model.model_dump(), original)
        self.assertEqual(model.reward_amount, 1111)
        self.assertEqual(model.model_dump(by_alias=False)['reward_amount'], 7)
        self.assertEqual(
            model.model_dump(include={'reward_amount'}),
            {'coinbasevalue': 1111, 'reward_amount': 7},
        )
        self.assertEqual(payload, original)
