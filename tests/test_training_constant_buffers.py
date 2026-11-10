import hashlib
from contextlib import contextmanager
import unittest
from unittest.mock import patch

from workrunner import fixed_constants
from workrunner.fixed_constants import FixedAlias, constant_ascii_matches, constant_buffer, constant_bytes, constant_text
from workrunner.native_text_codec import encode_fixed


class FixedConstantTests(unittest.TestCase):
    def test_exact_binary_values_and_immutable_api(self):
        expected = bytes(range(256))
        alias = encode_fixed(expected)
        self.assertIsInstance(alias, FixedAlias)
        self.assertEqual(constant_bytes(alias), expected)
        with constant_buffer(alias) as value:
            self.assertIsInstance(value, bytearray)
            self.assertEqual(value, expected)
            digest = hashlib.sha256(value).digest()
        self.assertEqual(digest, hashlib.sha256(expected).digest())
        self.assertEqual(value, bytearray(256))

    def test_cleanup_on_exception(self):
        with self.assertRaisesRegex(RuntimeError, "consumer failed"):
            with constant_buffer(FixedAlias('training.fixture.text_sample')) as value:
                self.assertEqual(value, b'abc')
                raise RuntimeError("consumer failed")
        self.assertEqual(value, bytearray(3))

    def test_text_and_empty_values(self):
        self.assertEqual(constant_text(FixedAlias('training.fixture.text_sample')), 'abc')
        self.assertEqual(constant_bytes(b''), b'')
        self.assertEqual(constant_text(b''), '')
        self.assertEqual(constant_bytes(encode_fixed(b'')), b'')

    def test_fixed_aliases_decode_inside_the_neutral_wire_context(self):
        alias = FixedAlias('training.field.acceptance_bound')
        with fixed_constants.neutral_protocol_text():
            self.assertEqual(constant_text(alias), 'target')
            self.assertEqual(constant_bytes(bytes(alias)), bytes(alias))
        with self.assertRaisesRegex(ValueError, 'unknown fixed constant'):
            constant_bytes(FixedAlias('training.field.missing'))
        with self.assertRaisesRegex(ValueError, 'unknown fixed constant'):
            encode_fixed(b'unlisted original value')

    def test_native_word_buffers_are_cleared_after_use_and_on_error(self):
        expected = bytes.fromhex('706561726c68617368')
        with constant_buffer('legacy_protocol') as value:
            self.assertEqual(value, expected)
        self.assertEqual(value, bytearray(len(expected)))
        with self.assertRaisesRegex(RuntimeError, 'consumer failed'):
            with constant_buffer('legacy_protocol') as value:
                self.assertEqual(value, expected)
                raise RuntimeError('consumer failed')
        self.assertEqual(value, bytearray(len(expected)))
        self.assertTrue(constant_ascii_matches('legacy_protocol', expected.decode('ascii')))
        self.assertFalse(constant_ascii_matches('legacy_protocol', 'unknown'))

    def test_ascii_comparison_clears_buffers_for_matches_mismatches_and_errors(self):
        buffers = []

        @contextmanager
        def observe(encoded):
            with constant_buffer(encoded) as buffer:
                buffers.append(buffer)
                yield buffer

        class BrokenText(str):
            def __iter__(self):
                raise RuntimeError('comparison failed')

        with patch.object(fixed_constants, 'constant_buffer', observe):
            self.assertTrue(constant_ascii_matches(FixedAlias('training.fixture.text_sample'), 'abc'))
            for value in ('abd', 'ab', 'ab\u00e7', None, [], b'abc'):
                self.assertFalse(constant_ascii_matches(FixedAlias('training.fixture.text_sample'), value))
            self.assertTrue(constant_ascii_matches(b'', ''))
            with self.assertRaisesRegex(RuntimeError, 'comparison failed'):
                constant_ascii_matches(FixedAlias('training.fixture.text_sample'), BrokenText('abc'))
        self.assertTrue(buffers)
        self.assertTrue(all(not any(buffer) for buffer in buffers))
