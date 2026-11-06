import hashlib
from contextlib import contextmanager
import unittest
from unittest.mock import patch

from workrunner import fixed_constants
from workrunner.fixed_constants import constant_ascii_matches, constant_buffer, constant_bytes, constant_text


class FixedConstantTests(unittest.TestCase):
    def test_exact_binary_values_and_immutable_api(self):
        encoded = b'r>3Q(qoSdpudc1EtE#D}x3;ykv$C<UzrMY^ySll!hlYiOgMxv8kB*Iui;9VemzI^3lai5;pPrqZo0^%J=jP?)<Kp4q@9ypF>+0#~_xAPl^YZcV|Ni~_`}+C#$Hv9P!@|M9&(6)v%gV{f*Vfh4)6&t<-`?Ha+uGUKCnhB%BO)OnFD@-CD=H}{H#RjiGcqwSKR!J?J32Wy2L=TM0|Eg64-O3s3knGc7Zw#16A}>+A08bX8yXoHXJ%z&V`5=oZ*FaDYiem|cXoAjb8>NTe|~*@dwO|zM@B_NLqb77PfkrtOG-&dS5{S2Q&Le-UtV2YTUuFI'
        expected = bytes(range(256))
        self.assertEqual(constant_bytes(encoded), expected)
        with constant_buffer(encoded) as value:
            self.assertIsInstance(value, bytearray)
            self.assertEqual(value, expected)
            digest = hashlib.sha256(value).digest()
        self.assertEqual(digest, hashlib.sha256(expected).digest())
        self.assertEqual(value, bytearray(256))

    def test_cleanup_on_exception(self):
        with self.assertRaisesRegex(RuntimeError, "consumer failed"):
            with constant_buffer(b'#>K<') as value:
                self.assertEqual(value, b'abc')
                raise RuntimeError("consumer failed")
        self.assertEqual(value, bytearray(3))

    def test_text_and_empty_values(self):
        self.assertEqual(constant_text(b'#>K<'), 'abc')
        self.assertEqual(constant_bytes(b''), b'')
        self.assertEqual(constant_text(b''), '')

    def test_native_word_buffers_are_cleared_after_use_and_on_error(self):
        expected = bytes.fromhex('706561726c68617368')
        with constant_buffer('legacyMethod') as value:
            self.assertEqual(value, expected)
        self.assertEqual(value, bytearray(len(expected)))
        with self.assertRaisesRegex(RuntimeError, 'consumer failed'):
            with constant_buffer('legacyMethod') as value:
                self.assertEqual(value, expected)
                raise RuntimeError('consumer failed')
        self.assertEqual(value, bytearray(len(expected)))
        self.assertTrue(constant_ascii_matches('legacyMethod', expected.decode('ascii')))
        self.assertFalse(constant_ascii_matches('legacyMethod', 'unknown'))

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
            self.assertTrue(constant_ascii_matches(b'#>K<', 'abc'))
            for value in ('abd', 'ab', 'ab\u00e7', None, [], b'abc'):
                self.assertFalse(constant_ascii_matches(b'#>K<', value))
            self.assertTrue(constant_ascii_matches(b'', ''))
            with self.assertRaisesRegex(RuntimeError, 'comparison failed'):
                constant_ascii_matches(b'#>K<', BrokenText('abc'))
        self.assertTrue(buffers)
        self.assertTrue(all(not any(buffer) for buffer in buffers))
