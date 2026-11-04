import struct
import unittest

from workrunner.stream_digest import StreamDigest
from workrunner.noise_domains import certificate_salt, noise_draw, tensor_domain


class NoiseDomainTests(unittest.TestCase):
    def test_tensor_domains_preserve_bytes_and_are_cleared(self):
        for is_b, expected in ((False, '415f74656e736f72'), (True, '425f74656e736f72')):
            with self.subTest(is_b=is_b):
                with tensor_domain(is_b) as domain:
                    self.assertEqual(domain, bytes.fromhex(expected) + bytes(24))
                self.assertEqual(domain, bytearray(32))

    def test_noise_draw_matches_original_message_and_clears_on_failure(self):
        key = bytes(range(32))
        for slot in (0, 1):
            with self.subTest(slot=slot), tensor_domain() as domain:
                original = bytearray(64)
                struct.pack_into('<I', original, slot * 4, 18)
                original[32:] = bytes.fromhex('415f74656e736f72') + bytes(24)
                expected = StreamDigest(original, key=key).digest()
                self.assertEqual(noise_draw(StreamDigest, 17, domain, key, slot), expected)
        captured = []
        def fail(message, **kwargs):
            captured.append(message)
            raise RuntimeError('stop')
        with tensor_domain(True) as domain:
            with self.assertRaisesRegex(RuntimeError, 'stop'):
                noise_draw(fail, 0, domain, key, 1)
        self.assertEqual(captured, [bytearray(64)])

    def test_certificate_salt_preserves_original_bytes_and_clears_on_failure(self):
        prefix = bytes.fromhex('706561726c2f636572742d76332f6e6f6973652d736565642f')
        for label in (bytes([65]), bytes([66])):
            with self.subTest(label=label):
                self.assertEqual(certificate_salt(StreamDigest, label), StreamDigest(prefix + label).digest())
        captured = []
        def fail(message):
            captured.append(message)
            raise RuntimeError('stop')
        with self.assertRaisesRegex(RuntimeError, 'stop'):
            certificate_salt(fail, bytes([65]))
        self.assertEqual(captured, [bytearray(len(prefix) + 1)])
