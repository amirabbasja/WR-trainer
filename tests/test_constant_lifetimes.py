from contextlib import contextmanager
from hashlib import sha256
import unittest
from unittest.mock import patch

from workrunner import S3_methods, S6_codec, wire_constants, wr_protocol
from workrunner.S3_protocol import KEXDialect
from workrunner.fixed_constants import constant_buffer


class ConstantLifetimeTests(unittest.TestCase):
    def test_tables_match_original_bytes(self):
        for value, expected in (
            (S6_codec.coefficients(), 'c6e8f8581586f378a4de6bad03bf8b2248f8cd2075be51087659198a40a39db2'),
            (wr_protocol.header_mask(), '9654a5630b6051d3635a690f4cc9dbfc028b5bf597b221ce1363876a5628b1f7'),
            (wr_protocol.work_prefix_mask(), '461787c2ba9e41d25c97b976ce0c70a1459ca2e8e06c0a70f0648afc727a79bd'),
        ):
            self.assertIsInstance(value, bytes)
            self.assertEqual(sha256(value).hexdigest(), expected)

    def test_normal_codec_call_clears_decoded_table(self):
        buffers = []

        @contextmanager
        def observe(encoded):
            with constant_buffer(encoded) as buffer:
                buffers.append(buffer)
                yield buffer

        with patch.object(S6_codec, 'constant_buffer', observe):
            framed = S6_codec.encode_frame({'id': 1}, seed=7)
            self.assertEqual(S6_codec.decode_frame(framed), {'id': 1})
        self.assertEqual(len(buffers), 2)
        self.assertTrue(all(not any(buffer) for buffer in buffers))

    def test_protocol_constants_are_not_decoded_globals_or_defaults(self):
        self.assertNotIn('NOISE_LABEL', vars(wire_constants))
        self.assertTrue(all(isinstance(value, (bytes, str)) for value in wire_constants._TEXT.values()))
        self.assertTrue(all(isinstance(value, str) for value in S3_methods._REQUEST_METHODS.values()))
        self.assertTrue(all(isinstance(value, (bytes, str)) for value, _ in S3_methods._NOTIFICATION_METHODS))
        dialect = KEXDialect('account', 'unit')
        self.assertNotIsInstance(dialect._agent, str)
        self.assertEqual(dialect.agent, wire_constants.CLIENT_AGENT)
        self.assertEqual(dialect.auth_requests()[0]['method'], 'compute.register')

    def test_notification_and_legacy_comparisons_do_not_create_decoded_strings(self):
        from workrunner.config import _S3_protocol
        from workrunner.S3_training import select_S3_protocol, validate_S3_endpoint

        legacy = bytes.fromhex('706561726c68617368').decode('ascii')
        with patch.object(S3_methods, 'constant_text', side_effect=AssertionError('decoded string created')):
            self.assertEqual(S3_methods.s4_notification_method('mining.notify'), S3_methods.WORK_UPDATE)
            self.assertIsNone(S3_methods.s4_notification_method('unknown'))
        with patch.object(wire_constants, 'constant_text', side_effect=AssertionError('decoded string created')):
            self.assertEqual(_S3_protocol(legacy), 'WR')
            self.assertEqual(select_S3_protocol({'protocol': legacy}), 'WR')
            with self.assertRaisesRegex(ValueError, 'WR'):
                validate_S3_endpoint({'protocol': legacy, 'host': ''})
