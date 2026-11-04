import json
import unittest

from workrunner.paths import DEFAULT_SOURCE
from workrunner.protocol import load_official_helpers


def ascii_fixture(hex_bytes):
    return bytes.fromhex(hex_bytes).decode('ascii')


class RecordCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_official_helpers(DEFAULT_SOURCE)

    def test_template_json_round_trip(self):
        from job_gateway.rpc_types import WorkTemplateResponse

        entry = {
            'data': '12', 'hash': '01' * 32, ascii_fixture('74786964'): '02' * 32,
            'depends': [], 'fee': 0, 'vsize': 1,
        }
        payload = {
            'bits': '1d00ffff', 'curtime': 100, 'height': 14, 'version': 1,
            'target': 'ffff', 'maxtime': 200, 'mintime': 1, 'mutable': [], 'capabilities': [],
            ascii_fixture('70726576696f7573626c6f636b68617368'): '00' * 32,
            ascii_fixture('7673697a656c696d6974'): 100,
            ascii_fixture('7472616e73616374696f6e73'): [entry],
            ascii_fixture('6c6f6e67706f6c6c6964'): 'abc',
            ascii_fixture('6e6f6e636572616e6765'): '00000000ffffffff',
            ascii_fixture('636f696e62617365617578'): {'flags': 'cc'},
            ascii_fixture('636f696e6261736576616c7565'): 1111,
            ascii_fixture('64656661756c745f7769746e6573735f636f6d6d69746d656e74'): '00' * 32,
            ascii_fixture('72657175697265646365727476657273696f6e'): 3,
        }
        model = WorkTemplateResponse.model_validate_json(json.dumps(payload))
        self.assertEqual(model.records[0].record_id, '02' * 32)
        self.assertEqual(model.reward_amount, 1111)
        self.assertEqual(json.loads(model.model_dump_json()), payload)

    def test_tagged_reward_record_bytes(self):
        from job_gateway.record_format.assembly import build_reward_record

        address = ascii_fixture(
            '70726c3170717171737971637971357271777a7166706739736372677770756770'
            '7a79736e7a73323376396363727964706b3871617263307373383732396b'
        )
        expected = bytes.fromhex(
            '0100000000010100000000000000000000000000000000000000000000000000'
            '00000000000000ffffffff035e00ccffffffff02570400000000000022512000'
            '0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f00'
            '00000000000000266a24aa21' 'a9ed000000000000000000000000000000000000'
            '0000000000000000000000000000012000000000000000000000000000000000'
            '0000000000000000000000000000000000000000'
        )
        record = build_reward_record(14, 1111, address, {'flags': 'cc'}, '00' * 32)
        self.assertEqual(record.to_bytes(), expected)
        self.assertEqual(record.get_record_id(), 'a5725f77701acfd821da76c0efc82b8621eb4c57fd710c5e568eca8792f76343')
