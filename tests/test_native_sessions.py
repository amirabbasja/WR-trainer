"""Real socket boundaries: wire text never enters the Python JSON/framing path."""

import json
import socket
import threading
import unittest
from unittest.mock import patch

from workrunner.S3_client import S3Session
from workrunner.S3_transport import S3Transport
from workrunner.S6_codec import decode_frame, encode_frame
from workrunner.fixed_constants import constant_ascii_matches, constant_text, neutral_protocol_text
from workrunner.wr_client import WRSession


class IntoSocket:
    def __init__(self, connection):
        self.connection = connection
        self.read_views = 0
        self.write_views = 0

    def recv(self, *_args):
        raise AssertionError('production message path called recv instead of recv_into')

    def recv_into(self, target):
        if not isinstance(target, memoryview):
            raise AssertionError('native channel did not provide a borrowed memoryview')
        self.read_views += 1
        return self.connection.recv_into(target)

    def sendall(self, data):
        if not isinstance(data, memoryview):
            raise AssertionError('wire message escaped its native buffer')
        self.write_views += 1
        return self.connection.sendall(data)

    def shutdown(self, how):
        self.connection.shutdown(how)

    def close(self):
        self.connection.close()


class NativeSessionTests(unittest.TestCase):
    def test_neutral_context_is_scoped_and_does_not_change_binary_domains(self):
        self.assertEqual(constant_text('account'), 'wallet')
        with self.assertRaisesRegex(RuntimeError, 'test'):
            with neutral_protocol_text():
                self.assertEqual(constant_text('account'), 'account')
                self.assertTrue(constant_ascii_matches('extracounter', 'extracounter'))
                self.assertFalse(constant_ascii_matches('extracounter', 'extranonce'))
                raise RuntimeError('test')
        self.assertEqual(constant_text('account'), 'wallet')
        self.assertTrue(constant_ascii_matches('extracounter', 'extranonce'))

    def test_s3_and_wr_sessions_use_native_socket_buffers(self):
        for session_type, profile in ((S3Session, 's3'), (WRSession, 'wr')):
            with self.subTest(profile=profile):
                connection, peer = socket.socketpair()
                peer.settimeout(2)
                connection.settimeout(2)
                wrapper = IntoSocket(connection)
                settings = dict(host='127.0.0.1', port=1, KEX_uname='test', WR_S2='test', worker='device')
                session = session_type(settings)
                session.transport._socket = wrapper
                handled = threading.Event()
                received = []

                def handle(message):
                    received.append(message)
                    handled.set()

                session._handle = handle
                reader = threading.Thread(target=session._read_loop)
                try:
                    with patch.object(session.transport, 'send', side_effect=AssertionError('raw send')), patch.object(session.transport, 'receive', side_effect=AssertionError('raw receive')):
                        reader.start()
                        if profile == 's3':
                            outgoing = {'id': 7, 'method': 'compute.session', 'params': {'account': 'test'}}
                            session._send(outgoing)
                            raw = bytearray()
                            while not raw.endswith(b"\n"):
                                raw.extend(peer.recv(4096))
                            self.assertEqual(json.loads(raw), {'id': 7, 'method': 'mining.authorize', 'params': {'wallet': 'test'}})
                            incoming = b'{"method":"mining.notify","params":{"wallet":"value","hash":"unchanged"}}\n'
                            expected = {'method': 'training.notify', 'params': {'account': 'value', 'hash': 'unchanged'}}
                        else:
                            outgoing = {'id': 7, 'method': 'report_info', 'params': [{'prochPerSecond': 3.5}]}
                            session._send(outgoing)
                            raw = peer.recv(4096)
                            self.assertEqual(decode_frame(raw), {'id': 7, 'method': 'report_info', 'params': [{'hashrate': 3.5}]})
                            incoming = encode_frame({'id': -1, 'status': 1, 'result': ['extranonce', '0011223344556677']}, 7)
                            expected = {'id': -1, 'status': 1, 'result': ['extracounter', '0011223344556677']}
                        # The native channel must retain and combine fragments itself.
                        peer.sendall(incoming[:5])
                        peer.sendall(incoming[5:])
                        self.assertTrue(handled.wait(2))
                        self.assertEqual(received, [expected])
                        self.assertGreater(wrapper.read_views, 0)
                        self.assertGreater(wrapper.write_views, 0)
                finally:
                    session.close()
                    peer.close()
                    reader.join(2)
                self.assertFalse(reader.is_alive())


if __name__ == '__main__':
    unittest.main()
