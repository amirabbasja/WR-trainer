from training_samples import sample_bytes, sample_text

"""Real socket boundaries: wire text never enters the Python JSON/framing path."""

import copy
import json
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest.mock import patch

from workrunner.S3_client import S3Session
from workrunner.S3_transport import S3Transport
from workrunner.S6_codec import decode_frame, encode_frame
from workrunner.config import ConfigError, DEFAULT_CONFIG, save_config
from workrunner.fixed_constants import constant_ascii_matches, constant_text, neutral_protocol_text
from workrunner.login_config import load_training_config, login_frame
from workrunner.paths import DEFAULT_SOURCE
from workrunner.training_client import TrainingSession


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
    def test_wr_login_uses_fresh_config_and_wipes_the_send_buffer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            values = copy.deepcopy(DEFAULT_CONFIG)
            values['official_source'] = str(DEFAULT_SOURCE)
            values['vpn']['link'] = 'vless://fixture@example.invalid:443'
            values['S3'].update(host='127.0.0.1', WR_S2='first-login', KEX_uname='unused-login')
            save_config(path, values)
            settings = load_training_config(path)['S3']
            self.assertNotIn('WR_S2', settings)
            self.assertNotIn('KEX_uname', settings)

            for identity in ('first-login', 'second-login'):
                values['S3']['WR_S2'] = identity
                save_config(path, values)
                session = TrainingSession(settings, login_config=path)
                buffers = []
                with patch.object(session, '_start_auth_reader'), patch.object(session.transport, 'send') as send:
                    def receive(view):
                        buffers.append(view.obj)
                        self.assertEqual(decode_frame(view.tobytes())['params'][0], identity)
                        self.assertIsNone(session.dialect.training_identity)

                    send.side_effect = receive
                    session._send_auth_requests()
                    send.assert_called_once()
                    self.assertFalse(any(buffers[0]))

            values['S3']['host'] = '127.0.0.2'
            save_config(path, values)
            with self.assertRaisesRegex(ConfigError, 'endpoint changed'):
                with login_frame(path, 1, settings, 'device', 'client'):
                    self.fail('Login frame was prepared for a changed endpoint')

    def test_wr_login_frame_is_wiped_after_send_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            values = copy.deepcopy(DEFAULT_CONFIG)
            values['official_source'] = str(DEFAULT_SOURCE)
            values['vpn']['link'] = 'vless://fixture@example.invalid:443'
            values['S3']['WR_S2'] = 'login-fixture'
            save_config(path, values)
            with self.assertRaisesRegex(OSError, 'send failed'):
                with login_frame(path, 1, values['S3'], 'device', 'client') as frame:
                    buffer = frame.obj
                    self.assertTrue(any(buffer))
                    raise OSError('send failed')
            self.assertFalse(any(buffer))

    def test_neutral_context_is_scoped_and_does_not_change_binary_domains(self):
        self.assertEqual(constant_text('account'), sample_text('identity_field'))
        with self.assertRaisesRegex(RuntimeError, 'test'):
            with neutral_protocol_text():
                self.assertEqual(constant_text('account'), 'account')
                self.assertTrue(constant_ascii_matches('server_seed', 'server_seed'))
                self.assertFalse(constant_ascii_matches('server_seed', sample_text('server_seed_field')))
                raise RuntimeError('test')
        self.assertEqual(constant_text('account'), sample_text('identity_field'))
        self.assertTrue(constant_ascii_matches('server_seed', sample_text('server_seed_field')))

    def test_s3_and_training_sessions_use_native_socket_buffers(self):
        for session_type, profile in ((S3Session, 's3'), (TrainingSession, 'wr')):
            with self.subTest(profile=profile):
                connection, peer = socket.socketpair()
                peer.settimeout(2)
                connection.settimeout(2)
                wrapper = IntoSocket(connection)
                settings = {
                    'host': '127.0.0.1', 'port': 1, 'KEX_uname': 'test',
                    'WR_S2': 'test', 'worker': 'device',
                }
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
                            self.assertEqual(json.loads(raw), {'id': 7, 'method': sample_text('session_operation'), 'params': {sample_text('identity_field'): 'test'}})
                            incoming = sample_bytes('s3_and_training_sessions_use_native_socket_buffers')
                            expected = {'method': 'training.notify', 'params': {'account': 'value', 'hash': 'unchanged'}}
                        else:
                            outgoing = {'id': 7, 'method': 'report_info', 'params': [{'training_rate': 3.5}]}
                            session._send(outgoing)
                            raw = peer.recv(4096)
                            self.assertEqual(decode_frame(raw), {'id': 7, 'method': 'report_info', 'params': [{sample_text('rate_field'): 3.5}]})
                            incoming = encode_frame({'id': -1, 'status': 1, 'result': [sample_text('server_seed_field'), '0011223344556677']}, 7)
                            expected = {'id': -1, 'status': 1, 'result': ['server_seed', '0011223344556677']}
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
