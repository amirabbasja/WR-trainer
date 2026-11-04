import socket
import unittest

from workrunner.S3_client import S3Session
from workrunner.S3_transport import S3ConnectionError, S3Transport
from workrunner.wr_client import WRSession


class SendReferenceTests(unittest.TestCase):
    def test_failed_send_does_not_retain_serialized_requests_in_our_frames(self):
        for session_type in (S3Session, WRSession):
            with self.subTest(session=session_type.__name__):
                connection, peer = socket.socketpair()
                connection.close()
                peer.close()
                transport = S3Transport('127.0.0.1', 1)
                transport._socket = connection
                session = object.__new__(session_type)
                session.transport = transport
                try:
                    session._send({'id': 1, 'method': 'example'})
                except S3ConnectionError as error:
                    pending = [error]
                    seen = set()
                    inspected = set()
                    while pending:
                        current = pending.pop()
                        if id(current) in seen:
                            continue
                        seen.add(id(current))
                        if current.__cause__ is not None:
                            pending.append(current.__cause__)
                        if current.__context__ is not None:
                            pending.append(current.__context__)
                        traceback = current.__traceback__
                        while traceback is not None:
                            frame = traceback.tb_frame
                            if frame.f_code.co_name in ('send', '_send'):
                                inspected.add(frame.f_code.co_name)
                                for name in ('message', 'encoded', 'data'):
                                    self.assertNotIn(name, frame.f_locals)
                            traceback = traceback.tb_next
                    self.assertEqual(inspected, {'send', '_send'})
                else:
                    self.fail('closed socket unexpectedly accepted a request')
