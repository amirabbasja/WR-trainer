import json
import socket
import sys
import threading
from types import SimpleNamespace
import unittest

from workrunner.S3_client import S3Session
from workrunner.S3_transport import S3ConnectionError, S3Transport
from workrunner.S6_codec import encode_frame
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
                            if frame.f_code.co_name in ('send_message', '_send'):
                                inspected.add(frame.f_code.co_name)
                                for name in ('message', 'encoded', 'data'):
                                    self.assertNotIn(name, frame.f_locals)
                            traceback = traceback.tb_next
                    self.assertEqual(inspected, {'send_message', '_send'})
                else:
                    self.fail('closed socket unexpectedly accepted a request')


class ReaderReferenceTests(unittest.TestCase):
    def test_processed_frames_are_released_before_waiting_for_more_input(self):
        response = {'id': -1, 'status': 1, 'result': ['extranonce', '00' * 8]}
        for session_type in (S3Session, WRSession):
            with self.subTest(session=session_type.__name__):
                session = object.__new__(session_type)
                session._closed_event = threading.Event()
                frame = encode_frame(response) if session_type is WRSession else json.dumps(response).encode() + b'\n'
                incoming = [dict(response), dict(response)]
                observations = []
                handled = []
                failures = []

                def receive_message(**_options):
                    if incoming:
                        return incoming.pop()
                    reader_locals = sys._getframe(1).f_locals
                    observations.append({name for name in ('frame', 'response') if name in reader_locals})
                    session._closed_event.set()
                    return None

                session.transport = SimpleNamespace(receive_message=receive_message, discard_message_frame=lambda: None)
                session._handle = lambda message: handled.append(message['id'])
                session._fail = failures.append
                session._read_loop()
                self.assertEqual(handled, [-1, -1])
                self.assertEqual(observations, [set()])
                self.assertFalse(failures)
