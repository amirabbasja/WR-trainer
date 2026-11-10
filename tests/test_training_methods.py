from training_samples import sample_text

import base64
import gzip
import json
import socket
import unittest

from workrunner.S3_client import S3Session
from workrunner.S3_protocol import S3Job, S3ProtocolError


class TrainingMethodBoundaryTests(unittest.TestCase):
    def session(self, **settings):
        session = S3Session({
            "host": "127.0.0.1", "port": 1, "KEX_uname": "account",
            "worker": "device", "agent": "client/1", **settings,
        })
        self.addCleanup(session.close)
        return session

    def capture_requests(self, session, requests):
        connection, peer = socket.socketpair()
        peer.settimeout(1)
        session.transport._socket = connection
        with connection, peer, peer.makefile("rb") as reader:
            captured = []
            for request in requests:
                session._send(request)
                captured.append(json.loads(reader.readline()))
            return captured

    def test_neutral_requests_become_exact_s4_frames_at_send(self):
        session = self.session()
        job = S3Job("job-1", bytes(76), 1, 3, 10)
        requests = session.dialect.auth_requests(start_id=7)
        requests.append(session.dialect.submission_request(job, "eA==", 9))
        self.assertEqual([r["method"] for r in requests], [
            "compute.register", "compute.session", "work.complete",
        ])
        self.assertEqual(self.capture_requests(session, requests), [
            {"jsonrpc": "2.0", "id": 7, "method": sample_text('register_operation'),
             "params": ["client/1"]},
            {"jsonrpc": "2.0", "id": 8, "method": sample_text('session_operation'),
             "params": ["account.device", "x"]},
            {"jsonrpc": "2.0", "id": 9, "method": sample_text('complete_operation'),
             "params": {"worker": "account.device", "job_id": "job-1",
                        "plain_proof": "eA=="}},
        ])
        self.assertEqual([r["method"] for r in requests], [
            "compute.register", "compute.session", "work.complete",
        ])
        self.assertEqual(session.AUTH_METHOD, "compute.session")

    def test_gzip_authentication_and_proof_keep_the_wire_schema(self):
        session = self.session(compression="gzip")
        job = S3Job("job-1", bytes(76), 1, 3, 10)
        raw = b"native proof bytes" * 20
        requests = session.dialect.auth_requests(start_id=7)
        requests.append(session.dialect.submission_request(
            job, base64.b64encode(raw).decode("ascii"), 9, negotiated_gzip=True,
        ))
        auth, completed = self.capture_requests(session, requests)
        self.assertEqual(auth, {
            "jsonrpc": "2.0", "id": 7, "method": sample_text('session_operation'),
            "params": {sample_text('identity_field'): "account.device", "agent": "client/1",
                       "password": "x", "type": "v2"},
        })
        self.assertEqual(completed["method"], sample_text('complete_operation'))
        self.assertEqual(completed["id"], 9)
        self.assertEqual(completed["params"]["worker"], "account.device")
        self.assertEqual(completed["params"]["job_id"], "job-1")
        self.assertEqual(gzip.decompress(base64.b64decode(
            completed["params"]["plain_proof"])), raw)

    def test_only_wire_notifications_update_jobs(self):
        session = self.session()
        params = {"job_id": "job-1", "header": "00" * 76, "height": 1,
                  "target": "a", "cert_version": 3}
        session._handle({"method": sample_text('update_operation'), "params": params})
        self.assertEqual(session._job.job_id, "job-1")
        generation = session._generation
        for method in ("work.update", "compute.reconnect", "unknown", []):
            session._handle({
                "method": method, "params": {**params, "job_id": "job-2"},
            })
        self.assertEqual(session._job.job_id, "job-1")
        self.assertEqual(session._generation, generation)
        self.assertIsNone(session._failure)

    def test_both_wire_reconnect_methods_invalidate_work(self):
        for method in ("client.reconnect", sample_text('reconnect_operation')):
            with self.subTest(method=method):
                session = self.session()
                session._job = S3Job("job-1", bytes(76), 1, 3, 10)
                session._authorized = True
                session._handle({"method": method, "params": ["other.example", 9]})
                self.assertIsNone(session._job)
                self.assertFalse(session._authorized)
                self.assertIn("configured endpoint", str(session._failure))

    def test_invalid_update_reports_the_neutral_operation(self):
        session = self.session()
        with self.assertRaisesRegex(
            S3ProtocolError, r"work\.update params must be an object",
        ):
            session.dialect.parse_job([])


if __name__ == "__main__":
    unittest.main()
