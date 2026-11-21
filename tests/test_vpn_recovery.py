"""Exercise VPN recovery through the real scheduling and batch-control loop."""

import copy
from contextlib import ExitStack, nullcontext
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import MagicMock, patch

from workrunner import S3_training as worker, training_work
from workrunner.config import DEFAULT_CONFIG
from workrunner.gpu_noisy_gemm import TrainingCancelled
from workrunner.socks_proxy import SocksProxy, routed
from workrunner.vpn_session_guard import VPNJobGuard


class VPNRecoveryLoopTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.now = 0.0
        self.guards = []
        self.prepared_at = []
        self.connected_at = []
        self.statuses = []
        config = self.config = copy.deepcopy(DEFAULT_CONFIG)
        config['training'].update(m=128, n=128, k=2048, tile_h=8, tile_w=8,
            max_batches=1, max_seconds=0,
            job_sequence=[{'title': 'job-0', 'time': 120, 'max_gpu_utilization': 100}])
        config['S3'].update(protocol='WR', host='fixture.invalid', WR_S2='fixture', reconnect_delay=1)
        config['vpn'].update(enabled=True, check_url='https://health.invalid')
        self.stack.enter_context(patch.object(worker.time, 'monotonic', side_effect=lambda: self.now))
        outer = self
        class ClockEvent:
            def __init__(self):
                self.stopped = False
            def is_set(self):
                return self.stopped
            def set(self):
                self.stopped = True
            def wait(self, seconds):
                outer.now += seconds
                return self.stopped
        self.stack.enter_context(patch.object(worker.threading, 'Event', ClockEvent))
        self.stack.enter_context(patch.object(VPNJobGuard, 'start',
            lambda guard: self.guards.append(guard) or guard))
        for name in ('load_official_helpers', 'validate_S3_identity', '_release_work'):
            self.stack.enter_context(patch.object(worker, name))
        self.stack.enter_context(patch.object(worker, 'resolve_cuda_devices', return_value=(worker.torch.device('cuda:0'),)))
        self.stack.enter_context(patch.object(worker.native_cuda, 'available', return_value=False))
        self.stack.enter_context(patch.object(worker.torch.cuda, 'get_device_name', return_value='test GPU'))
        self.stack.enter_context(patch.object(worker.torch.cuda, 'device', side_effect=lambda _: nullcontext()))
        self.stack.enter_context(patch.object(worker, 'GpuPacer'))
        self.stack.enter_context(patch.object(worker, 'GpuUtilization'))
        self.stack.enter_context(patch.object(worker, 'TelegramUpdates'))
        self.stack.enter_context(patch.object(worker, 'S3_training_threshold', return_value=1))
        self.display = self.stack.enter_context(patch.object(worker, 'LiveProgress')).return_value.__enter__.return_value
        self.display.update.side_effect = lambda status, **_: self.statuses.append(status)
        primitives = ModuleType('workrunner.protocol_types')
        primitives.PartialHeader = SimpleNamespace(from_bytes=lambda _: object())
        self.stack.enter_context(patch.dict(sys.modules, {'workrunner.protocol_types': primitives}))
        self.prepare = self.stack.enter_context(patch.object(training_work, 'prepare_training_batch',
            side_effect=lambda *_, **__: self.prepared_at.append(self.now) or object()))
        self.compute = self.stack.enter_context(patch.object(worker, 'search_batch', return_value=[]))
        self.verify = self.stack.enter_context(patch.object(training_work, 'verified_training_submission', return_value='proof'))
        self.sessions = []
        def connect(*_, **__):
            session = MagicMock()
            session.__enter__.return_value = session
            snapshot = SimpleNamespace(generation=len(self.sessions), job=SimpleNamespace(
                job_id='fresh-job', cert_version=3, header_bytes=bytes(76), training_threshold=1))
            session.wait_job.return_value = snapshot
            session.is_current.return_value = True
            session.send_result.return_value = SimpleNamespace(confirmed=True)
            self.sessions.append(session)
            self.connected_at.append(self.now)
            return session
        self.stack.enter_context(patch('workrunner.training_client.TrainingSession', side_effect=connect))
        self.stack.enter_context(routed(SocksProxy('127.0.0.1', 19080)))

    def test_no_gpu_work_or_pool_session_until_health_retry_succeeds(self):
        def probe(guard):
            self.statuses.append(self.display.start_refresh.call_args.args[0]())
            if self.now < 2:
                self.assertEqual(self.prepare.call_count, 0)
                self.assertEqual(self.sessions, [])
                raise OSError('offline')
            return 42.0
        with patch.object(VPNJobGuard, '_probe', probe):
            self.assertEqual(worker.run_S3_training(self.config), 0)
        self.assertEqual(self.connected_at, [2])
        self.assertEqual(self.prepared_at, [2])
        self.assertTrue(any('VPN: Disconnected; retrying in' in status for status in self.statuses))
        self.assertTrue(any('VPN: Disconnected; retrying' in status and 'retrying in' not in status for status in self.statuses))
        self.assertTrue(any('VPN: Connected (42 ms)' in status for status in self.statuses))

    def test_loss_during_compute_discards_old_work_then_uses_fresh_session(self):
        def compute(_batch, _threshold, _device, cancelled):
            if self.compute.call_count == 1:
                self.guards[0].connection_lost()
                self.assertTrue(cancelled())
                raise TrainingCancelled('disconnected')
            self.assertFalse(cancelled())
            return [(0, 0, bytes(32))]
        self.compute.side_effect = compute
        with patch.object(VPNJobGuard, '_probe', return_value=42.0):
            self.assertEqual(worker.run_S3_training(self.config), 0)
        self.assertEqual(self.connected_at, [0, 1])
        self.assertEqual(self.prepared_at, [0, 1])
        self.sessions[0].close.assert_called_once_with()
        self.sessions[0].send_result.assert_not_called()
        self.verify.assert_called_once()
        self.sessions[1].send_result.assert_called_once_with(self.sessions[1].wait_job.return_value, 'proof')


if __name__ == '__main__':
    unittest.main()
