import unittest
from unittest.mock import Mock, patch

from workrunner import vpn_session_guard


class Schedule:
    def __init__(self):
        self.number = 1
        self.kind = 'job-0'
        self.percentage = 100
        self.boundary = 100.0

    def position(self):
        return self.number, self.kind, self.percentage, self.boundary

    def current_job(self):
        return self.kind, self.percentage, self.boundary


class VPNLatencyTests(unittest.TestCase):
    def setUp(self):
        self.schedule = Schedule()
        self.updates = Mock()
        self.manager = Mock()
        self.manager.process.poll.return_value = None
        self.clock = 0.0
        self.monotonic = patch.object(vpn_session_guard.time, 'monotonic', return_value=10.0)
        self.monotonic.start()
        self.addCleanup(self.monotonic.stop)

    def guard(self, *, enabled=True, routed=True, check_url='https://health.invalid'):
        config = {'vpn': {'enabled': enabled, 'check_url': check_url}}
        with patch.object(vpn_session_guard, 'current_proxy', return_value=Mock() if routed else None), patch.object(vpn_session_guard, 'current_vpn', return_value=self.manager):
            guard = vpn_session_guard.VPNJobGuard(self.schedule, config, self.updates)
        with patch.object(guard, "_probe", return_value=None):
            guard.resume()
        return guard

    def probe(self, guard, *, during_read=None, failure=None):
        """Execute one iteration of the existing health-check loop without I/O."""
        guard._stopped.clear()
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)

        def read(size):
            self.assertEqual(size, 256)
            self.clock += 0.071
            if during_read is not None:
                during_read()
            return b'ok'

        response.read.side_effect = read
        opener = Mock()

        def open_response(*args, **kwargs):
            self.clock += 0.040
            if failure is not None:
                raise failure
            return response

        opener.open.side_effect = open_response
        with patch.object(vpn_session_guard, 'http_opener', return_value=opener), patch.object(vpn_session_guard.time, 'perf_counter', side_effect=lambda: self.clock), patch.object(guard._stopped, 'wait', side_effect=lambda timeout: guard._stopped.set()):
            guard._run()
        guard._stopped.clear()
        return opener, response

    def test_times_open_and_read_without_extra_requests(self):
        guard = self.guard()
        self.assertEqual(guard.status(), 'Checking')
        opener, response = self.probe(guard)
        self.assertEqual(guard.status(), 'Connected (111 ms)')
        for _ in range(5):
            self.assertEqual(guard.status(), 'Connected (111 ms)')
        opener.open.assert_called_once()
        self.assertEqual(opener.open.call_args.kwargs, {'timeout': 5.0})
        response.read.assert_called_once_with(256)

    def test_failed_check_clears_successful_latency(self):
        guard = self.guard()
        self.probe(guard)
        self.assertEqual(guard.status(), 'Connected (111 ms)')
        opener, _ = self.probe(guard, failure=OSError('offline'))
        opener.open.assert_called_once()
        self.assertTrue(guard.status().startswith('Disconnected; retrying'))
        self.assertIsNone(guard._latency_ms)

    def test_late_success_cannot_override_connection_loss(self):
        guard = self.guard()
        self.probe(guard, during_read=guard.connection_lost)
        self.assertTrue(guard.status().startswith('Disconnected; retrying'))
        self.assertIsNone(guard._latency_ms)

    def test_late_success_cannot_be_used_for_next_job(self):
        guard = self.guard()

        def next_job():
            self.schedule.number += 1
            self.schedule.boundary += 100

        self.probe(guard, during_read=next_job)
        with patch.object(guard, "_probe", return_value=None):
            guard.resume()
        self.assertEqual(guard.status(), 'Checking')
        self.probe(guard)
        self.assertEqual(guard.status(), 'Connected (111 ms)')

    def test_next_job_waits_for_its_own_successful_check(self):
        guard = self.guard()
        self.probe(guard)
        self.schedule.number += 1
        self.schedule.boundary += 100
        with patch.object(guard, "_probe", return_value=None):
            guard.resume()
        self.assertEqual(guard.status(), 'Checking')

    def test_disabled_unrouted_missing_url_and_idle_status(self):
        self.assertEqual(self.guard(enabled=False).status(), 'Disabled')
        self.assertEqual(self.guard(routed=False).status(), 'Disabled')
        self.assertEqual(self.guard(check_url=None).status(), 'Connected (latency unavailable)')
        guard = self.guard()
        self.schedule.percentage = 0
        self.assertEqual(guard.status(), 'Idle')
        self.schedule.percentage = 100
        self.schedule.kind = 'job-1'
        self.assertEqual(guard.status(), 'Idle')


if __name__ == '__main__':
    unittest.main()
