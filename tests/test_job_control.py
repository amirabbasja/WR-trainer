import threading
import unittest
from unittest.mock import Mock, patch

from workrunner import vpn_session_guard as guard_module
from workrunner.duty_cycle import JobSequence
from workrunner.socks_proxy import SocksProxy
from workrunner.vpn import VPNError


class VPNJobGuardTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        clock = patch.object(guard_module.time, "monotonic", side_effect=lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        self.schedule = JobSequence([
            {"title": "job-0", "time": 10, "max_gpu_utilization": 80},
            {"title": "job-1", "time": 20, "max_gpu_utilization": 70},
            {"title": "job-0", "time": 5, "max_gpu_utilization": 60},
        ], started=100)
        self.updates = Mock()
        self.proxy = SocksProxy("127.0.0.1", 19080)
        self.manager = Mock()
        self.manager.process.poll.return_value = None
        self.config = {"vpn": {"enabled": True, "check_url": "https://example.invalid/probe"},
                       "S3": {"reconnect_delay": 3}}
        self.opener = Mock()
        response = self.opener.open.return_value
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        factory = patch.object(guard_module, "http_opener", return_value=self.opener)
        factory.start()
        self.addCleanup(factory.stop)

    def guard(self, *, proxy=True, enabled=True):
        config = dict(self.config, vpn=dict(self.config["vpn"], enabled=enabled))
        with patch.object(guard_module, "current_proxy", return_value=self.proxy if proxy else None), \
                patch.object(guard_module, "current_vpn", return_value=self.manager):
            return guard_module.VPNJobGuard(self.schedule, config, self.updates)

    def run_one_probe(self, guard):
        guard._stopped.clear()
        with patch.object(guard._stopped, "wait", side_effect=lambda _: guard._stopped.set()):
            guard._run()
        guard._stopped.clear()

    def test_disabled_or_unrouted_does_not_probe_or_restart(self):
        for options in ({"enabled": False}, {"proxy": False}):
            guard = self.guard(**options)
            self.assertFalse(guard.connection_lost())
            self.assertFalse(guard.check())
            self.assertTrue(guard.resume())
            self.assertEqual(guard.status(), "Disabled")
        self.manager.ensure_ready.assert_not_called()
        self.opener.open.assert_not_called()
        self.updates.event.assert_not_called()

    def test_initial_check_blocks_hashing_until_success(self):
        guard = self.guard()
        self.assertTrue(guard.check())
        self.assertEqual(guard.status(), "Checking")
        self.opener.open.side_effect = OSError("offline")
        self.assertFalse(guard.resume())
        self.assertTrue(guard.blocked())
        self.assertEqual(guard.status(), "Disconnected; retrying in 3.0s")
        self.now = 103
        self.opener.open.side_effect = None
        self.assertTrue(guard.resume())
        self.assertFalse(guard.check())
        self.assertTrue(guard.status().startswith("Connected ("))
        request = self.opener.open.call_args.args[0]
        self.assertEqual(request.full_url, self.config["vpn"]["check_url"])
        self.assertEqual(self.opener.open.call_args.kwargs, {"timeout": 5.0})

    def test_connection_loss_retries_and_recovers_in_same_interval(self):
        guard = self.guard()
        self.assertTrue(guard.resume())
        session = Mock()
        guard.attach(session)
        self.assertTrue(guard.connection_lost())
        self.assertTrue(guard.connection_lost())
        self.assertFalse(guard.resume())
        session.close.assert_called_once_with()
        self.updates.event.assert_called_once_with(
            "Job 1 job-0: VPN connection lost; hashing paused; retrying connection", direct=True,
        )
        self.now = 103
        self.assertTrue(guard.resume())
        self.assertFalse(guard.check())
        self.assertEqual(self.schedule.position()[0], 1)

    def test_failed_retries_remain_blocked_and_back_off_without_repeated_alerts(self):
        guard = self.guard()
        self.assertTrue(guard.resume())
        guard.connection_lost()
        self.now = 103
        self.opener.open.side_effect = OSError("offline")
        self.assertFalse(guard.resume())
        attempts = self.opener.open.call_count
        self.now = 104
        self.assertFalse(guard.resume())
        self.assertEqual(self.opener.open.call_count, attempts)
        self.assertTrue(guard.blocked())
        self.now = 106
        self.assertFalse(guard.resume())
        self.assertEqual(self.opener.open.call_count, attempts + 1)
        self.updates.event.assert_called_once()

    def test_retry_status_is_visible_while_connection_check_is_running(self):
        guard = self.guard()
        guard.resume()
        guard.connection_lost()
        self.now = 103
        def checking(*_args, **_kwargs):
            self.assertTrue(guard.blocked())
            self.assertEqual(guard.status(), "Disconnected; retrying")
            return self.opener.open.return_value
        self.opener.open.side_effect = checking
        self.assertTrue(guard.resume())

    def test_dead_core_restarts_on_training_thread_before_same_interval_recovery(self):
        guard = self.guard()
        guard.resume()
        self.manager.ensure_ready.reset_mock()
        self.manager.process.poll.return_value = 1
        self.assertTrue(guard.check())
        self.assertFalse(guard.resume())
        self.now = 103
        def restart():
            self.assertEqual(threading.current_thread(), threading.main_thread())
            self.manager.process.poll.return_value = None
        self.manager.ensure_ready.side_effect = restart
        self.assertTrue(guard.resume())
        self.manager.ensure_ready.assert_called_once_with()
        self.assertFalse(guard.check())

    def test_failed_core_restart_does_not_hash_or_probe(self):
        guard = self.guard()
        self.manager.ensure_ready.side_effect = VPNError("unavailable")
        self.assertFalse(guard.resume())
        self.assertTrue(guard.blocked())
        self.opener.open.assert_not_called()
        self.now = 103
        self.assertFalse(guard.resume())
        self.assertEqual(self.manager.ensure_ready.call_count, 2)
        self.updates.event.assert_called_once()

    def test_session_attached_while_disconnected_is_closed_immediately(self):
        guard = self.guard()
        guard.connection_lost()
        session = Mock()
        guard.attach(session)
        session.close.assert_called_once_with()

    def test_detached_session_is_not_closed_when_another_is_active(self):
        guard = self.guard()
        guard.resume()
        old, active = Mock(), Mock()
        guard.attach(old)
        guard.attach(active)
        guard.detach(old)
        guard.connection_lost()
        old.close.assert_not_called()
        active.close.assert_called_once_with()

    def test_health_monitor_pauses_work_and_only_foreground_can_recover(self):
        guard = self.guard()
        guard.resume()
        self.opener.open.side_effect = OSError("tunnel unavailable")
        self.run_one_probe(guard)
        self.assertTrue(guard.check())
        attempts = self.opener.open.call_count
        self.run_one_probe(guard)
        self.assertEqual(self.opener.open.call_count, attempts)
        self.now = 103
        self.opener.open.side_effect = None
        self.assertTrue(guard.resume())
        self.assertFalse(guard.check())

    def test_old_probe_success_cannot_clear_new_failure(self):
        guard = self.guard()
        guard.resume()
        self.opener.open.return_value.read.side_effect = lambda _: guard.connection_lost()
        self.run_one_probe(guard)
        self.assertTrue(guard.blocked())
        self.assertIsNone(guard._latency_ms)

    def test_old_probe_failure_cannot_invalidate_new_same_interval_connection(self):
        guard = self.guard()
        guard.resume()
        generation = guard._generation
        boundary = self.schedule.current_job()[2]
        guard.connection_lost()
        self.now = 103
        guard.resume()
        self.assertFalse(guard.connection_lost(boundary, generation=generation))
        self.assertFalse(guard.blocked())

    def test_late_probe_failure_is_ignored_after_schedule_boundary(self):
        guard = self.guard()
        guard.resume()
        def late(*_args, **_kwargs):
            self.now = 130
            raise OSError("late result")
        self.opener.open.side_effect = late
        self.run_one_probe(guard)
        self.updates.event.assert_not_called()
        self.assertEqual(guard.status(), "Checking")

    def test_schedule_boundary_does_not_clear_disconnection(self):
        guard = self.guard()
        guard.connection_lost()
        self.now = 130
        self.assertTrue(guard.blocked())
        self.opener.open.side_effect = OSError("still offline")
        self.assertFalse(guard.resume())
        self.assertTrue(guard.blocked())

    def test_side_job_and_pauses_do_not_trigger_retries(self):
        guard = self.guard()
        self.now = 110
        self.assertFalse(guard.connection_lost())
        self.assertFalse(guard.resume())
        self.assertEqual(guard.status(), "Idle")
        self.run_one_probe(guard)
        self.opener.open.assert_not_called()
        self.manager.ensure_ready.assert_not_called()
        guard.schedule = JobSequence([
            {"title": "job-0", "time": 10, "max_gpu_utilization": 0},
        ], started=100)
        self.assertFalse(guard.connection_lost())
        self.assertFalse(guard.resume())

    def test_wait_updates_progress_until_retry_without_waiting_for_job_end(self):
        guard = self.guard()
        guard.connection_lost()
        stopped = Mock()
        stopped.is_set.return_value = False
        waits = []
        def advance(seconds):
            waits.append(seconds)
            self.now += seconds
        stopped.wait.side_effect = advance
        progress = Mock()
        self.assertTrue(guard.wait(stopped, on_progress=progress))
        self.assertEqual(waits, [1.0, 1.0, 1.0])
        self.assertEqual(self.now, 103)
        self.assertGreaterEqual(progress.call_count, 3)
        self.assertTrue(guard.blocked())
        self.assertTrue(guard.resume())

    def test_wait_returns_at_schedule_boundary_and_respects_deadline_and_stop(self):
        guard = self.guard()
        self.now = 109
        guard.connection_lost()
        stopped = Mock()
        stopped.is_set.return_value = False
        stopped.wait.side_effect = lambda seconds: setattr(self, "now", self.now + seconds)
        self.assertTrue(guard.wait(stopped))
        self.assertEqual(self.now, 110)
        self.assertTrue(guard.blocked())
        self.now = 100
        self.assertFalse(guard.wait(stopped, deadline=101.5))
        self.assertEqual(self.now, 101.5)
        external_stop = threading.Event()
        external_stop.set()
        self.assertFalse(guard.wait(external_stop))
        guard.close()
        self.assertFalse(guard.resume())

    def test_stop_during_probe_cannot_resume_hashing(self):
        guard = self.guard()
        self.opener.open.return_value.read.side_effect = lambda _: guard.close()
        self.assertFalse(guard.resume())
        self.assertTrue(guard.blocked())

    def test_missing_health_url_relies_on_core_and_fresh_pool_connection(self):
        self.config["vpn"]["check_url"] = None
        guard = self.guard()
        self.assertTrue(guard.resume())
        self.assertEqual(guard.status(), "Connected (latency unavailable)")
        self.opener.open.assert_not_called()
        guard.connection_lost()
        self.now = 103
        self.assertTrue(guard.resume())
        self.assertEqual(self.manager.ensure_ready.call_count, 2)

    def test_zero_retry_delay_has_a_minimum_to_avoid_busy_loop(self):
        self.config["S3"]["reconnect_delay"] = 0
        guard = self.guard()
        guard.connection_lost()
        self.assertFalse(guard.resume())
        self.now += 0.1
        self.assertTrue(guard.resume())


if __name__ == "__main__":
    unittest.main()
