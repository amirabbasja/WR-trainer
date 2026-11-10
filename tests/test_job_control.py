import threading
import unittest
from unittest.mock import Mock, patch

from workrunner import vpn_session_guard
from workrunner.duty_cycle import JobSequence
from workrunner.socks_proxy import SocksProxy
from workrunner.vpn import VPNError


class JobEventsTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        clock = patch.object(vpn_session_guard.time, "monotonic", side_effect=lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        self.schedule = JobSequence([{"title": 'job-0', "time": 10, "max_gpu_utilization": 80}, {"title": 'job-1', "time": 20, "max_gpu_utilization": 70}, {"title": 'job-0', "time": 5, "max_gpu_utilization": 60}], started=100)
        self.updates = Mock()
        self.events = vpn_session_guard.JobEvents(self.schedule, self.updates)

    def test_start_finish_and_role_changes_are_emitted_once(self):
        self.events.update()
        self.events.update()
        self.now = 110
        self.events.update()
        self.events.update()
        self.events.finish()
        self.events.finish()
        self.assertEqual([call.args[0] for call in self.updates.event.call_args_list], [
            'Job 1 job-0 starting', 'Job 1 job-0 finished',
            'Job 2 job-1 starting', 'Job 2 job-1 finished',
        ])

    def test_repeating_jobs_keep_cumulative_numbers(self):
        self.events.update()
        self.now = 135
        self.events.update()
        self.assertEqual(self.schedule.position(), (4, 'job-0', 80, 145))
        self.assertEqual(self.updates.event.call_args.args, ('Job 4 job-0 starting',))

    def test_exit_finishes_active_job_without_starting_another(self):
        schedule = JobSequence([{"title": 'job-0', "time": 10, "max_gpu_utilization": 80}, {"title": 'job-0', "time": 0, "max_gpu_utilization": -1}], started=100)
        events = vpn_session_guard.JobEvents(schedule, self.updates)
        events.update()
        self.now = 110
        events.update()
        events.finish()
        self.assertEqual([call.args[0] for call in self.updates.event.call_args_list], [
            'Job 1 job-0 starting', 'Job 1 job-0 finished',
        ])


class VPNJobGuardTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        clock = patch.object(vpn_session_guard.time, "monotonic", side_effect=lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        self.schedule = JobSequence([{"title": 'job-0', "time": 10, "max_gpu_utilization": 80}, {"title": 'job-1', "time": 20, "max_gpu_utilization": 70}, {"title": 'job-0', "time": 5, "max_gpu_utilization": 60}], started=100)
        self.updates = Mock()
        self.proxy = SocksProxy("127.0.0.1", 19080)
        self.manager = Mock()
        self.manager.process.poll.return_value = None
        self.config = {"vpn": {"enabled": True, "check_url": "https://example.invalid/probe"}}

    def guard(self, *, proxy=True, enabled=True):
        config = {"vpn": dict(self.config["vpn"], enabled=enabled)}
        with patch.object(vpn_session_guard, "current_proxy", return_value=self.proxy if proxy else None), \
                patch.object(vpn_session_guard, "current_vpn", return_value=self.manager):
            return vpn_session_guard.VPNJobGuard(self.schedule, config, self.updates)

    def run_one_probe(self, guard, opener):
        with patch.object(vpn_session_guard, "http_opener", return_value=opener) as factory, \
                patch.object(guard._stopped, "wait", side_effect=lambda _: guard._stopped.set()):
            guard._run()
        return factory

    def test_disabled_or_unrouted_jobs_do_not_latch_or_restart_vpn(self):
        for options in ({"enabled": False}, {"proxy": False}):
            with self.subTest(options=options):
                guard = self.guard(**options)
                self.assertFalse(guard.connection_lost())
                self.assertFalse(guard.check())
                self.assertTrue(guard.resume())
        self.manager.ensure_ready.assert_not_called()
        self.updates.event.assert_not_called()

    def test_connection_loss_is_latched_until_boundary_and_alerts_once_directly(self):
        guard = self.guard()
        self.assertTrue(guard.connection_lost())
        self.now = 105
        self.assertTrue(guard.connection_lost())
        self.assertTrue(guard.check())
        self.assertFalse(guard.resume())
        self.manager.ensure_ready.assert_not_called()
        self.updates.event.assert_called_once_with(
            'Job 1 job-0: VPN connection lost; waiting until the job finishes', direct=True,
        )
        self.now = 110
        self.assertFalse(guard.check())

    def test_dead_core_can_restart_at_next_main_job_without_relatching(self):
        guard = self.guard()
        self.assertTrue(guard.resume())
        self.manager.ensure_ready.reset_mock()
        self.manager.process.poll.return_value = 1
        self.assertTrue(guard.check())
        self.now = 105
        self.assertFalse(guard.resume())
        self.manager.ensure_ready.assert_not_called()
        self.now = 130
        self.manager.ensure_ready.side_effect = lambda: setattr(
            self.manager.process.poll, "return_value", None,
        )
        self.assertTrue(guard.resume())
        self.manager.ensure_ready.assert_called_once_with()
        self.assertFalse(guard.check())
        self.updates.event.assert_called_once()

    def test_monitor_does_not_block_next_job_before_dead_core_recovery(self):
        guard = self.guard()
        guard.connection_lost()
        self.now = 130
        self.manager.process.poll.return_value = 1
        opener = Mock()
        opener.open.side_effect = OSError("local core has stopped")
        self.run_one_probe(guard, opener)
        opener.open.assert_not_called()
        self.manager.ensure_ready.side_effect = lambda: setattr(
            self.manager.process.poll, "return_value", None,
        )
        self.assertTrue(guard.resume())
        self.assertFalse(guard.check())
        self.updates.event.assert_called_once()

    def test_auxiliary_failure_cannot_block_next_job_before_foreground_activation(self):
        guard = self.guard()
        guard.resume()
        self.now = 130
        self.manager.process.poll.return_value = 1
        self.assertFalse(guard.blocked())
        self.assertFalse(guard.check())
        self.assertFalse(guard.connection_lost(135))
        self.updates.event.assert_not_called()
        self.manager.ensure_ready.side_effect = lambda: setattr(
            self.manager.process.poll, "return_value", None,
        )
        self.assertTrue(guard.resume())
        self.assertTrue(guard.connection_lost(135))
        self.assertTrue(guard.blocked())
        self.updates.event.assert_called_once()

    def test_auxiliary_failure_during_recovery_does_not_latch_new_job(self):
        guard = self.guard()
        guard.resume()
        self.now = 130

        def restart():
            self.assertFalse(guard.connection_lost(135))

        self.manager.ensure_ready.side_effect = restart
        self.assertTrue(guard.resume())
        self.assertFalse(guard.blocked())
        self.updates.event.assert_not_called()

    def test_connection_loss_closes_attached_session_once_to_interrupt_io(self):
        guard = self.guard()
        session = Mock()
        guard.attach(session)
        guard.connection_lost()
        guard.connection_lost()
        session.close.assert_called_once_with()
        self.updates.event.assert_called_once()

    def test_session_attached_during_blocked_job_is_closed_immediately(self):
        guard = self.guard()
        guard.connection_lost()
        session = Mock()
        guard.attach(session)
        session.close.assert_called_once_with()

    def test_detached_session_is_not_closed_when_another_session_is_active(self):
        guard = self.guard()
        old_session, active_session = Mock(), Mock()
        guard.attach(old_session)
        guard.attach(active_session)
        guard.detach(old_session)
        guard.connection_lost()
        old_session.close.assert_not_called()
        active_session.close.assert_called_once_with()

    def test_failed_restart_blocks_new_job_and_sends_a_new_single_alert(self):
        guard = self.guard()
        guard.connection_lost()
        self.now = 130
        self.manager.ensure_ready.side_effect = VPNError("unavailable")
        self.assertFalse(guard.resume())
        self.assertFalse(guard.resume())
        self.manager.ensure_ready.assert_called_once_with()
        self.assertEqual(self.updates.event.call_count, 2)
        self.assertIn('Job 3 job-0', self.updates.event.call_args.args[0])

    def test_loss_of_running_remote_tunnel_latches_job_through_captured_route(self):
        guard = self.guard()
        self.assertTrue(guard.resume())
        opener = Mock()
        opener.open.side_effect = OSError("remote tunnel unavailable")
        factory = self.run_one_probe(guard, opener)
        factory.assert_called_once_with(self.proxy)
        self.assertIsNone(self.manager.process.poll())
        self.assertTrue(guard.check())
        self.updates.event.assert_called_once()
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, self.config["vpn"]["check_url"])
        self.assertEqual(opener.open.call_args.kwargs, {"timeout": 5.0})

    def test_late_probe_failure_does_not_block_the_following_job(self):
        guard = self.guard()
        self.assertTrue(guard.resume())
        opener = Mock()

        def crosses_boundary(*_args, **_kwargs):
            self.now = 130
            raise OSError("late probe result")

        opener.open.side_effect = crosses_boundary
        self.run_one_probe(guard, opener)
        self.assertFalse(guard.check())
        self.updates.event.assert_not_called()

    def test_side_job_and_pauses_do_not_latch_main_vpn_failures(self):
        guard = self.guard()
        self.now = 110
        self.assertFalse(guard.connection_lost())
        opener = Mock()
        self.run_one_probe(guard, opener)
        opener.open.assert_not_called()
        self.updates.event.assert_not_called()
        paused = JobSequence([{"title": 'job-0', "time": 10, "max_gpu_utilization": 0}, {"title": 'job-0', "time": 10, "max_gpu_utilization": 80}], started=100)
        guard.schedule = paused
        self.now = 100
        self.assertFalse(guard.connection_lost())

    def test_blocked_job_does_not_probe_or_restart_before_its_end(self):
        guard = self.guard()
        guard.connection_lost()
        self.now = 109
        opener = Mock()
        self.run_one_probe(guard, opener)
        opener.open.assert_not_called()
        self.assertFalse(guard.resume())
        self.manager.ensure_ready.assert_not_called()

    def test_wait_updates_countdown_and_keeps_original_boundary(self):
        guard = self.guard()
        self.now = 107
        guard.connection_lost()
        stopped = Mock()
        stopped.is_set.return_value = False
        waits = []
        progress = Mock()

        def advance(seconds):
            waits.append(seconds)
            self.now += seconds

        stopped.wait.side_effect = advance
        self.assertTrue(guard.wait(stopped, on_progress=progress))
        self.assertEqual(waits, [1.0, 1.0, 1.0])
        self.assertEqual(progress.call_count, 3)
        self.assertEqual(self.now, 110)
        self.assertEqual(self.schedule.position(), (2, 'job-1', 70, 130))
        self.manager.ensure_ready.assert_not_called()

    def test_wait_obeys_deadline_and_external_stop(self):
        guard = self.guard()
        guard.connection_lost()
        stopped = Mock()
        stopped.is_set.return_value = False
        stopped.wait.side_effect = lambda seconds: setattr(self, "now", self.now + seconds)
        self.assertFalse(guard.wait(stopped, deadline=101.5))
        self.assertEqual(self.now, 101.5)
        stopped.wait.assert_called_with(0.5)
        external_stop = threading.Event()
        external_stop.set()
        self.assertFalse(guard.wait(external_stop))


if __name__ == "__main__":
    unittest.main()
