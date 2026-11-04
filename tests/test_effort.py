"""Deterministic checks for timed effort schedules and lifecycle shutdown."""

import math
import threading
import time
import unittest
from unittest.mock import Mock, patch

from workrunner import effort


class JobSequenceValidationTests(unittest.TestCase):
    def test_job_dictionaries_are_copied_without_aliasing_the_input(self):
        original = [
            {"title": "job-0", "time": 60, "max_gpu_utilization": 80},
            {"title": "job-1", "time": 120.5, "max_gpu_utilization": 0},
            {"title": "job-0", "time": 0, "max_gpu_utilization": -1},
        ]
        result = effort.normalize_job_sequence(original)
        self.assertEqual(result, original)
        original[0]["max_gpu_utilization"] = 100
        self.assertEqual(result[0]["max_gpu_utilization"], 80)

    def test_sequence_must_not_be_empty(self):
        with self.assertRaisesRegex(ValueError, "job_sequence"):
            effort.normalize_job_sequence([])

    def test_rejects_legacy_arrays_tolerance_and_invalid_fields_without_values(self):
        valid = {"title": "job-0", "time": 60, "max_gpu_utilization": 80}
        invalid = [None, "secret", {}, (), [None], [["main", 60, 80]],
                   [["job-0", 60, 80]], [["job-0", 60, 80, 5]],
                   [dict(valid, tolerance=5)], [dict(valid, secret=1)]]
        for field, values in (
            ("title", ("main", "side-job", "secret", True, None, [], "JOB-0")),
            ("time", (True, False, "secret", -1, 0, math.nan, math.inf, 10 ** 1000)),
            ("max_gpu_utilization", (-2, 101, True, 80.0, "secret", None)),
        ):
            invalid.extend([dict(valid, **{field: value})] for value in values)
        invalid.extend([{k: v for k, v in valid.items() if k != missing}] for missing in valid)
        invalid.append([dict(valid, time=1e308), dict(valid, time=1e308)])
        for value in invalid:
            with self.subTest(value=repr(value)[:80]), self.assertRaisesRegex(ValueError, "job_sequence") as caught:
                effort.normalize_job_sequence(value)
            self.assertNotIn("secret", str(caught.exception))


class JobSequenceTests(unittest.TestCase):
    def schedule(self, entries=None, **kwargs):
        if entries is None:
            entries = [
                {"title": "job-0", "time": 60, "max_gpu_utilization": 80},
                {"title": "job-0", "time": 120, "max_gpu_utilization": 70},
                {"title": "job-0", "time": 180, "max_gpu_utilization": 10},
            ]
        return effort.JobSequence(entries, started=100, **kwargs)

    def test_intervals_use_wall_clock_and_repeat_at_exact_boundaries(self):
        schedule = self.schedule()
        for now, expected in [
            (100, (80, 160)), (159.999, (80, 160)),
            (160, (70, 280)), (280, (10, 460)),
            (460, (80, 520)), (100 + 360 * 100 + 70, (70, 36280)),
        ]:
            with self.subTest(now=now), patch.object(effort.time, "monotonic", return_value=now):
                self.assertEqual(schedule.current(), expected)
                self.assertFalse(schedule.ended())

    def test_utilization_stays_at_the_configured_value_across_time_and_cycles(self):
        sequence = self.schedule([
            {"title": "job-0", "time": 10, "max_gpu_utilization": 70},
            {"title": "job-1", "time": 10, "max_gpu_utilization": 30},
        ])
        for now, expected in [(100, ("job-0", 70, 110)), (104.9, ("job-0", 70, 110)),
                              (105, ("job-0", 70, 110)), (110, ("job-1", 30, 120)),
                              (120, ("job-0", 70, 130))]:
            with self.subTest(now=now), patch.object(effort.time, "monotonic", return_value=now):
                self.assertEqual(sequence.current_job(), expected)

    def test_mixed_roles_share_one_repeating_timeline(self):
        sequence = self.schedule([{"title": 'job-0', "time": 10, "max_gpu_utilization": 80}, {"title": 'job-1', "time": 20, "max_gpu_utilization": 30}])
        for now, expected in [
            (100, ('job-0', 80, 110)), (110, ('job-1', 30, 130)),
            (130, ('job-0', 80, 140)), (140, ('job-1', 30, 160)),
        ]:
            with self.subTest(now=now), patch.object(effort.time, "monotonic", return_value=now):
                self.assertEqual(sequence.current_job(), expected)

    def test_side_job_pause_and_exit_obey_the_shared_timeline(self):
        sequence = self.schedule([{"title": 'job-1', "time": 10, "max_gpu_utilization": 0}, {"title": 'job-0', "time": 20, "max_gpu_utilization": 70}, {"title": 'job-1', "time": 0, "max_gpu_utilization": -1}])
        stopped = Mock()
        stopped.is_set.return_value = False
        with patch.object(effort.time, "monotonic", side_effect=[102, 110]):
            self.assertTrue(sequence.wait(stopped))
        stopped.wait.assert_called_once_with(8)
        with patch.object(effort.time, "monotonic", return_value=130):
            self.assertEqual(sequence.current_job(), ('job-1', -1, math.inf))
            self.assertFalse(sequence.wait(stopped))
        stopped.set.assert_called_once()

    def test_exit_deadline_is_absolute_and_optional(self):
        self.assertIsNone(self.schedule().exit_deadline)
        self.assertEqual(self.schedule([{"title": 'job-0', "time": 60, "max_gpu_utilization": 80}, {"title": 'job-0', "time": 120, "max_gpu_utilization": 0}, {"title": 'job-0', "time": 0, "max_gpu_utilization": -1}]).exit_deadline, 280)
        self.assertEqual(self.schedule([{"title": 'job-0', "time": 0, "max_gpu_utilization": -1}]).exit_deadline, 100)

    def test_fractional_boundary_with_large_origin_does_not_repeat_a_pause(self):
        schedule = effort.JobSequence([{"title": 'job-0', "time": 0.1, "max_gpu_utilization": 0}, {"title": 'job-0', "time": 0.2, "max_gpu_utilization": 80}], started=1e6)
        with patch.object(effort.time, "monotonic", return_value=1e6 + 0.1):
            percentage, boundary = schedule.current()
            self.assertEqual(percentage, 80)
            self.assertGreater(boundary, 1e6 + 0.1)
        exiting = effort.JobSequence([{"title": 'job-0', "time": 0.1, "max_gpu_utilization": 80}, {"title": 'job-0', "time": 0, "max_gpu_utilization": -1}], started=1e6)
        with patch.object(effort.time, "monotonic", return_value=exiting.exit_deadline):
            self.assertTrue(exiting.ended())

    def test_intervals_below_clock_precision_still_return_a_future_boundary(self):
        schedule = effort.JobSequence([{"title": 'job-0', "time": 1e-30, "max_gpu_utilization": 0}, {"title": 'job-0', "time": 1e-30, "max_gpu_utilization": 80}], started=1e6)
        with patch.object(effort.time, "monotonic", return_value=1e6):
            self.assertGreater(schedule.current()[1], 1e6)

    def test_exit_is_terminal_and_its_duration_is_ignored(self):
        schedule = self.schedule([{"title": 'job-0', "time": 60, "max_gpu_utilization": 80}, {"title": 'job-0', "time": 999, "max_gpu_utilization": -1}, {"title": 'job-0', "time": 10, "max_gpu_utilization": 100}])
        for now in (160, 10000):
            with patch.object(effort.time, "monotonic", return_value=now):
                self.assertEqual(schedule.current(), (-1, math.inf))
                self.assertTrue(schedule.ended())

    def test_initial_zero_duration_exit(self):
        with patch.object(effort.time, "monotonic", return_value=100):
            self.assertTrue(self.schedule([{"title": 'job-0', "time": 0, "max_gpu_utilization": -1}]).ended())

    def test_wait_sleeps_through_pause_then_resumes_without_setting_stop(self):
        schedule = self.schedule([{"title": 'job-0', "time": 10, "max_gpu_utilization": 0}, {"title": 'job-0', "time": 20, "max_gpu_utilization": 50}])
        stopped = Mock()
        stopped.is_set.return_value = False
        with patch.object(effort.time, "monotonic", side_effect=[102, 110]):
            self.assertTrue(schedule.wait(stopped))
        stopped.wait.assert_called_once_with(8)
        stopped.set.assert_not_called()

    def test_pause_can_span_multiple_intervals(self):
        schedule = self.schedule([{"title": 'job-0', "time": 10, "max_gpu_utilization": 0}, {"title": 'job-0', "time": 20, "max_gpu_utilization": 0}, {"title": 'job-0', "time": 30, "max_gpu_utilization": 60}])
        stopped = Mock()
        stopped.is_set.return_value = False
        with patch.object(effort.time, "monotonic", side_effect=[100, 110, 130]):
            self.assertTrue(schedule.wait(stopped))
        self.assertEqual([call.args[0] for call in stopped.wait.call_args_list], [10, 20])

    def test_deadline_bounds_pause(self):
        schedule = self.schedule([{"title": 'job-0', "time": 10, "max_gpu_utilization": 0}, {"title": 'job-0', "time": 20, "max_gpu_utilization": 80}])
        stopped = Mock()
        stopped.is_set.return_value = False
        with patch.object(effort.time, "monotonic", side_effect=[102, 105]):
            self.assertFalse(schedule.wait(stopped, deadline=105))
        stopped.wait.assert_called_once_with(3)

    def test_stop_interrupts_pause_and_exit_sets_stop(self):
        stopped = threading.Event()
        stopped.set()
        with patch.object(effort.time, "monotonic") as clock:
            self.assertFalse(self.schedule([{"title": 'job-0', "time": 10, "max_gpu_utilization": 0}]).wait(stopped))
        clock.assert_not_called()
        stopped.clear()
        with patch.object(effort.time, "monotonic", return_value=160):
            self.assertFalse(self.schedule([{"title": 'job-0', "time": 60, "max_gpu_utilization": 0}, {"title": 'job-0', "time": 0, "max_gpu_utilization": -1}]).wait(stopped))
        self.assertTrue(stopped.is_set())

    def test_timer_stops_operations_while_caller_is_waiting(self):
        stopped = threading.Event()
        schedule = effort.JobSequence([{"title": 'job-0', "time": 0.03, "max_gpu_utilization": 0}, {"title": 'job-0', "time": 0, "max_gpu_utilization": -1}])
        with schedule.stopping(stopped):
            self.assertTrue(stopped.wait(1))
            self.assertTrue(schedule.ended())

    def test_no_exit_means_no_timer_and_no_forced_stop(self):
        stopped = threading.Event()
        with patch.object(effort.threading, "Timer") as timer:
            with self.schedule().stopping(stopped):
                self.assertFalse(stopped.is_set())
        timer.assert_not_called()
        self.assertFalse(stopped.is_set())

    def test_timer_is_cancelled_when_run_finishes_early(self):
        stopped = threading.Event()
        with patch.object(effort.threading, "Timer") as timer, \
                patch.object(effort.time, "monotonic", return_value=105):
            with self.schedule([{"title": 'job-0', "time": 10, "max_gpu_utilization": 80}, {"title": 'job-0', "time": 0, "max_gpu_utilization": -1}]).stopping(stopped):
                timer.assert_called_once_with(5, stopped.set)
                timer.return_value.start.assert_called_once()
        timer.return_value.cancel.assert_called_once()
        timer.return_value.join.assert_called_once()
        self.assertFalse(stopped.is_set())

    def test_exit_beyond_platform_timeout_uses_cancellable_bounded_waits(self):
        stopped = threading.Event()
        schedule = effort.JobSequence([{"title": 'job-0', "time": 0.04, "max_gpu_utilization": 80}, {"title": 'job-0', "time": 0, "max_gpu_utilization": -1}])
        with patch.object(effort.threading, "TIMEOUT_MAX", 0.01), schedule.stopping(stopped):
            self.assertTrue(stopped.wait(1))
            self.assertTrue(schedule.ended())

    def test_long_exit_timer_cancellation_does_not_wait_for_deadline(self):
        stopped = threading.Event()
        schedule = effort.JobSequence([{"title": 'job-0', "time": 60, "max_gpu_utilization": 80}, {"title": 'job-0', "time": 0, "max_gpu_utilization": -1}])
        started = time.monotonic()
        with patch.object(effort.threading, "TIMEOUT_MAX", 0.001), schedule.stopping(stopped):
            # Enter the timer callback's capped wait loop before cancellation.
            time.sleep(0.01)
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertFalse(stopped.is_set())

    def test_initial_exit_sets_stop_without_starting_timer(self):
        stopped = threading.Event()
        with patch.object(effort.threading, "Timer") as timer, \
                patch.object(effort.time, "monotonic", return_value=100):
            with self.schedule([{"title": 'job-0', "time": 0, "max_gpu_utilization": -1}]).stopping(stopped):
                self.assertTrue(stopped.is_set())
        timer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
