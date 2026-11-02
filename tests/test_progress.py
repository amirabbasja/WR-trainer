import unittest

from LRAEP_trainer.progress import job_status

class JobStatusTests(unittest.TestCase):
    def test_main_rate_is_a_number_without_units(self):
        self.assertEqual(job_status(1, "main", 80, 60, 123.45),
                         "1 | main | 80% | 00:01:00 | 123.45")

    def test_side_speed_is_the_epoch_number(self):
        self.assertEqual(job_status(2, "side", 50, 30, 4),
                         "2 | side | 50% | 00:00:30 | 4")

    def test_zero_and_missing_values_remain_five_fields(self):
        self.assertEqual(job_status(3, "main", None, None, None),
                         "3 | main | N/A | --:--:-- | 0")
        self.assertEqual(job_status(3, "side-job", 0, -1, 0),
                         "3 | side | 0% | 00:00:00 | 0")

    def test_nonfinite_remaining_time_is_displayed_without_crashing(self):
        self.assertEqual(job_status(3, "main", 0, float("inf"), float("nan")),
                         "3 | main | 0% | --:--:-- | 0")

    def test_long_job_time_does_not_wrap_after_a_day(self):
        self.assertEqual(job_status(4, "main", 1, 90061, 1000.0),
                         "4 | main | 1% | 25:01:01 | 1000")


