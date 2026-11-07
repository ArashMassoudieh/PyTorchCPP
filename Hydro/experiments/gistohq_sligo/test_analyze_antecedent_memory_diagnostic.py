import unittest
from datetime import datetime, timedelta, timezone
from analyze_antecedent_memory_diagnostic import daily_window_sum, validate_event_interval

class FixedCalendarTest(unittest.TestCase):
    def setUp(self):
        self.end = datetime(2020, 1, 5, 12, tzinfo=timezone.utc)
        self.midnight = self.end.replace(hour=0)
        self.series = {self.midnight-timedelta(hours=h): 1.0 for h in range(1, 97)}

    def test_midday_uses_only_previous_full_days(self):
        self.series[self.midnight] = 1000
        self.assertEqual(daily_window_sum(self.series, self.end, 2), (48, 1, 2))

    def test_gap_never_substitutes_older_day(self):
        del self.series[self.midnight-timedelta(hours=2)]
        self.assertEqual(daily_window_sum(self.series, self.end, 2), (24, .5, 1))

    def test_empty_history(self):
        self.assertEqual(daily_window_sum({}, self.end, 14), (0, 0, 0))

    def test_invalid_duration(self):
        for days in [0, -1, 1.5, True]:
            with self.assertRaises(ValueError): daily_window_sum({}, self.end, days)

    def test_event_seal_and_order(self):
        validate_event_interval(self.end, self.end)
        seal = datetime(2023, 10, 15, tzinfo=timezone.utc)
        with self.assertRaises(ValueError): validate_event_interval(self.end, seal)
        with self.assertRaises(ValueError): validate_event_interval(self.end, self.end-timedelta(hours=1))
        with self.assertRaises(ValueError): validate_event_interval(self.end.replace(tzinfo=None), self.end)

if __name__ == '__main__': unittest.main()
