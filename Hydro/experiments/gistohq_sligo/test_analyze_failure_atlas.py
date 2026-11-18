#!/usr/bin/env python3
import unittest

from analyze_failure_atlas import event_cores, event_metrics, expanded_events


class FailureAtlasTest(unittest.TestCase):
    def test_event_merge_and_expansion(self):
        observed = {x: (10.0 if x in (5, 7, 30) else 0.0) for x in range(40)}
        self.assertEqual(event_cores(observed, 9.0, 3), [(5, 7), (30, 30)])
        self.assertEqual(expanded_events([(5, 7), (30, 30)], 2, 3), [(3, 10), (28, 33)])

    def test_peak_errors_use_event_peak_and_timestamp(self):
        xs = [0, 1, 2, 3]
        observed = {0: 1.0, 1: 4.0, 2: 2.0, 3: 1.0}
        predicted = {0: 1.0, 1: 2.0, 2: 6.0, 3: 1.0}
        result = event_metrics(xs, observed, predicted)
        self.assertEqual(result["peak_timing_error_hours"], 1)
        self.assertAlmostEqual(result["peak_magnitude_error_percent"], 50.0)
        self.assertAlmostEqual(result["volume_error_percent"], 25.0)

    def test_peak_matching_rejects_distant_pulse(self):
        xs = list(range(30))
        observed = {x: (10.0 if x == 20 else 1.0) for x in xs}
        predicted = {x: (20.0 if x == 0 else (8.0 if x == 21 else 1.0)) for x in xs}
        result = event_metrics(xs, observed, predicted, peak_match_radius_hours=3)
        self.assertEqual(result["peak_timing_error_hours"], 1)
        self.assertAlmostEqual(result["peak_magnitude_error_percent"], -20.0)


if __name__ == "__main__":
    unittest.main()
