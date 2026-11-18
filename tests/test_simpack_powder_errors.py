# This file is part of xrayutilities, distributed under the GNU GPL v2 or later.
# Copyright (C) 2026 CAOShurong

import functools
import unittest
from unittest.mock import patch

import numpy
import xrayutilities as xu
from xrayutilities.simpack.powder import PowderDiffraction


class TestPowderWorkerErrors(unittest.TestCase):
    def setUp(self):
        with patch.object(xu.config, "NTHREADS", 2):
            self.pd = PowderDiffraction(
                xu.materials.Si, tt_cutoff=60, enable_simulation=True
            )
        self.addCleanup(self.pd.close)
        self.workers = [thread for thread, _, _ in self.pd.threads]
        self.processes = [manager._process for manager in self.pd.managers]
        # Bound a regression without replacing the real result queue.
        queue = self.pd.output_queue
        guard = patch.object(
            queue, "get", side_effect=functools.partial(queue.get, timeout=5)
        )
        guard.start()
        self.addCleanup(guard.stop)

    def tearDown(self):
        self.pd.close()
        self.assertFalse(any(worker.is_alive() for worker in self.workers))
        self.assertFalse(any(proc.is_alive() for proc in self.processes))

    def test_successful_calculation(self):
        angles = numpy.linspace(20, 60, 801)
        expected = self.pd.Convolve(angles, mode="local")
        actual = self.pd.Convolve(angles)
        numpy.testing.assert_allclose(actual, expected)

    def assert_calculation_error_and_recovery(self, angles):
        expected = self.pd.Convolve(angles, mode="local")
        self.pd.update_settings(
            {"emission": {"emiss_intensities": (1.0, 0.5, 0.2)}}
        )
        with self.assertRaisesRegex(ValueError, "broadcast"):
            self.pd.Convolve(angles)
        self.assertTrue(self.pd.output_queue.empty())
        self.assertTrue(all(worker.is_alive() for worker in self.workers))
        self.pd.update_settings(
            {"emission": {"emiss_intensities": (1.0, 0.5)}}
        )
        numpy.testing.assert_allclose(self.pd.Convolve(angles), expected)

    def test_all_workers_fail(self):
        self.assert_calculation_error_and_recovery(numpy.linspace(20, 60, 801))

    def test_one_worker_fails(self):
        # Only the first worker has a reflection in this interval.
        self.assert_calculation_error_and_recovery(numpy.linspace(27, 30, 61))

    def test_parameter_update_error(self):
        angles = numpy.linspace(20, 60, 801)
        expected = self.pd.Convolve(angles, mode="local")
        # The dispatch thread can also fail before it starts the calculation.
        with (
            patch.object(
                self.pd.conv_handlers[0],
                "update_parameters",
                side_effect=ValueError("parameter update failed"),
            ),
            self.assertRaisesRegex(ValueError, "parameter update failed"),
        ):
            self.pd.Convolve(angles)
        self.assertTrue(self.pd.output_queue.empty())
        self.assertTrue(all(worker.is_alive() for worker in self.workers))
        numpy.testing.assert_allclose(self.pd.Convolve(angles), expected)


if __name__ == "__main__":
    unittest.main()
