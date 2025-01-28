# This file is part of xrayutilities.
#
# xrayutilities is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2025 Dominik Kriegner <dominik.kriegner@gmail.com>

import time
import unittest

import numpy as np
import xrayutilities as xu
from scipy.stats import linregress, uniform
import plotly.graph_objects as go


# get maximum number of threads
# if this is -1 no openmp is available and we skip the test class
max_threads = xu.cxrayutilities.get_max_openmp_threads()


@unittest.skipIf(max_threads == -1, "extension was not compiled with OpenMP")
class TestParallelPerformance(unittest.TestCase):
    def setUp(self):
        """Set up the HXRD object and generate random angles."""
        mat = xu.materials.Si  # Example material
        self.hxrd = xu.HXRD(mat.Q(1, 1, 0), mat.Q(0, 0, 1))
        self.num_angles = 10000000  # Number of random angles to test
        self.tt_angles = uniform.rvs(size=self.num_angles, loc=0, scale=80)
        self.pp_angles = uniform.rvs(size=self.num_angles, loc=0, scale=140)

    def test_parallel_performance(self):
        """Test the performance scaling of parallel processing.

        Tests angle to Q-space conversion performance with varying numbers of
        threads. Calculates speedup and efficiency metrics for different thread
        counts. Checks that parallel processing achieves reasonable efficiency
        thresholds.
        """
        if max_threads < 2:
            self.skipTest("OpenMP available, but not enough allowed threads")
            return
        times = {}
        n_threads_tested = list(range(1, min(13, max_threads+1)))
        for nthreads in n_threads_tested:
            xu.config.NTHREADS = nthreads
            start_time = time.time()
            qx, qy, qz = self.hxrd.Ang2Q(self.pp_angles, self.tt_angles)
            end_time = time.time()
            times[nthreads] = end_time - start_time
            print(f"Nthreads: {nthreads}, Time: {times[nthreads]:.4f} seconds")

        # Calculate Efficiency
        efficiencies = {}
        time_1 = times[1]  # Time for 1 thread
        for nthreads in n_threads_tested:
            speedup = time_1 / times[nthreads]
            efficiency = speedup / nthreads
            efficiencies[nthreads] = efficiency
            print(f"Nthreads: {nthreads}, Speedup: {speedup:.2f}, "
                  f"Efficiency: {efficiency:.2f}")

        # Average Efficiency (excluding 1 thread, as efficiency=1.0)
        avg_efficiency = np.mean(list(efficiencies.values())[1:])
        print(f"Average Efficiency (excluding 1 thread): {avg_efficiency:.2f}")

        # Assertions based on the average efficiency
        self.assertGreater(
            avg_efficiency, 0.25, "Average efficiency should be reasonable")
        self.assertGreater(
            efficiencies[2], 0.5, "Efficiency at 2 threads should be high.")


if __name__ == '__main__':
    unittest.main()
