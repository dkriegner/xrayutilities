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
# Copyright (C) 2014-2020 Dominik Kriegner <dominik.kriegner@gmail.com>

import os.path
import unittest

import numpy
import xrayutilities as xu

xu.config.VERBOSITY = 0  # make no output during test
datadir = os.path.join(os.path.dirname(__file__), "data")
fullfilename = os.path.join(datadir, "detalign.xrdml.bz2")


@unittest.skipIf(
    not os.path.isfile(fullfilename),
    "additional test data needed (http://xrayutilities.sf.io)",
)
class TestLinear_calib(unittest.TestCase):
    pw0 = 1.4813e-04
    cch0 = 633.90
    tilt0 = -0.498

    @classmethod
    def setUpClass(cls):
        tt, det = xu.io.getxrdml_scan(fullfilename)
        cls.ang = tt[:, 639]
        cls.spectra = det

    def test_linear_calib(self):
        pwidth, cch, tilt = xu.analysis.linear_detector_calib(
            self.ang, self.spectra
        )
        self.assertAlmostEqual(numpy.abs(pwidth), self.pw0, places=7)
        self.assertAlmostEqual(cch, self.cch0, places=1)
        self.assertAlmostEqual(tilt, self.tilt0, places=2)


if __name__ == "__main__":
    unittest.main()
