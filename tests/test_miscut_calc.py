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
# Copyright (C) 2016 Dominik Kriegner <dominik.kriegner@gmail.com>

import unittest

import numpy

import xrayutilities as xu

xu.config.VERBOSITY = 0  # make no output during test


class TestMiscutCalc(unittest.TestCase):

    def test_miscut4(self):
        phi = numpy.asarray((0, 45, 90, 135))
        miscut = numpy.random.rand()
        azimuth = numpy.random.rand() * 360
        aom0 = numpy.random.rand() * 45
        aom = miscut * numpy.cos(numpy.radians(phi - azimuth)) + aom0

        om0, p0, mc = xu.analysis.miscut_calc(phi, aom, plot=False)
        self.assertAlmostEqual(om0, aom0, places=5)
        self.assertAlmostEqual(mc, miscut, places=5)
        self.assertAlmostEqual(p0, azimuth, places=5)

    def test_miscut3(self):
        phi = numpy.asarray((0, 60, 120))
        miscut = numpy.random.rand()
        azimuth = numpy.random.rand() * 360
        aom0 = numpy.random.rand() * 45
        aom = miscut * numpy.cos(numpy.radians(phi - azimuth)) + aom0

        om0, p0, mc = xu.analysis.miscut_calc(phi, aom, plot=False)
        self.assertAlmostEqual(om0, aom0, places=5)
        self.assertAlmostEqual(mc, miscut, places=5)
        self.assertAlmostEqual(p0, azimuth, places=5)

    def test_miscut2pom0(self):
        phi = numpy.asarray((0, 90))
        miscut = numpy.random.rand()
        azimuth = numpy.random.rand() * 360
        aom0 = numpy.random.rand() * 45
        aom = miscut * numpy.cos(numpy.radians(phi - azimuth)) + aom0

        om0, p0, mc = xu.analysis.miscut_calc(phi, aom, omega0=aom0,
                                              plot=False)
        self.assertAlmostEqual(om0, aom0, places=5)
        self.assertAlmostEqual(mc, miscut, places=5)
        self.assertAlmostEqual(p0, azimuth, places=5)


if __name__ == '__main__':
    unittest.main()
