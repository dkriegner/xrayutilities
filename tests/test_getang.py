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
# Copyright (C) 2014-2018 Dominik Kriegner <dominik.kriegner@gmail.com>

import unittest

import numpy

import xrayutilities as xu


class TestGetAngles(unittest.TestCase):
    chi111 = 70.528779365509308

    @classmethod
    def setUpClass(cls):
        amp = numpy.random.rand()
        fwhm = numpy.random.rand() * 1.5 + 0.1
        cls.x = numpy.arange(-3, 3, 0.0003)
        cls.p = [0., fwhm, amp, 0.]
        cls.p2d = [0., 0., fwhm, fwhm, amp, 0.,
                   2 * numpy.pi * numpy.random.rand()]
        cls.sigma = fwhm / (2 * numpy.sqrt(2 * numpy.log(2)))

    def test_getang111(self):
        chi, phi = xu.analysis.getangles([1, 1, -1], [1, 1, 1], [2, 2, -4])
        self.assertAlmostEqual(chi, self.chi111, places=10)
        self.assertAlmostEqual(phi, 0, places=10)

    def test_getang001(self):
        chi, phi = xu.analysis.getangles([1, 1, 0], [0, 1, 0], [1, 0, 0])
        self.assertAlmostEqual(chi, 45, places=10)
        self.assertAlmostEqual(phi, 0, places=10)
        chi, phi = xu.analysis.getangles([1, 0, 1], [0, 1, 0], [1, 0, 0])
        self.assertAlmostEqual(chi, 90, places=10)
        self.assertAlmostEqual(phi, -45, places=10)

    def test_getunitvector(self):
        hkl = numpy.random.randint(-5, 5, 3)
        chi, phi = xu.analysis.getangles(hkl, [1, 1, 1], [1, -1, 0])
        hklvec = xu.analysis.getunitvector(chi, phi, [1, 1, 1], [1, -1, 0])
        for i in range(3):
            self.assertAlmostEqual(hkl[i] / numpy.linalg.norm(hkl),
                                   hklvec[i])


if __name__ == '__main__':
    unittest.main()
